from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from attention_sinks import (
    FLASH_ATTN_AVAILABLE as _FLASH_ATTN_AVAILABLE,
    cached_attention_with_sink,
    flash_attention_with_sink,
    sdpa_attention_with_sink,
)

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
except Exception:
    LigerFusedLinearCrossEntropyLoss = None
    _LIGER_AVAILABLE = False


@dataclass
class MiniPLEConfig:
    """Configuration for the 146.5M MiniPLE model with learned attention sinks."""

    architecture_version: int = 2
    vocab_size: int = 32_768
    max_seq_len: int = 2_048

    n_layers: int = 18
    d_model: int = 640
    d_ff: int = 3_072

    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    local_window: int = 256
    attention_pattern: str = "GGLLLGLLLGLLLGLLLG"
    global_nope: bool = True  # Local layers keep RoPE; global layers omit it.

    # Gemma-4-inspired per-layer token memory.
    ple_dim: int = 64

    dropout: float = 0.0
    rope_base: float = 10_000.0
    rms_eps: float = 1e-6

    # auto: FlashAttention-2 on compatible CUDA, else PyTorch SDPA
    # flash: require FlashAttention-2
    # sdpa: force PyTorch SDPA
    attn_backend: str = "auto"

    # auto: Liger on CUDA when installed; otherwise PyTorch's memory-efficient
    # linear CE (when LinearCrossEntropyOptions is available); otherwise chunked PyTorch.
    # torch: require the memory-efficient PyTorch linear CE API.
    # liger: require Liger Kernel
    # chunked: portable chunked logits + CE
    loss_backend: str = "auto"
    ce_chunk_size: int = 2048

    gradient_checkpointing: bool = False

    @classmethod
    def from_checkpoint(cls, config: dict) -> "MiniPLEConfig":
        if config.get("architecture_version") != 2:
            raise ValueError(
                "Checkpoint architecture is incompatible with MiniPLE v2. "
                "Use the source snapshot recorded for that checkpoint."
            )
        cfg = cls(**config)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.architecture_version != 2:
            raise ValueError("architecture_version must be 2")
        for name in ("vocab_size", "max_seq_len", "n_layers", "d_model", "d_ff",
                     "n_heads", "n_kv_heads", "head_dim", "local_window",
                     "ple_dim", "ce_chunk_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_heads * self.head_dim > self.d_model:
            raise ValueError("attention inner width must not exceed d_model in this repo")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if len(self.attention_pattern) != self.n_layers or set(self.attention_pattern) - {"L", "G"}:
            raise ValueError("attention_pattern must contain one L or G per layer")
        if not isinstance(self.global_nope, bool):
            raise ValueError("global_nope must be a boolean")
        if self.attn_backend not in {"auto", "flash", "sdpa"}:
            raise ValueError("attn_backend must be one of: auto, flash, sdpa")
        if self.loss_backend not in {"auto", "torch", "liger", "chunked"}:
            raise ValueError("loss_backend must be one of: auto, torch, liger, chunked")


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cache. Input layout is [B, T, H, D]."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        cos = self.cos[:t].to(device=x.device, dtype=x.dtype)[None, :, None, :]
        sin = self.sin[:t].to(device=x.device, dtype=x.dtype)[None, :, None, :]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        return torch.stack((y_even, y_odd), dim=-1).flatten(-2)


class GQAAttention(nn.Module):
    def __init__(self, cfg: MiniPLEConfig, is_local: bool):
        super().__init__()
        self.cfg = cfg
        self.is_local = is_local
        self.kv_repeat = cfg.n_heads // cfg.n_kv_heads

        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.q_width = cfg.n_heads * cfg.head_dim
        self.kv_width = cfg.n_kv_heads * cfg.head_dim
        self.qkv_proj = nn.Linear(
            cfg.d_model,
            self.q_width + 2 * self.kv_width,
            bias=False,
        )

        # QK-Norm applies to every layer; RoPE is optional on global layers.
        self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_eps)
        self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_eps)
        self.rope = (
            RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
            if is_local or not cfg.global_nope else None
        )
        self.sink_logits = nn.Parameter(torch.zeros(cfg.n_heads))

        self.out_proj = nn.Linear(self.q_width, cfg.d_model, bias=False)

    def _use_flash(self, x: torch.Tensor) -> bool:
        if self.cfg.attn_backend == "sdpa":
            return False
        if self.cfg.attn_backend == "flash":
            if not _FLASH_ATTN_AVAILABLE:
                raise RuntimeError(
                    "attn_backend='flash' requested but flash-attn is not installed"
                )
            if not x.is_cuda:
                raise RuntimeError("FlashAttention-2 requires CUDA in this repo")
            if x.dtype not in {torch.float16, torch.bfloat16}:
                raise RuntimeError("FlashAttention-2 requires BF16/FP16 autocast")
            return True
        return (_FLASH_ATTN_AVAILABLE and x.is_cuda
                and x.dtype in {torch.float16, torch.bfloat16})

    def forward(self, x: torch.Tensor, local_mask: Optional[torch.Tensor],
                kv_cache=None, cache_lengths=None) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.norm(x)

        # FlashAttention native layout: [B, T, H, D].
        qkv = self.qkv_proj(h)
        q, k, v = qkv.split(
            (self.q_width, self.kv_width, self.kv_width), dim=-1
        )
        q = q.view(b, t, self.cfg.n_heads, self.cfg.head_dim)
        k = k.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim)
        v = v.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim)

        # Older RMSNorm implementations can promote activations with FP32 weights.
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)
        if kv_cache is not None:
            out = cached_attention_with_sink(
                q, k, v, kv_cache[0], kv_cache[1], cache_lengths, self.sink_logits,
                window_left=self.cfg.local_window - 1 if self.is_local else -1,
                rotary_cos=self.rope.cos if self.rope is not None else None,
                rotary_sin=self.rope.sin if self.rope is not None else None,
            )
            return self.out_proj(out.reshape(b, t, self.q_width))

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        if self._use_flash(q):
            # FlashAttention supports GQA without physically duplicating K/V heads.
            window = (self.cfg.local_window - 1, 0) if self.is_local else (-1, -1)
            dropout_p = self.cfg.dropout if self.training else 0.0
            out = flash_attention_with_sink(q, k, v, self.sink_logits, dropout_p, window[0])
            out = out.reshape(b, t, self.q_width)
        else:
            # Portable SDPA fallback. K/V expansion is less memory-efficient than the
            # FlashAttention path but works on CUDA, MPS, and CPU backends.
            q_sdpa = q.transpose(1, 2)  # [B, Hq, T, D]
            k_sdpa = k.transpose(1, 2).repeat_interleave(self.kv_repeat, dim=1)
            v_sdpa = v.transpose(1, 2).repeat_interleave(self.kv_repeat, dim=1)

            if self.is_local and local_mask is None:
                raise RuntimeError("local attention requires local_mask on SDPA fallback")
            out = sdpa_attention_with_sink(
                q_sdpa, k_sdpa, v_sdpa, self.sink_logits,
                mask=local_mask if self.is_local else None,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )
            out = out.transpose(1, 2).contiguous().view(b, t, self.q_width)

        return self.out_proj(out)


class ReLU2MLP(nn.Module):
    def __init__(self, cfg: MiniPLEConfig):
        super().__init__()
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.norm(x))
        h = F.relu(h).square()
        return self.fc2(h)


class PLEInjection(nn.Module):
    """Gemma-4-inspired gated injection of a layer-specific token embedding."""

    def __init__(self, cfg: MiniPLEConfig):
        super().__init__()
        self.ple_norm = nn.RMSNorm(cfg.ple_dim, eps=cfg.rms_eps)
        self.gate = nn.Linear(cfg.d_model, cfg.ple_dim, bias=False)
        self.out_proj = nn.Linear(cfg.ple_dim, cfg.d_model, bias=False)
        self.out_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)

    def forward(self, x: torch.Tensor, ple: torch.Tensor) -> torch.Tensor:
        ple = self.ple_norm(ple)
        gate = F.gelu(self.gate(x), approximate="tanh")
        update = self.out_proj(gate * ple)
        return self.out_norm(update)


class MiniPLEBlock(nn.Module):
    def __init__(self, cfg: MiniPLEConfig, is_local: bool):
        super().__init__()
        self.attn = GQAAttention(cfg, is_local=is_local)
        self.mlp = ReLU2MLP(cfg)
        self.ple = PLEInjection(cfg)

    def forward(
        self,
        x: torch.Tensor,
        ple: torch.Tensor,
        local_mask: Optional[torch.Tensor],
        kv_cache=None,
        cache_lengths=None,
    ) -> torch.Tensor:
        x = x + self.attn(x, local_mask, kv_cache, cache_lengths).to(x.dtype)
        x = x + self.mlp(x).to(x.dtype)
        x = x + self.ple(x, ple).to(x.dtype)
        return x


class MiniPLE(nn.Module):
    """
    146,496,016-parameter causal LM for single-RTX-3090 training.

    Global layers: 1, 2, 6, 10, 14, 18; all other layers are local.
    Local layers use RoPE; global layers default to NoPE.
    Every attention head has a learned zero-value softmax sink.
    The output head is tied to the token embedding.
    """

    def __init__(self, cfg: Optional[MiniPLEConfig] = None):
        super().__init__()
        cfg = MiniPLEConfig() if cfg is None else cfg
        cfg.validate()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # One packed lookup gives every token a distinct PLE vector per layer.
        self.ple_embedding = nn.Embedding(cfg.vocab_size, cfg.n_layers * cfg.ple_dim)
        self.ple_context_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ple_context_proj = nn.Linear(
            cfg.d_model, cfg.n_layers * cfg.ple_dim, bias=False
        )

        self.blocks = nn.ModuleList(
            [
                MiniPLEBlock(cfg, is_local=(kind == "L"))
                for kind in cfg.attention_pattern
            ]
        )
        self.final_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)

        # Shared local-causal mask for the portable SDPA fallback only.
        idx = torch.arange(cfg.max_seq_len)
        i = idx[:, None]
        j = idx[None, :]
        local_mask = (j <= i) & ((i - j) < cfg.local_window)
        self.register_buffer("local_mask", local_mask, persistent=False)

        self._liger_ce = (
            LigerFusedLinearCrossEntropyLoss() if _LIGER_AVAILABLE else None
        )

        self.apply(self._init_weights)
        self._zero_init_residual_projections()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _zero_init_residual_projections(self) -> None:
        # Residual branches start as identity paths, as in the speedrun-inspired recipe.
        for block in self.blocks:
            nn.init.zeros_(block.attn.out_proj.weight)
            nn.init.zeros_(block.mlp.fc2.weight)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _cast_embedding_activation(x: torch.Tensor) -> torch.Tensor:
        # nn.Embedding is not autocast to BF16/FP16. Without this explicit cast,
        # the residual stream stays FP32 after every residual add, wasting VRAM and
        # throughput on consumer GPUs while the linear layers run in low precision.
        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            target_dtype = torch.get_autocast_dtype(device_type)
            if target_dtype in {torch.float16, torch.bfloat16}:
                return x.to(dtype=target_dtype)
        return x

    def _build_ple(self, input_ids: torch.Tensor, token_emb: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        static = self.ple_embedding(input_ids).to(dtype=token_emb.dtype).view(
            b, t, self.cfg.n_layers, self.cfg.ple_dim
        )
        contextual = self.ple_context_proj(self.ple_context_norm(token_emb)).view(
            b, t, self.cfg.n_layers, self.cfg.ple_dim
        )
        return (static + contextual) * (1.0 / math.sqrt(2.0))

    def _decoder(self, input_ids: torch.Tensor, kv_cache=None, cache_lengths=None) -> torch.Tensor:
        _, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {t} exceeds max_seq_len={self.cfg.max_seq_len}"
            )

        token_emb = self._cast_embedding_activation(self.token_embedding(input_ids))
        x = token_emb * math.sqrt(self.cfg.d_model)
        # Split once so backward assembles layer gradients together.
        layer_ples = self._build_ple(input_ids, token_emb).unbind(dim=2)
        mask = self.local_mask[:t, :t]

        for index, (block, layer_ple) in enumerate(zip(self.blocks, layer_ples)):
            if self.cfg.gradient_checkpointing and self.training:
                x = checkpoint(block, x, layer_ple, mask, use_reentrant=False)
            else:
                x = block(x, layer_ple, mask,
                          kv_cache[index] if kv_cache is not None else None, cache_lengths)

        hidden = self.final_norm(x).to(x.dtype)
        if cache_lengths is not None:
            cache_lengths.add_(t)
        return hidden

    def forward_cached(self, input_ids: torch.Tensor, kv_cache, cache_lengths) -> torch.Tensor:
        """Append tokens to the inference cache and project only the final position.

        Use inference.KVGenerator to allocate caches and enforce context bounds.
        """
        hidden = self._decoder(input_ids, kv_cache, cache_lengths)
        return F.linear(hidden[:, -1], self.token_embedding.weight)

    @staticmethod
    def _chunked_linear_cross_entropy(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """Bound logits memory in both forward and backward.

        Ordinary chunking retains every chunk's log-softmax output for backward.
        Checkpoint only this small loss region to release those outputs; fused
        linear CE backends avoid this recomputation and are preferred on CUDA.
        """
        # Cast once outside checkpoint regions. Autocast's weight cache can hold
        # a cast made under no_grad by a fused loss, changing the saved tensors
        # between forward and checkpoint recomputation on older CUDA runtimes.
        if torch.is_autocast_enabled(hidden.device.type):
            dtype = torch.get_autocast_dtype(hidden.device.type)
            hidden = hidden.to(dtype)
            weight = weight.to(dtype)

        def chunk_loss(h, w, y):
            return F.cross_entropy(F.linear(h, w).float(), y, reduction="sum")

        total = hidden.new_zeros((), dtype=torch.float32)
        n = hidden.size(0)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
                part = checkpoint(
                    chunk_loss, hidden[start:end], weight, targets[start:end],
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                part = chunk_loss(hidden[start:end], weight, targets[start:end])
            total = total + part
        return total / n

    @staticmethod
    def _torch_memory_efficient_ce_available() -> bool:
        # Feature-detect linear_cross_entropy. Its default options=None
        # is the reference path and can materialize full logits; passing an explicit
        # LinearCrossEntropyOptions object selects the chunked/memory-efficient path.
        return hasattr(F, "linear_cross_entropy") and hasattr(
            nn, "LinearCrossEntropyOptions"
        )

    def _linear_ce(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        backend = self.cfg.loss_backend
        weight = self.token_embedding.weight if weight is None else weight

        # On CUDA, prefer Liger in auto mode because it is designed as a fused
        # linear+CE kernel and composes cleanly with the rest of this training path.
        if backend in {"auto", "liger"} and self._liger_ce is not None and hidden.is_cuda:
            return self._liger_linear_ce(weight, hidden, targets)
        if backend == "liger":
            raise RuntimeError(
                "loss_backend='liger' requested but Liger Kernel/CUDA is unavailable"
            )

        if backend in {"auto", "torch"} and self._torch_memory_efficient_ce_available():
            options = nn.LinearCrossEntropyOptions()
            return F.linear_cross_entropy(
                hidden,
                weight,
                targets,
                options=options,
            )
        if backend == "torch":
            raise RuntimeError(
                "loss_backend='torch' requires PyTorch's memory-efficient "
                "linear_cross_entropy API (F.linear_cross_entropy + "
                "nn.LinearCrossEntropyOptions). Use --loss-backend chunked or "
                "install a sufficiently recent PyTorch."
            )

        return self._chunked_linear_cross_entropy(
            hidden,
            weight,
            targets,
            self.cfg.ce_chunk_size,
        )

    @torch.compiler.disable
    def _liger_linear_ce(self, weight, hidden, targets):
        # Liger already runs fused CUDA kernels. Keep its Python dispatch outside
        # Dynamo: some supported torch/Liger versions fail tracing that dispatch.
        return self._liger_ce(weight, hidden, targets)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        sampled_candidates: Optional[torch.Tensor] = None,
        sampled_targets: Optional[torch.Tensor] = None,
    ):
        hidden = self._decoder(input_ids)

        loss = None
        if targets is not None:
            flat_h = hidden.reshape(-1, self.cfg.d_model)
            if sampled_candidates is not None:
                if sampled_targets is None:
                    raise ValueError("sampled_targets is required with sampled_candidates")
                if sampled_candidates.ndim != 1:
                    raise ValueError("sampled_candidates must be a 1-D tensor")
                flat_y = sampled_targets.reshape(-1)
                if flat_y.numel() != flat_h.size(0):
                    raise ValueError("sampled_targets must have one label per hidden row")
                # Gather candidate rows from the tied input/output embedding.
                # Keep the temporary small while gradients scatter to FP32 weights.
                sampled_weight = self.token_embedding.weight.index_select(
                    0, sampled_candidates
                )
                if hidden.dtype in {torch.float16, torch.bfloat16}:
                    sampled_weight = sampled_weight.to(dtype=hidden.dtype)
                loss = self._linear_ce(flat_h, flat_y, weight=sampled_weight)
            else:
                flat_y = targets.reshape(-1)
                loss = self._linear_ce(flat_h, flat_y)
            if not return_logits:
                return loss

        logits = F.linear(hidden, self.token_embedding.weight)
        return {"loss": loss, "logits": logits}


if __name__ == "__main__":
    model = MiniPLE()
    print(f"trainable parameters: {model.num_parameters():,}")
    print(f"model size: {model.num_parameters() / 1e6:.3f}M")

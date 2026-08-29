"""CPU smoke tests for the inference KV cache.

Run from the repository root with:

    uv run python -m unittest tests.test_kvcache_consistency
"""

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "nanogpt"))

from config import BATCH_AXIS_NAME, Config, ModelConfig, ShardingRules
from kvcache import KVCache
from model import GPT, forward, forward_infer, precompute_frequencies


def make_tiny_model(*, seqlen=8, window_pattern="L", local_window_size=None):
    """Build a deterministic model whose attention/MLP paths are non-zero."""
    mesh = Mesh(np.array(jax.devices()[:1]), (BATCH_AXIS_NAME,))
    model_config = ModelConfig(
        seqlen=seqlen,
        vocab_size=32,
        d_emb=16,
        num_layers=2,
        q_heads=4,
        kv_heads=2,
        dtype=jnp.float32,
        window_pattern=window_pattern,
        local_window_size=local_window_size,
    )
    cfg = Config(
        mesh=mesh,
        rules=ShardingRules(batch=BATCH_AXIS_NAME),
        model=model_config,
    )
    params = GPT.init(jax.random.key(0), cfg)

    # The default residual output projections are zero-initialized. Make every
    # array non-zero so these tests exercise cache-backed attention rather than
    # merely comparing the embedding-only residual path.
    params = jax.tree.map(
        lambda value: value + jnp.asarray(0.01, dtype=value.dtype)
        if isinstance(value, jax.Array)
        else value,
        params,
    )
    return params, cfg


def full_logits(params, tokens, mesh):
    with jax.set_mesh(mesh):
        positions = jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]
        freqs = precompute_frequencies(
            positions, features=params.blocks[0].attn.head_dim, dtype=jnp.float32
        )
        return forward(params, tokens, segment_ids=None, freqs=freqs)


def cached_logits(params, tokens, segment_ids, cache, cfg):
    with jax.set_mesh(cfg.mesh):
        return forward_infer(
            params, tokens, segment_ids, cache, cfg.model.attn.head_dim
        )


class KVCacheConsistencyTest(unittest.TestCase):
    def assert_logits_close(self, actual, expected):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-5
        )

    def test_prefill_and_single_token_decode_match_full_attention(self):
        params, cfg = make_tiny_model(seqlen=8)
        prompt = jnp.array([[1, 4, 7, 3]], dtype=jnp.int32)
        segment_ids = jnp.ones_like(prompt)
        cache = KVCache.init(jax.random.key(1), cfg.mesh, cfg.rules, 1, cfg)

        cached_prefill_logits, cache = cached_logits(
            params, prompt, segment_ids, cache, cfg
        )
        self.assert_logits_close(
            cached_prefill_logits, full_logits(params, prompt, cfg.mesh)
        )

        next_token = jnp.array([[9]], dtype=jnp.int32)
        cached_next_logits, _ = cached_logits(
            params, next_token, jnp.ones_like(next_token), cache, cfg
        )
        reference = full_logits(
            params, jnp.concatenate([prompt, next_token], axis=1), cfg.mesh
        )
        self.assert_logits_close(cached_next_logits[:, -1], reference[:, -1])

    def test_ring_buffer_decode_matches_full_local_attention_after_wraparound(self):
        # A local window makes the full-sequence reference semantically match a
        # bounded cache. The prefix deliberately exceeds cache capacity.
        params, cfg = make_tiny_model(
            seqlen=4, window_pattern="S", local_window_size=2
        )
        prefix = jnp.array([[2, 5]], dtype=jnp.int32)
        cache = KVCache.init(jax.random.key(2), cfg.mesh, cfg.rules, 1, cfg)
        _, cache = cached_logits(params, prefix, jnp.ones_like(prefix), cache, cfg)

        sequence = prefix
        for token_id in (6, 8, 11, 13, 17):
            token = jnp.array([[token_id]], dtype=jnp.int32)
            cached_step_logits, cache = cached_logits(
                params, token, jnp.ones_like(token), cache, cfg
            )
            sequence = jnp.concatenate([sequence, token], axis=1)
            reference = full_logits(params, sequence, cfg.mesh)
            self.assert_logits_close(cached_step_logits[:, -1], reference[:, -1])

        self.assertGreater(int(cache.end), cfg.model.seqlen)


if __name__ == "__main__":
    unittest.main()

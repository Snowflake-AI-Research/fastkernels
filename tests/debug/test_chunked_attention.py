#!/usr/bin/env python3
"""Tests for chunked local attention metadata remapping.

Verifies that _chunked_prefill_remap and _chunked_decode_remap produce
correct virtual-batch metadata that restricts attention to local windows.
"""

from __future__ import annotations

import sys
import os
import unittest

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
sys.path.insert(0, PROJECT_ROOT)

from kb_nano.tasks.baseline.L2.attention_impl import (
    _chunked_prefill_remap,
    _chunked_decode_remap,
)


class TestChunkedPrefillRemap(unittest.TestCase):
    """Test _chunked_prefill_remap against known examples from vLLM."""

    def test_single_seq_fits_in_one_chunk(self):
        """A sequence shorter than chunk_size should produce one virtual batch."""
        cu_q = torch.tensor([0, 3], dtype=torch.int32)
        cu_k = torch.tensor([0, 3], dtype=torch.int32)

        cu_q_out, cu_k_out, msq, msk, bt_out = _chunked_prefill_remap(
            cu_q, cu_k, None, attention_chunk_size=8, block_size=1,
        )

        self.assertEqual(cu_q_out.tolist(), [0, 3])
        self.assertEqual(cu_k_out.tolist(), [0, 3])
        self.assertEqual(msq, 3)
        self.assertEqual(msk, 3)
        self.assertIsNone(bt_out)

    def test_single_seq_spans_two_chunks(self):
        """A 6-token prefill with chunk_size=4 should split into 2 virtual batches."""
        cu_q = torch.tensor([0, 6], dtype=torch.int32)
        cu_k = torch.tensor([0, 6], dtype=torch.int32)

        cu_q_out, cu_k_out, msq, msk, _ = _chunked_prefill_remap(
            cu_q, cu_k, None, attention_chunk_size=4, block_size=1,
        )

        q_lens = (cu_q_out[1:] - cu_q_out[:-1]).tolist()
        k_lens = (cu_k_out[1:] - cu_k_out[:-1]).tolist()
        self.assertEqual(len(q_lens), 2)
        self.assertEqual(sum(q_lens), 6)
        self.assertEqual(q_lens, [4, 2])
        self.assertEqual(k_lens, [4, 2])

    def test_vllm_example(self):
        """Reproduce the example from vLLM's make_local_attention_virtual_batches doc.

        q_seqlens  = [4, 10, 5]
        kv_seqlens = [6, 17, 9]
        chunk_size = 4

        Expected virtual q_seqlens:  [2, 2, 1, 4, 4, 1, 4, 1]
        Expected virtual kv_seqlens: [4, 2, 4, 4, 4, 1, 4, 1]
        """
        cu_q = torch.tensor([0, 4, 14, 19], dtype=torch.int32)
        cu_k = torch.tensor([0, 6, 23, 32], dtype=torch.int32)

        cu_q_out, cu_k_out, msq, msk, _ = _chunked_prefill_remap(
            cu_q, cu_k, None, attention_chunk_size=4, block_size=1,
        )

        q_lens = (cu_q_out[1:] - cu_q_out[:-1]).tolist()
        k_lens = (cu_k_out[1:] - cu_k_out[:-1]).tolist()

        self.assertEqual(q_lens, [2, 2, 1, 4, 4, 1, 4, 1])
        self.assertEqual(k_lens, [4, 2, 4, 4, 4, 1, 4, 1])
        self.assertEqual(sum(q_lens), 19)
        self.assertEqual(msq, 4)
        self.assertEqual(msk, 4)

    def test_block_table_remapping(self):
        """Block tables should be sliced to only reference each chunk's pages."""
        cu_q = torch.tensor([0, 8], dtype=torch.int32)
        cu_k = torch.tensor([0, 8], dtype=torch.int32)

        block_tables = torch.arange(4).unsqueeze(0)  # [[0, 1, 2, 3]]

        _, _, _, _, bt_out = _chunked_prefill_remap(
            cu_q, cu_k, block_tables,
            attention_chunk_size=4, block_size=2,
        )

        self.assertIsNotNone(bt_out)
        self.assertEqual(bt_out.shape[0], 2)
        self.assertEqual(bt_out.shape[1], 2)
        self.assertEqual(bt_out[0].tolist(), [0, 1])
        self.assertEqual(bt_out[1].tolist(), [2, 3])

    def test_total_query_tokens_preserved(self):
        """The total number of query tokens across virtual batches must equal the original."""
        cu_q = torch.tensor([0, 7, 20], dtype=torch.int32)
        cu_k = torch.tensor([0, 15, 35], dtype=torch.int32)

        cu_q_out, _, _, _, _ = _chunked_prefill_remap(
            cu_q, cu_k, None, attention_chunk_size=8, block_size=1,
        )
        total = cu_q_out[-1].item()
        self.assertEqual(total, 20)


class TestChunkedDecodeRemap(unittest.TestCase):
    """Test _chunked_decode_remap."""

    def test_short_seqlens_unchanged(self):
        """Sequences shorter than chunk_size should be unchanged."""
        cache_seqlens = torch.tensor([3, 5, 2], dtype=torch.int32)
        local, _, max_ctx = _chunked_decode_remap(
            cache_seqlens, None, attention_chunk_size=8, block_size=1,
        )
        self.assertEqual(local.tolist(), [3, 5, 2])
        self.assertEqual(max_ctx, 5)

    def test_long_seqlens_clamped(self):
        """Sequences longer than chunk_size should be clamped."""
        cache_seqlens = torch.tensor([10, 20, 4], dtype=torch.int32)
        local, _, max_ctx = _chunked_decode_remap(
            cache_seqlens, None, attention_chunk_size=8, block_size=1,
        )
        self.assertEqual(local.tolist(), [8, 8, 4])
        self.assertEqual(max_ctx, 8)

    def test_block_table_sliced(self):
        """Block tables should be shifted to reference only the last chunk."""
        cache_seqlens = torch.tensor([16], dtype=torch.int32)
        block_tables = torch.arange(8).unsqueeze(0)  # [[0,1,2,3,4,5,6,7]]

        local, bt_out, max_ctx = _chunked_decode_remap(
            cache_seqlens, block_tables,
            attention_chunk_size=8, block_size=2,
        )
        self.assertEqual(local.tolist(), [8])
        self.assertEqual(max_ctx, 8)
        self.assertIsNotNone(bt_out)
        self.assertEqual(bt_out[0].tolist(), [4, 5, 6, 7])

    def test_block_table_partial_chunk(self):
        """When seqlen doesn't fill a full chunk, only the tail blocks are used."""
        cache_seqlens = torch.tensor([12], dtype=torch.int32)
        block_tables = torch.arange(8).unsqueeze(0)  # [[0,..,7]]

        local, bt_out, _ = _chunked_decode_remap(
            cache_seqlens, block_tables,
            attention_chunk_size=8, block_size=2,
        )
        self.assertEqual(local.tolist(), [8])
        self.assertEqual(bt_out[0].tolist(), [2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from dacp.tools.checkpoint_io import get_size_breakdown


class TestCheckpointSizeBreakdown(unittest.TestCase):
    def test_counts_packbits_padding_per_masked_tensor(self) -> None:
        state_dict = {
            "single": torch.ones(1),
            "seven": torch.ones(7),
        }
        masks = {
            "single": torch.ones(1),
            "seven": torch.ones(7),
        }

        breakdown = get_size_breakdown(state_dict, masks, use_fp16=True)

        self.assertEqual(breakdown["mask_bytes"], 2)
        self.assertEqual(breakdown["values_bytes"], 16)
        self.assertEqual(breakdown["total_compressed_bytes"], 18)
        self.assertEqual(breakdown["original_bytes"], 32)
        self.assertEqual(breakdown["sparsity"], 0)


if __name__ == "__main__":
    unittest.main()

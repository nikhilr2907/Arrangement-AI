"""
PyTorch Dataset classes for music data.

TokenSequenceDataset: Wraps token sequence iterator for DataLoader.
"""

from typing import Iterator, List
import torch
from torch.utils.data import IterableDataset


class TokenSequenceDataset(IterableDataset):
    """
    Wraps a token sequence iterator for use with DataLoader.

    Converts: Iterator[List[int]] → Iterator[torch.Tensor]
      Input: [5, 12, 5, 3, 8, ...]  (token sequence from one song)
      Output: tensor([1, 5, 12, 5, 3, 8, ..., 2])  (with BOS/EOS)

    Usage:
        dataset = TokenSequenceDataset(token_iterator, bos_idx=1, eos_idx=2)
        loader = DataLoader(dataset, batch_size=32, collate_fn=pad_collate)
        for batch in loader:
            # batch = padded token sequences
    """

    def __init__(
        self,
        seq_iterator: Iterator[List[int]],
        bos_idx: int = 1,
        eos_idx: int = 2
    ):
        """
        Args:
            seq_iterator: Iterator yielding token lists (one per song)
            bos_idx: Beginning-of-sequence token (default: 1)
            eos_idx: End-of-sequence token (default: 2)
        """
        self.seq_iterator = seq_iterator
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __iter__(self):
        """
        Iterate through token sequences, adding BOS/EOS and converting to tensor.
        """
        for tokens in self.seq_iterator:
            # Add BOS/EOS and convert to tensor
            seq = torch.tensor(
                [self.bos_idx] + tokens + [self.eos_idx],
                dtype=torch.long
            )
            yield seq

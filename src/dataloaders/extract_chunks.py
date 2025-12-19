"""Dataloader for extracting and loading chunked training data."""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ChunkDataset(Dataset):
    """PyTorch Dataset for loading chunked training data from .npz files."""

    def __init__(self, chunks_dir: str = "save/chunks", file_pattern: str = "*.npz"):
        """
        Initialize the chunk dataset.

        Args:
            chunks_dir: Directory containing the chunk .npz files
            file_pattern: Glob pattern for matching chunk files
        """
        self.chunks_dir = Path(chunks_dir)
        self.file_pattern = file_pattern

        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {self.chunks_dir}")

        # Load all chunk files
        self.chunk_files = sorted(list(self.chunks_dir.glob(file_pattern)))

        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {self.chunks_dir} matching {file_pattern}")

        # Load and concatenate all chunks
        self.data = self._load_all_chunks()

        print(f"Loaded {len(self.chunk_files)} chunk files with {len(self.data)} total samples")

    def _load_all_chunks(self) -> np.ndarray:
        """Load all chunk files and concatenate them."""
        all_chunks = []

        for chunk_file in self.chunk_files:
            with np.load(chunk_file) as data:
                # The key used in process_chunks.py is 'chunks'
                if 'chunks' in data:
                    chunks = data['chunks']
                # Fallback to other possible keys
                elif 'my_array' in data:
                    chunks = data['my_array']
                else:
                    # Take the first array found
                    chunks = data[list(data.keys())[0]]

                all_chunks.append(chunks)
                print(f"  Loaded {chunk_file.name}: {chunks.shape}")

        # Concatenate all chunks along the first dimension
        return np.concatenate(all_chunks, axis=0) if all_chunks else np.array([])

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tensor containing the chunk data
        """
        sample = self.data[idx]
        return torch.from_numpy(sample).float()


class ChunkDataLoader:
    """Helper class for creating DataLoaders from chunk files."""

    @staticmethod
    def create_dataloader(
        chunks_dir: str = "save/chunks",
        file_pattern: str = "*.npz",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader for chunk data.

        Args:
            chunks_dir: Directory containing the chunk .npz files
            file_pattern: Glob pattern for matching chunk files
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            **kwargs: Additional arguments to pass to DataLoader

        Returns:
            PyTorch DataLoader instance
        """
        dataset = ChunkDataset(chunks_dir=chunks_dir, file_pattern=file_pattern)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )

    @staticmethod
    def load_as_numpy(
        chunks_dir: str = "save/chunks",
        file_pattern: str = "*.npz"
    ) -> np.ndarray:
        """
        Load all chunks directly as a numpy array (without PyTorch).

        Args:
            chunks_dir: Directory containing the chunk .npz files
            file_pattern: Glob pattern for matching chunk files

        Returns:
            Concatenated numpy array of all chunks
        """
        chunks_path = Path(chunks_dir)

        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_path}")

        chunk_files = sorted(list(chunks_path.glob(file_pattern)))

        if not chunk_files:
            raise ValueError(f"No chunk files found in {chunks_path} matching {file_pattern}")

        all_chunks = []

        for chunk_file in chunk_files:
            with np.load(chunk_file) as data:
                if 'chunks' in data:
                    chunks = data['chunks']
                elif 'my_array' in data:
                    chunks = data['my_array']
                else:
                    chunks = data[list(data.keys())[0]]

                all_chunks.append(chunks)
                print(f"Loaded {chunk_file.name}: {chunks.shape}")

        concatenated = np.concatenate(all_chunks, axis=0) if all_chunks else np.array([])
        print(f"Total shape: {concatenated.shape}")

        return concatenated

    @staticmethod
    def load_specific_files(
        file_paths: List[Union[str, Path]],
        concatenate: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Load specific chunk files by their paths.

        Args:
            file_paths: List of file paths to load
            concatenate: Whether to concatenate all chunks into a single array

        Returns:
            Either a concatenated numpy array or list of arrays
        """
        chunks_list = []

        for file_path in file_paths:
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"Chunk file not found: {path}")

            with np.load(path) as data:
                if 'chunks' in data:
                    chunks = data['chunks']
                elif 'my_array' in data:
                    chunks = data['my_array']
                else:
                    chunks = data[list(data.keys())[0]]

                chunks_list.append(chunks)
                print(f"Loaded {path.name}: {chunks.shape}")

        if concatenate:
            result = np.concatenate(chunks_list, axis=0)
            print(f"Total concatenated shape: {result.shape}")
            return result
        else:
            return chunks_list


# Convenience function for quick loading
def load_chunks(
    chunks_dir: str = "save/chunks",
    file_pattern: str = "*.npz",
    as_tensors: bool = False,
    batch_size: Optional[int] = None
) -> Union[np.ndarray, DataLoader]:
    """
    Convenience function to load chunks.

    Args:
        chunks_dir: Directory containing the chunk .npz files
        file_pattern: Glob pattern for matching chunk files
        as_tensors: If True, return a DataLoader; if False, return numpy array
        batch_size: If provided with as_tensors=True, create a DataLoader

    Returns:
        Either a numpy array or a DataLoader depending on parameters
    """
    if as_tensors and batch_size is not None:
        return ChunkDataLoader.create_dataloader(
            chunks_dir=chunks_dir,
            file_pattern=file_pattern,
            batch_size=batch_size
        )
    elif as_tensors:
        dataset = ChunkDataset(chunks_dir=chunks_dir, file_pattern=file_pattern)
        # Return all data as a single tensor
        return torch.stack([dataset[i] for i in range(len(dataset))])
    else:
        return ChunkDataLoader.load_as_numpy(
            chunks_dir=chunks_dir,
            file_pattern=file_pattern
        )

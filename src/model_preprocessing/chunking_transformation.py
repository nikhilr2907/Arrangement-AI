from typing import List, Optional

import numpy as np


class QuantizedMatrix:
    """Dummy datatype placeholder for quantized feature matrices."""
    pass


def chunking_transformation(
    quantized_matrix: np.ndarray,
    chunk_size: int,
    overlap: int,
    random_chunks: bool = False,
    num_chunks: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Slices the full time-series of quantized feature vectors into chunks.

    Args:
        quantized_matrix: 2D array of shape (num_frames, N_features)
        chunk_size: Number of frames per chunk (used if random_chunks=False)
        overlap: Number of overlapping frames between consecutive chunks (used if random_chunks=False)
        random_chunks: If True, generate random chunks with varying positions and sizes
        num_chunks: Number of random chunks to generate (only used if random_chunks=True)
        min_chunk_size: Minimum chunk size for random chunks (default: chunk_size // 2)
        max_chunk_size: Maximum chunk size for random chunks (default: chunk_size * 2)
        seed: Random seed for reproducibility

    Returns:
        List of chunks, each of shape (chunk_length, N_features).
        Note: When random_chunks=True, chunks may have different lengths.
    """
    if seed is not None:
        np.random.seed(seed)

    num_frames = quantized_matrix.shape[0]
    N_features = quantized_matrix.shape[1]

    if random_chunks:
        # Generate random chunks with varying positions and sizes
        if num_chunks is None:
            raise ValueError("num_chunks must be specified when random_chunks=True")

        if min_chunk_size is None:
            min_chunk_size = max(4, chunk_size // 2)
        if max_chunk_size is None:
            max_chunk_size = min(num_frames, chunk_size * 2)

        if min_chunk_size > num_frames:
            raise ValueError(
                f"min_chunk_size ({min_chunk_size}) is larger than num_frames ({num_frames})"
            )

        chunks = []
        for _ in range(num_chunks):
            # Random chunk size
            current_chunk_size = np.random.randint(min_chunk_size, max_chunk_size + 1)
            current_chunk_size = min(current_chunk_size, num_frames)

            # Random start position
            max_start = num_frames - current_chunk_size
            start_idx = np.random.randint(0, max_start + 1)
            end_idx = start_idx + current_chunk_size

            chunk = quantized_matrix[start_idx:end_idx, :]
            chunks.append(chunk)

        return chunks  # Return list of variable-length chunks

    else:
        # Regular sliding window chunking
        if overlap >= chunk_size:
            raise ValueError("Overlap must be strictly less than chunk_size.")

        step = chunk_size - overlap
        num_complete_chunks = (num_frames - chunk_size) // step + 1

        if num_complete_chunks <= 0:
            raise ValueError(
                f"Input too short ({num_frames} frames) for chunk_size={chunk_size}"
            )

        chunked_data = np.empty(
            (num_complete_chunks, chunk_size, N_features), dtype=quantized_matrix.dtype
        )

        for i in range(num_complete_chunks):
            start_index = i * step
            end_index = start_index + chunk_size
            chunked_data[i] = quantized_matrix[start_index:end_index, :]

        return chunked_data


def batch_chunking_transformation(
    quantized_matrices: List[np.ndarray],
    chunk_size: int,
    overlap: int,
    random_chunks: bool = False,
    target_total_chunks: int = 1000,
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Apply chunking transformation to a batch of quantized matrices.

    Args:
        quantized_matrices: List/array of 2D matrices, each of shape (num_frames, N_features)
        chunk_size: Number of frames per chunk (base size)
        overlap: Number of overlapping frames (for regular chunking)
        random_chunks: If True, generate random chunks with varying sizes and positions
        target_total_chunks: Target number of total chunks across all matrices (default: 1000)
        min_chunk_size: Minimum chunk size for random chunks
        max_chunk_size: Maximum chunk size for random chunks
        seed: Random seed for reproducibility

    Returns:
        List of chunks (variable-length if random_chunks=True)
    """
    if not quantized_matrices:
        return []

    if random_chunks:
        # Distribute chunks across matrices proportionally to their length
        total_frames = sum(m.shape[0] for m in quantized_matrices)
        all_chunks = []

        for matrix in quantized_matrices:
            # Calculate proportional number of chunks for this matrix
            matrix_proportion = matrix.shape[0] / total_frames
            num_chunks_for_matrix = max(1, int(target_total_chunks * matrix_proportion))

            chunks = chunking_transformation(
                matrix,
                chunk_size=chunk_size,
                overlap=overlap,
                random_chunks=True,
                num_chunks=num_chunks_for_matrix,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                seed=seed,
            )
            all_chunks.extend(chunks)

        return all_chunks  # Return list of variable-length chunks

    else:
        # Regular sliding window chunking - stack into 3D tensor
        all_chunks = []

        for matrix in quantized_matrices:
            chunks = chunking_transformation(matrix, chunk_size, overlap)
            all_chunks.append(chunks)

        # Stack all chunks into a single 3D tensor
        return np.vstack(all_chunks)

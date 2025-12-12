import numpy as np

class QuantizedMatrix:
    def __init__(self, data: np.ndarray):
        self.data = data


def chunking_transformation(quantized_matrix: QuantizedMatrix, chunk_size: int, overlap: int) -> np.ndarray:
    """
    Slices the full time-series of quantized feature vectors into a set of 
    overlapping chunks for sequence model training using a sliding window.
    """
    
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size.")
    
    step = chunk_size - overlap
    num_frames = quantized_matrix.shape[0]
    
    # Calculate the number of complete, valid chunks
    num_chunks = (num_frames - chunk_size) // step + 1

    N_features = quantized_matrix.shape[2]
    timesteps = quantized_matrix.shape[1]
    chunked_data = np.empty((num_chunks, chunk_size,timesteps,N_features), dtype=quantized_matrix.dtype)
    
    for i in range(num_chunks):
        start_index = i * step
        end_index = start_index + chunk_size
        
        chunked_data[i] = quantized_matrix[start_index:end_index, :, :]
        
    return chunked_data


def batch_chunking_transformation(quantized_matrices: np.ndarray[QuantizedMatrix], chunk_size: int, overlap: int) -> list[np.ndarray]:
    """
    Apply chunking transformation to a batch of quantized matrices.
    """
    return [chunking_transformation(qm.data, chunk_size, overlap) for qm in quantized_matrices]

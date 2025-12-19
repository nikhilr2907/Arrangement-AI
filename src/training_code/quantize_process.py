
from pathlib import Path
import numpy as np
import torch

from src.model_preprocessing.quantization.vq_vae import VQ_VAE, VQ_VAE_TRAINER
from src.dataloaders.extract_chunks import ChunkDataLoader


class QuantizationProcessor:
    """Process chunks through VQ-VAE and save quantized versions with matching filenames."""

    def __init__(self, vq_vae_trainer: VQ_VAE_TRAINER):
        """
        Initialize the quantization processor.

        Args:
            vq_vae_trainer: Trained VQ-VAE trainer instance
        """
        self.trainer = vq_vae_trainer

    def quantize_and_save(
        self,
        chunks_dir: str = "save/chunks",
        output_dir: str = "save/quantized_chunks",
        batch_size: int = 32
    ):
        """
        Quantize all chunks and save with matching filenames.

        Args:
            chunks_dir: Directory containing original chunk files
            output_dir: Directory to save quantized chunks
            batch_size: Batch size for processing
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load chunks by name to preserve filenames
        chunks_by_name = ChunkDataLoader.load_by_name(
            chunks_dir=chunks_dir,
            return_dict=True
        )

        print(f"\nQuantizing {len(chunks_by_name)} files...")

        for filename, chunks in chunks_by_name.items():
            print(f"\nProcessing {filename}...")

            # Reshape chunks for VQ-VAE if needed
            original_shape = chunks.shape
            chunks_reshaped = chunks.reshape(chunks.shape[0], -1)

            # Convert to tensor
            chunks_tensor = torch.from_numpy(chunks_reshaped).float()

            # Get quantized indices (single index per time step)
            all_indices = []
            for i in range(0, len(chunks_tensor), batch_size):
                batch = chunks_tensor[i:i + batch_size]
                indices = self.trainer.get_encoded_indices(batch, flatten=True)
                all_indices.append(indices.cpu().numpy())

            # Concatenate all indices
            quantized_indices = np.concatenate(all_indices, axis=0)

            # Save with matching filename
            output_file = output_path / f"{filename}.npz"
            np.savez_compressed(
                output_file,
                quantized_indices=quantized_indices,
                original_shape=original_shape
            )

            print(f"  Saved {filename}.npz: {quantized_indices.shape}")

        print(f"\nQuantization complete! Saved {len(chunks_by_name)} files to {output_dir}")

    def quantize_single_file(
        self,
        input_file: str,
        output_file: str,
        batch_size: int = 32
    ):
        """
        Quantize a single chunk file.

        Args:
            input_file: Path to input chunk file
            output_file: Path to save quantized output
            batch_size: Batch size for processing
        """
        # Load chunks
        with np.load(input_file) as data:
            if 'chunks' in data:
                chunks = data['chunks']
            else:
                chunks = data[list(data.keys())[0]]

        print(f"Loaded {Path(input_file).name}: {chunks.shape}")

        # Reshape for VQ-VAE
        original_shape = chunks.shape
        chunks_reshaped = chunks.reshape(chunks.shape[0], -1)
        chunks_tensor = torch.from_numpy(chunks_reshaped).float()

        # Quantize in batches (single index per time step)
        all_indices = []
        for i in range(0, len(chunks_tensor), batch_size):
            batch = chunks_tensor[i:i + batch_size]
            indices = self.trainer.get_encoded_indices(batch, flatten=True)
            all_indices.append(indices.cpu().numpy())

        quantized_indices = np.concatenate(all_indices, axis=0)

        # Save
        np.savez_compressed(
            output_file,
            quantized_indices=quantized_indices,
            original_shape=original_shape
        )

        print(f"Saved quantized indices to {output_file}: {quantized_indices.shape}")


class QuantizationProcess:
    """Complete pipeline for VQ-VAE training and quantization."""

    def __init__(self, input_size: int, batch_size: int = 32, k: int = 10, hidden: int = 200):
        """
        Initialize the quantization process.

        Args:
            input_size: Size of input features
            batch_size: Batch size for training
            k: Number of embeddings in codebook
            hidden: Hidden dimension size
        """
        self.vq_vae = VQ_VAE(hidden=hidden, k=k, input_size=input_size, batch_size=batch_size)
        self.trainer = VQ_VAE_TRAINER(self.vq_vae)
        self.batch_size = batch_size
        self.dataloader = None

    def initialize_dataloader(self, chunks_dir: str = "save/chunks", shuffle: bool = True):
        """
        Initialize the dataloader for training.

        Args:
            chunks_dir: Directory containing chunk files
            shuffle: Whether to shuffle the data
        """
        self.dataloader = ChunkDataLoader.create_dataloader(
            chunks_dir=chunks_dir,
            batch_size=self.batch_size,
            shuffle=shuffle
        )
        print(f"Initialized dataloader with {len(self.dataloader)} batches")

    def train_vq_vae(self, num_epochs: int, learning_rate: float = 1e-3):
        """
        Train the VQ-VAE model.

        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        if self.dataloader is None:
            raise ValueError("Dataloader not initialized. Call initialize_dataloader() first.")

        optimizer = torch.optim.Adam(self.trainer.parameters(), lr=learning_rate)

        print(f"\nTraining VQ-VAE for {num_epochs} epochs...")
        self.trainer.full_training_sequence(self.dataloader, optimizer, num_epochs)

        print("\nTraining complete!")

    def save_model_and_embeddings(
        self,
        model_path: str = "save/main_model_checkpoints/vq_vae_model.pt",
        embeddings_path: str = "save/main_model_checkpoints/vq_vae_embeddings.pt"
    ):
        """
        Save model weights and embeddings.

        Args:
            model_path: Path to save model weights
            embeddings_path: Path to save embeddings
        """
        # Create directory if needed
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        self.trainer.save_model_weights(model_path)
        self.trainer.save_embeddings(embeddings_path)

    def quantize_all_chunks(
        self,
        chunks_dir: str = "save/chunks",
        output_dir: str = "save/quantized_chunks"
    ):
        """
        Quantize all chunks and save with matching filenames.

        Args:
            chunks_dir: Directory containing original chunks
            output_dir: Directory to save quantized chunks
        """
        processor = QuantizationProcessor(self.trainer)
        processor.quantize_and_save(
            chunks_dir=chunks_dir,
            output_dir=output_dir,
            batch_size=self.batch_size
        )

    def run_full_pipeline(
        self,
        chunks_dir: str = "save/chunks",
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        model_path: str = "save/main_model_checkpoints/vq_vae_model.pt",
        embeddings_path: str = "save/main_model_checkpoints/vq_vae_embeddings.pt",
        quantized_output_dir: str = "save/quantized_chunks"
    ):
        """
        Run the complete training and quantization pipeline.

        Args:
            chunks_dir: Directory containing chunk files
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            model_path: Path to save model weights
            embeddings_path: Path to save embeddings
            quantized_output_dir: Directory to save quantized chunks
        """
        # Step 1: Initialize dataloader
        self.initialize_dataloader(chunks_dir)

        # Step 2: Train VQ-VAE
        self.train_vq_vae(num_epochs, learning_rate)

        # Step 3: Save model and embeddings
        self.save_model_and_embeddings(model_path, embeddings_path)

        # Step 4: Quantize all chunks
        self.quantize_all_chunks(chunks_dir, quantized_output_dir)

        print(f"\nFull pipeline complete!")
        print(f"  Model saved to: {model_path}")
        print(f"  Embeddings saved to: {embeddings_path}")
        print(f"  Quantized chunks saved to: {quantized_output_dir}")


# Convenience function
def quantize_chunks_from_model(
    model_path: str,
    chunks_dir: str = "save/chunks",
    output_dir: str = "save/quantized_chunks",
    input_size: int = None,
    **vq_vae_kwargs
):
    """
    Load a trained VQ-VAE model and quantize chunks.

    Args:
        model_path: Path to trained VQ-VAE model weights
        chunks_dir: Directory containing chunks
        output_dir: Directory to save quantized chunks
        input_size: Input size for VQ-VAE (auto-detected if None)
        **vq_vae_kwargs: Additional kwargs for VQ-VAE (k, hidden, etc.)
    """
    # Auto-detect input size if not provided
    if input_size is None:
        sample = ChunkDataLoader.load_as_numpy(chunks_dir)
        input_size = sample[0].reshape(-1).shape[0]
        print(f"Auto-detected input_size: {input_size}")

    # Create model
    vq_vae = VQ_VAE(input_size=input_size, **vq_vae_kwargs)
    trainer = VQ_VAE_TRAINER(vq_vae)

    # Load weights
    vq_vae.load_state_dict(torch.load(model_path))
    print(f"Loaded VQ-VAE model from {model_path}")

    # Quantize
    processor = QuantizationProcessor(trainer)
    processor.quantize_and_save(chunks_dir=chunks_dir, output_dir=output_dir)

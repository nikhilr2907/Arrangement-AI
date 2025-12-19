"""Full transformer training pipeline with checkpoint management."""

from pathlib import Path
import torch

from src.main_model.transformers_improved import VariableLengthTrainer


class TransformerPipeline:
    """Pipeline for training transformer models with checkpoint support."""

    def __init__(self, trainer, dataloader):
        """
        Initialize the transformer pipeline.

        Args:
            trainer: VariableLengthTrainer instance
            dataloader: DataLoader for training data
        """
        self.trainer = trainer
        self.dataloader = dataloader

    def call_checkpoints(
        self,
        checkpoint_path: str = "save/main_model_checkpoints/transformer_checkpoint.pt",
        load_if_exists: bool = True
    ) -> bool:
        """
        Load checkpoint if it exists in main_model_checkpoints, otherwise save current model.

        Args:
            checkpoint_path: Path to the checkpoint file
            load_if_exists: If True, load checkpoint if it exists; otherwise always save

        Returns:
            bool: True if checkpoint was loaded, False if it was saved
        """
        checkpoint_file = Path(checkpoint_path)

        # Try to load checkpoint if it exists and loading is enabled
        if load_if_exists and checkpoint_file.exists():
            try:
                checkpoint = torch.load(checkpoint_path)

                # Load model state
                self.trainer.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model state from {checkpoint_path}")

                # Load optimizer state if present
                if 'optimizer_state_dict' in checkpoint:
                    self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("  Loaded optimizer state")

                # Display checkpoint info
                epoch = checkpoint.get('epoch', 'unknown')
                loss = checkpoint.get('loss', None)

                print(f"  Checkpoint epoch: {epoch}")
                if loss is not None:
                    print(f"  Checkpoint loss: {loss:.4f}")

                return True

            except Exception as e:
                print(f"Failed to load checkpoint from {checkpoint_path}: {e}")
                print("Starting with fresh model...")
                return False

        # Save checkpoint if it doesn't exist or loading is disabled
        else:
            if not checkpoint_file.exists():
                print(f"No checkpoint found at {checkpoint_path}")
            self.save_checkpoint(checkpoint_path)
            return False

    def save_checkpoint(
        self,
        checkpoint_path: str = "save/main_model_checkpoints/transformer_checkpoint.pt",
        epoch: int = None,
        loss: float = None
    ):
        """
        Save model checkpoint including model and optimizer state.

        Args:
            checkpoint_path: Path to save the checkpoint
            epoch: Current epoch number (optional)
            loss: Current loss value (optional)
        """
        # Create directory if it doesn't exist
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint dictionary
        checkpoint = {
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'model_config': {
                'vocab_size': self.trainer.model.vocab_size,
                'model_dim': self.trainer.model.model_dim,
                'max_seq_len': self.trainer.model.max_seq_len,
            }
        }

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if loss is not None:
            checkpoint['loss'] = loss

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        if epoch is not None:
            print(f"  Epoch: {epoch}")
        if loss is not None:
            print(f"  Loss: {loss:.4f}")
    

    def train_model(
        self,
        num_epochs: int = 100,
        checkpoint_path: str = "save/main_model_checkpoints/transformer_checkpoint.pt",
        save_every: int = 10
    ):
        """
        Train the model using the provided dataloader with checkpoint saving.

        Args:
            num_epochs: Number of training epochs
            checkpoint_path: Path to save checkpoints
            save_every: Save checkpoint every N epochs
        """
        print(f"\nTraining transformer for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in self.dataloader:
                # Assuming batch is a list of sequences
                loss = self.trainer.train_step_next_token_prediction(batch)
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    epoch=epoch + 1,
                    loss=avg_loss
                )

        # Save final checkpoint
        self.save_checkpoint(
            checkpoint_path=checkpoint_path,
            epoch=num_epochs,
            loss=avg_loss
        )

        print("\nTraining complete!")

    def run_full_pipeline(
        self,
        num_epochs: int = 100,
        checkpoint_path: str = "save/main_model_checkpoints/transformer_checkpoint.pt",
        resume_training: bool = True,
        save_every: int = 10
    ):
        """
        Run the complete training pipeline with checkpoint management.

        Args:
            num_epochs: Number of training epochs
            checkpoint_path: Path to checkpoint file
            resume_training: If True, resume from checkpoint if it exists
            save_every: Save checkpoint every N epochs
        """
        # Load checkpoint if resuming
        if resume_training:
            loaded = self.call_checkpoints(
                checkpoint_path=checkpoint_path,
                load_if_exists=True
            )
            if loaded:
                print("Resuming training from checkpoint...")
        else:
            print("Starting fresh training (ignoring any existing checkpoints)...")

        # Train the model
        self.train_model(
            num_epochs=num_epochs,
            checkpoint_path=checkpoint_path,
            save_every=save_every
        )


    

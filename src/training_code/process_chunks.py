"""Data transformation pipeline for processing audio stems into training chunks."""

import numpy as np
from pathlib import Path

from audio_training_breakdown_automation.audio_breakdown import run as audio_breakdown_run


class DataTransformation:
    """Conduct the full data pipeline operation to create training data sequences."""

    @staticmethod
    def forward(
        stem_paths: list[str],
        tempo: float = None,
        use_manual_tempo: bool = True,
        output_dir: str = "save/chunks",
        chunk_name: str = None
    ) -> None:
        """
        Run the complete audio breakdown and processing pipeline.

        Args:
            stem_paths: List of file paths to audio stems
            tempo: Optional tempo value to use for beat tracking
            use_manual_tempo: Whether to use the provided tempo or estimate it
            output_dir: Directory to save the processed chunks
            chunk_name: Name for the output file (without extension)

        Returns:
            None
        """
        _, chunked_training_segments = audio_breakdown_run(
            stem_paths, tempo, use_manual_tempo
        )

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if chunk_name is None:
            chunk_name = "processed_training_data"

        output_file = output_path / f"{chunk_name}.npz"

        # Save compressed numpy array
        np.savez_compressed(
            output_file,
            chunks=chunked_training_segments,
        )

        print(f"Saved {len(chunked_training_segments)} chunks to {output_file}")

    @staticmethod
    def stem_path_generator(folder_path: str) -> list[str]:
        """
        Generate list of stem file paths from a given folder.

        Args:
            folder_path: Path to the folder containing stem audio files

        Returns:
            List of file paths to stem audio files
        """
        folder = Path(folder_path)
        return [str(file) for file in folder.glob("*.wav")]

    @staticmethod
    def save_tensors(folder_paths: list[str], save_dir: str = "save/chunks") -> None:
        """
        Save processed training data tensors from multiple folders.

        Args:
            folder_paths: List of folder paths containing stem audio files
            save_dir: Directory to save the processed chunks

        Returns:
            None
        """
        for folder_path in folder_paths:
            # Extract folder name from path for naming the output file
            folder_name = Path(folder_path).name

            # Generate stem paths from folder
            stem_paths = DataTransformation.stem_path_generator(folder_path)

            if not stem_paths:
                print(f"Warning: No .wav files found in {folder_path}")
                continue

            print(f"Processing folder: {folder_path} ({len(stem_paths)} stems)")

            # Process and save with folder name
            DataTransformation.forward(
                stem_paths=stem_paths,
                output_dir=save_dir,
                chunk_name=folder_name
            )

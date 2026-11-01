import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official MusicBERT weights to HF format")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    args = parser.parse_args()
    if not args.trusted_checkpoint:
        parser.error("Legacy fairseq pickle loading requires --trusted-checkpoint")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Conversion output directory must be empty")
    from musicbert_hf.checkpoints import load_musicbert_from_fairseq_checkpoint

    model = load_musicbert_from_fairseq_checkpoint(str(args.checkpoint), weights_only=False)
    # The port does not declare its shared MLM bias to modern Safetensors exporters.
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    model.save_pretrained(args.output, state_dict=state)


if __name__ == "__main__":
    main()

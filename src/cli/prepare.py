import argparse
import logging
from pathlib import Path

from src.data.dataset import replay
from src.data.preparation import EpisodeBuilder, LakhSource, SlakhSource
from src.data.serialization import read_episodes, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare bar-aligned clip-reuse episodes")
    parser.add_argument("--source", choices=("lakh", "slakh", "observed"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    failures: list[tuple[str, str]] = []
    if args.source == "observed":
        episodes = read_episodes(args.root)
        for episode in episodes:
            replay(episode)
    else:
        source = LakhSource(args.root) if args.source == "lakh" else SlakhSource(args.root)
        builder = EpisodeBuilder()
        episodes = []
        for composition_id, clips in source.iter_compositions():
            try:
                episode = builder.build(composition_id, clips)
                replay(episode)
                episodes.append(episode)
            except ValueError as error:
                failures.append((composition_id, str(error)))
            if args.limit and len(episodes) >= args.limit:
                break
        failures.extend(source.failures)
    if not episodes:
        write_json(args.output.with_suffix(".report.json"), {"failures": failures})
        raise ValueError("No usable episodes; inspect preparation report")
    write_json(args.output, episodes)
    write_json(
        args.output.with_suffix(".report.json"), {"episodes": len(episodes), "failures": failures}
    )
    logging.info("Prepared %d episodes; excluded %d", len(episodes), len(failures))


if __name__ == "__main__":
    main()

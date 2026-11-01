# Stem arranger

An arrangement policy over supplied MIDI and audio stems. Each instrument track owns variable-length source material. At each fractional position within a destination bar, the model decides whether to continue, stop, or restart from a scored source position.

This generates placements, crops and repetitions of existing material. It does not generate new notes or audio. No full songs are required at inference.

## Model and agreed context plan

A **track** is an instrument lane; the **overall clip/song** is the arrangement being built. A track owns variable-length **source material**. Existing Python/JSON names remain `Channel` for a track and `Clip` for its source material.

At each destination step, the model builds four context tokens, each 128-dimensional with the default configuration:

| Context | Input sequence | How it becomes one context token |
| --- | --- | --- |
| Musical history | Up to 16 completed bars across all tracks, plus the completed prefix of the current bar | Temporal attention within each track, attention across tracks at matching steps, learned bar pooling, bar-history attention, then a query-conditioned readout |
| Target track's progression | OFF / START / CONTINUE tokens at step resolution over the same history | Destination positions and source pointers are added; a causal temporal encoder preserves the sequence; the decision query reads it |
| Target track's source material | All available source-feature positions, including while the track is off | Separate MIDI/audio projections, source-position encoding, source temporal attention, then a query-conditioned readout |
| Other tracks now | Other tracks' source content at this destination step | Cross-track attention followed by learned pooling into one joint token |

```text
Pooled musical-history sequence -> query readout --+
Target playback-token sequence -> query readout --+-> four context tokens
Target source-feature sequence -> query readout --+         |
Other tracks at this step -> interaction + pooling +         v
                                                   fusion attention
                                                          |
                                                   one decision vector
                                                          |
                                       KEEP/change gate + source-position scorer
```

The query describes the current destination time, normalized progress through the requested arrangement, target role, playing/off state, continuation validity and source cursor. Its 15 numeric metadata values are six destination coordinates, five source-pointer coordinates, playing/continuation flags, role and overall progress. It also receives the target's available source-content summary and the content KEEP would play. Each readout is conditional on this query; it is not a permanent replacement for its input sequence. The shared scorer still evaluates every legal source position using that candidate's own musical/position features.

### Implementation sequence

1. Represent source and destination positions as a bar index plus a discrete within-bar step; use the same transitions during training, generation and rendering.
2. Preserve frozen MusicBERT/MERT feature caches and separate trainable modality projections. Encode source steps before any pooling; keep the selectable source-step axis.
3. Reconstruct musical history from the source intervals that actually played. Apply step-level RoPE attention within tracks, cross-track attention, then learned pooling within each bar.
4. Retain the target's playback-token sequence separately. Include START boundaries and source pointers so identical on/off patterns can describe different crops and repetitions.
5. Encode committed and explicitly provisional other-track content at the current step; form the joint current-step token without exposing undecided targets.
6. Read the three sequences, fuse their three readouts with the joint current-step token, and pass the resulting decision vector to the existing gate/scorer heads.
7. Verify within-bar repeats, MIDI/audio cuts, masks, causality, permutation behavior, mixed-modality gradients and checkpoint/training/generation round trips.

### Timing and positions

`steps_per_bar` is the number of equal **fractions of a bar**. New request creation and automatic preparation default to 16. In 4/4 these are sixteenth notes; 3/4 would use 12 steps for a sixteenth-note grid. A request uses one fixed fraction count across its bars, even if meter changes. This is not a fixed note-value grid across changing meters. The grid must divide the source and destination tick lengths exactly. Meters and audio-step durations must be compatible; there is no implicit time stretching.

The destination clock describes where material is placed in the overall arrangement. The source cursor describes where the target track would read next if KEEP is chosen. They are independent.

For a history step:

```text
destination position = [
    (history_bar - current_bar) / history_bars,
    phase, sin(2*pi*phase), cos(2*pi*phase),
    meter_numerator / 16, meter_denominator / 16
]
phase = step_within_bar / steps_per_bar

source pointer = [
    source_bar / source_bar_count,
    source_phase, sin(2*pi*source_phase), cos(2*pi*source_phase),
    source_bar_count / 32
]
```

The six destination numbers and five source-pointer numbers pass through learned linear projections into the common model width. Coordinates are computed from the grids, not inferred from arbitrary IDs. OFF has no active source pointer. Length scaling by 32 is numerical scaling, not a required loop length.

Every stream has a learned type vector: musical history, playback history, source material or other tracks now. Type and coordinate vectors are added to content embeddings. The playback vocabulary is OFF=0, START=1, CONTINUE=2; OFF means no placement, not necessarily a detected acoustic rest inside placed source material.

RoPE rotates attention queries/keys within each stream's own clock. It runs at feature/step resolution before musical-history pooling, on the bar-history sequence, on the playback sequence and on the target source sequence. Cross-track and four-token fusion attention use equal positions so instrument ordering is not mistaken for musical time. Cross-stream readouts do not apply one arbitrary concatenation clock.

With batch size B, history H=16, fraction count S=16, track count C and model width D=128:

| Representation | Shape |
| --- | --- |
| Musical history before pooling | B x (H+1) x S x C x D |
| Musical history memory | B x (H+2) x D, including BOS and a masked current-bar prefix |
| Playback memory | B x ((H+1)*S) x D, with unused/current-future positions masked |
| Target source memory | B x source_steps x D, independently padded |
| Current other-track members | B x other_tracks x D |
| Four context readouts | B x 4 x D |
| Final decision vector | B x D |

All supplied source material is available at inference. Only completed arrangement steps and already committed current-step decisions are visible. Track order is reproducibly shuffled at every destination step. Undecided tracks supply provisional natural continuation with a distinct status embedding; incompatible continuation uses an unknown-content representation.

### Actions and training

KEEP advances the source cursor by one step, or maintains silence. SILENCE stops the run. START chooses a source bar and `source_phase` step explicitly. Consecutive KEEP actions determine duration; another START can repeat a fragment within the same bar. There is no fraction-duration head.

The shared destination scorer evaluates legal restart positions and silence; the gate predicts change versus KEEP. Joint action probabilities are decoded with masks, using argmax or seeded sampling. Redundant restarts at the natural continuation position and SILENCE while already off are omitted. Candidate scoring uses content and coordinates, not learned stem IDs.

The loss remains masked gate BCE plus destination cross-entropy on labelled changes. Forced source-end decisions omit gate supervision. Optional `role` IDs remain supplied hints (0 unknown, 1 drums, 2 bass; consistent custom IDs below 16 are allowed).

The four-context architecture uses checkpoint schema 2 and requires retraining older architecture checkpoints. Existing JSON without `steps_per_bar` remains a one-step-per-bar manifest. New step manifests store one action row per destination step and a zero-based `source_phase` on START actions. Placements retain explicit start phases and exact `step_count`; `bar_count` is the rounded-up duration for legacy summaries. Renderers use the exact step duration.

An LLM planner or prompt interface is a future extension. It would need controls and training examples connecting semantic instructions to arrangement behavior; no LLM is part of this implementation.

## Environment

Use Python 3.11 or 3.12 in a clean environment with a compatible PyTorch installation:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,audio,midi]"
```

The native MIDI extra uses the pinned [MusicBERT Hugging Face port](https://github.com/malcolmsailor/musicbert_hf). Install only `.[dev]` for cached-feature training, tests and MIDI export; `.[audio]` adds MERT extraction. A broken pre-existing torchvision installation can break Transformers imports even though this project does not need torchvision; use a clean environment instead of modifying global packages.

## Data and pretrained weights

- [Lakh MIDI](https://colinraffel.com/projects/lmd/): download LMD-full and extract its MIDI tree locally. The loader reads separate parts from each file.
- [Slakh2100-redux](https://zenodo.org/records/4599666): aligned per-source MIDI and FLAC. Preserve the dataset directory structure; `omitted` is excluded. MIDI-derived composition groups determine our splits, not raw file counts. Audit near-duplicate compositions before reporting benchmark results.
- [MusicBERT](https://github.com/microsoft/muzic/tree/main/musicbert): obtain pretrained weights. Extraction accepts a Hugging Face MusicBERT checkpoint directory/model ID. Convert original fairseq `.pt` files once with the command below.
- [MERT-95M](https://huggingface.co/m-a-p/MERT-v1-95M): the audio extraction command downloads the checkpoint or accepts a local snapshot. The checkpoint's published license is CC BY-NC 4.0; dataset/instrument rights remain separate from code licensing.

Downloaded data and weights are not committed. Frozen extractor outputs are cached independently from the trainable projections, with content/preprocessing/revision keys.

```powershell
python -m src.cli.convert_musicbert --checkpoint D:/models/musicbert_base.pt --output D:/models/musicbert-hf --trusted-checkpoint
```

The conversion command uses the pinned port's weight mapping; the trust flag acknowledges legacy fairseq pickle loading. Use original weights from a trusted source. No random encoder is substituted when weights are absent.

## Prepare MIDI training examples

```powershell
python -m src.cli.prepare --source lakh --root D:/datasets/lmd_full --output data/episodes.json --limit 100
python -m src.cli.extract --episodes data/episodes.json --modality midi --midi-checkpoint D:/models/musicbert-hf --cache data/features --output data/midi_inventory.json
python -m src.cli.train --episodes data/episodes.json --inventory data/midi_inventory.json --model-config configs/model.yaml --config configs/train_midi.yaml --output checkpoints/midi
```

`--limit` counts accepted compositions. Preparation writes a report of exclusions. Train and validation splits must both contain usable compositions; the loader refuses empty training/validation datasets. Feature widths are read from the inventory; model width, heads and history come from the model configuration.

The automatic preparation baseline mines exact reusable source bars and their occurrences from multitrack MIDI. It currently accepts constant tempo and notes contained within bar boundaries. It preserves MIDI controller state and excludes pitch bends/program changes it cannot faithfully support. Inference and observed-manifest export support notes sustaining across bar boundaries, but this exact-bar miner deliberately does not fabricate recurrence labels for them.

These are **inferred reuse labels**, not recovered producer edits. A rest is not proof of a muted DAW clip. Automatic preparation expands bar-level edits onto the finer decision grid with KEEP between known boundaries; this preserves the observed coarse trajectory and does not create supervision for new within-bar cuts. Learning within-bar editing requires observed step-level crop/repeat annotations. Validate those with `--source observed --root <episodes.json>`. Use `--steps-per-bar 16` to refine a coarse observed manifest explicitly; omitting it preserves the observed resolution. Artificial fixtures are labelled `constructed_fixture` and used only for mechanical checks.

## Paired audio and mixed training

```powershell
python -m src.cli.prepare --source slakh --root D:/datasets/slakh2100_flac_redux --output data/slakh_episodes.json
python -m src.cli.extract --episodes data/slakh_episodes.json --modality both --midi-checkpoint D:/models/musicbert-hf --cache data/features --output data/paired_inventory.json
python -m src.cli.train --episodes data/slakh_episodes.json --inventory data/paired_inventory.json --initialise checkpoints/midi/best.pt --mix-modalities --config configs/train_midi.yaml --output checkpoints/mixed
```

Paired modalities remain one piece of source material, not two competing candidates. `--mix-modalities` chooses a reproducible modality for each track/episode/epoch. The audio branch must actually be trained before generating audio requests: an untrained branch is rejected by the generation CLI. This is synthetic-render adaptation; competent performance on real recorded audio still needs real-audio training/evaluation and appropriate labels.

`--resume checkpoints/midi/last.pt` resumes an interrupted run with optimiser/RNG state and validates dataset/extractor provenance. `--initialise` starts a new adaptation experiment from model weights without pretending it is the same dataset/run.

Resume permits changing the final epoch count and device while requiring the remaining training options to match. Adaptation retains the previously trained modality declarations and extractor revisions.

## Arrange new source material

```text
Initialise silent tracks and empty history
  -> next destination step
  -> shuffle track order reproducibly
  -> build the four contexts for one track
  -> compute the query, four readouts and fused decision vector
  -> gate/scorer probabilities with legal masks
  -> commit KEEP / SILENCE / START(source bar, source phase)
  -> decide remaining tracks at this step
  -> append the completed step to history
  -> advance the destination clock
  -> coalesce continuous runs and export
```

The source cursor advances only while playing. No automatic wrap or fixed loop length is assumed. A four-bar source at 16 steps per bar offers up to 64 source positions; START and later KEEP/SILENCE choices can reuse any compatible fragment. Source and destination phases can differ.

Create a MIDI-only request, or combine `--midi` and `--audio` arguments:

```powershell
python -m src.cli.request --midi clips/keys.mid --midi clips/drums.mid --bpm 120 --bars 32 --steps-per-bar 16 --output data/request.json
python -m src.cli.extract --request data/request.json --midi-checkpoint D:/models/musicbert-hf --phase-bins 16 --cache data/features --output data/request_inventory.json
python -m src.cli.generate --checkpoint checkpoints/midi/best.pt --request data/request.json --inventory data/request_inventory.json --output runs/arrangement
```

Feature `phase_bins` must be a multiple of `steps_per_bar`; extraction resolution must support the selected decision grid. For a twelve-step grid, extract with `--phase-bins 12` or a common multiple. Batches must share one decision resolution.

Audio requests require complete source bars at the declared tempo and `--audio-starts-on-downbeat`. For irregular timing, provide an explicit source grid in the manifest. Time signature, source/destination grids and audio duration are checked before placing each step.

Generation writes `arrangement.json` and separate MIDI tracks in `arrangement.mid`. Add `--render-audio` for prerendered audio tracks and their audio-only mix. MIDI instrument synthesis remains a separate renderer. Use `--sample --temperature 0.8` for seeded sampling.

Consecutive KEEP decisions coalesce across step and bar boundaries. START boundaries stay explicit. MIDI sustains retrigger at crop entry and release at actual crop ends; they are not retriggered on KEEP. Audio fades occur at actual cuts or discontinuous source spans, not every internal decision step.

## Evaluation and checks

```powershell
python -m src.cli.evaluate --episodes data/episodes.json --inventory data/midi_inventory.json --checkpoint checkpoints/midi/best.pt --split test --output runs/evaluation.json
python -m pytest
python -m ruff check src tests main.py inference.py
python -m ruff format --check src tests main.py inference.py
python -m mypy
```

Evaluation reports teacher-forced decisions, generated rollout behaviour and a fixed-loop baseline. Exact inferred-action accuracy measures the prepared task; it is not alone evidence of musical quality. Use held-out composition groups and listen to full rollouts.

The regular test suite uses deterministic small feature fixtures, explicitly not pretrained representations. Optional adapter checks:

```powershell
$env:RUN_EXTRACTOR_TESTS = "1"  # native adapter with a tiny local test model
$env:RUN_MERT_SMOKE = "1"      # real downloaded MERT weights and supplied audio sample
python -m pytest tests/test_extractors.py
```

## Organisation and migration

- `src/contracts.py`, `src/timing.py`: musical data and coordinates.
- `src/data/`: source adapters, episode construction, replay, batching and JSON.
- `src/features/`: frozen pretrained adapters and versioned caches.
- `src/models/`: temporal/set/history encoders, gate/scorer and masked losses.
- `src/training/`: training, checkpointing, evaluation and baselines.
- `src/inference/`: shared playback transitions, legal choices and generation.
- `src/rendering/`: MIDI and prerendered audio export.
- `src/cli/`: thin command-line entry points.

The previous 25-dimensional mixed-bar VQ pipeline and random bundle retrieval were removed. Its checkpoints are incompatible. `main.py` and `inference.py` forward to the new train/generate commands. The new finite map-style loader supports repeated epochs, replacing the old one-shot streaming iterator. Older documents under `docs/` describe the previous architecture; this README is the executable workflow.

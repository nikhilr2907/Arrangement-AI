# Stem-by-stem arrangement: implementation and refactor specification

> Historical bar-level baseline specification. The implemented within-bar, four-context architecture and its implementation sequence are documented in [README.md](README.md#model-and-agreed-context-plan). That plan supersedes this document's bar-only timing and history-encoding choices.

Status: planning only. Updated 2026-10-14. No model implementation is changed by this document.

This document supersedes the earlier design in `docs/ARCHITECTURE_REFACTOR_PLAN.md` where the two disagree. It incorporates variable-length clips, per-channel playback, bar-aligned crops and repeats, and the binary change gate with conditional destination scoring.

## 1. Objective and first-version boundary

Given a collection of channels, each with one supplied MIDI or audio clip, produce an arrangement by deciding what each channel plays at every destination bar. Clip lengths differ. Each channel can remain silent, continue through its source clip, stop, restart, or jump to a different source bar. Decisions depend on completed arrangement history and the other channels' available current-bar context.

The first version uses pretrained musical feature extractors and a small trainable arrangement model. It does not train a new audio foundation model, require an LLM, train VQ-VAEs, or recreate the note-generation architectures of related papers. Continuous bar summaries are enough to establish the baseline. Structural quantization remains a later experiment.

Source clip identity is a lookup key, never a learned closed-vocabulary class. Inference must work with clips not used in arranger training. This is generalisation to unseen clip instances; it is not a claim of universal zero-shot musical competence.

Initial timing constraints:

- Decision boundaries are destination bar boundaries.
- Source starts are source bar boundaries.
- A continuation consumes the next compatible complete source bar.
- A crop consists of a consecutive run of consumed source bars ending at a decision boundary.
- A repeat starts a new run from an earlier source bar.
- No crossfade or overlap of two placements in the same channel, except renderer boundary fades.
- Inputs may have variable tempo and meter maps, but unsupported source/destination mappings are rejected or masked explicitly.
- Mid-bar placements, partial bars, automatic transposition and tempo adaptation are extensions, not hidden behaviour.

Clip support must not imply that every MIDI/audio representation is interchangeable without adaptation. Native MIDI/audio feature spaces need separately learned projections and mixed-modality training.

## 2. Decision semantics: pin these down before coding

### 2.1 One channel, one source clip

For version 1, a channel owns one clip. A four-bar keys clip and a two-bar drum clip can be supplied together. The keys channel never chooses a drum clip. Multiple clips per channel can later add a clip ID to the destination choice without replacing the model's scorer.

At the start of destination bar t, channel state is either:

- `SILENT`: no active source playback.
- `PLAYING`: a source clip is assigned and `next_source_bar` identifies the bar that KEEP would consume now.

`next_source_bar` is a cursor, not a destination position. Source and destination bar numbers must have different types/field names.

### 2.2 Action family

The final decision has one of three forms:

| Action | Meaning at destination bar t | State after committing the bar |
|---|---|---|
| `KEEP` while silent | Remain silent | Silent |
| `KEEP` while playing | Play `next_source_bar` | Playing, cursor advanced by one |
| `SILENCE` | Stop playback and play no new source content this bar | Silent; no implicit source advancement |
| `START(j)` | Play source bar j | Playing, next cursor j + 1 |

Stopping ends the current source run. Re-entry selects a source bar explicitly. Silence does not invisibly advance or pause a hidden playhead. This removes ambiguity about what an entrance means.

KEEP is invalid at the end of a playing clip or when the next source bar is incompatible with the destination meter/duration. KEEP while silent remains valid, including at BOS. Otherwise an empty arrangement would be forced to start something.

### 2.3 Binary gate and conditional destination head

The binary target is **change the channel's playback state versus keep its natural evolution**. It is not an independent presence label and is not necessarily a crop label: entrances, exits and jumps are all changes.

The two heads share the same query/context representation:

- `change_logit: Tensor[B]` predicts change versus KEEP.
- `destination_logits: Tensor[B, J]` scores valid change destinations: SILENCE and START(j).

J is variable because source clip lengths vary. A shared scoring function evaluates candidate destination embeddings; there is no fixed output layer with one learned class per clip/bar ID.

For a valid KEEP:

    p_change = sigmoid(change_logit)
    P(KEEP) = 1 - p_change
    P(destination j) = p_change * softmax(masked_destination_logits)[j]

If KEEP is invalid, destinations alone define the distribution. At inference, compare or sample from the combined action distribution, rather than always thresholding the gate at 0.5 and then taking a destination argmax. The latter is a different policy and need not produce the most probable joint action.

Represent probabilities in log space for stable decoding:

    log P(KEEP) = logsigmoid(-change_logit)
    log P(j) = logsigmoid(change_logit) + log_softmax(destination_logits)[j]

Mask redundant choices. While silent, SILENCE is redundant with KEEP. While playing, START at exactly the natural next source bar is normally redundant with KEEP. Retain an explicit retrigger variant only if the renderer and labels actually distinguish it; do not claim that an identical next bar is a cut based on invented metadata.

If there is only one legal action, take it directly. Do not construct all-masked softmax rows. The candidate builder and policy together must define this case.

### 2.4 Example

For a four-bar clip A:

```text
Destination bar:  0         1       2         3       4        5
Action:          START(0)  KEEP    START(0)  KEEP    SILENCE  KEEP
Played source:   A[0]      A[1]    A[0]      A[1]    --       --
```

The first two bars are played twice, then the channel stops. Presence is `[1, 1, 1, 1, 0, 0]`, derived from playback actions. The crop extent is two bars on each run. There is no separate continuous crop-end regression.

## 3. Musical time and audio/MIDI boundaries

### 3.1 Canonical time

Use integer ticks in quarter-note units with an explicit PPQ. Preserve original MIDI timing and microtiming; bar-level decisions do not require quantizing every note to sixteenths. Tempo affects seconds, meter affects bar boundaries.

For numerator n, denominator d and PPQ q, a regular bar spans:

    bar_ticks = q * n * 4 / d

Require an integer result under the chosen PPQ. Beat grouping is separate from duration: 6/8 and 3/4 can have equal tick lengths but different grouping. Keep numerator, denominator and optional beat groups in the model metadata.

Canonical conversion functions:

- `tick_to_seconds(tick: Tick, tempo_map: TempoMap) -> float`
- `seconds_to_tick(seconds: float, tempo_map: TempoMap) -> Tick`
- `bar_span(index: BarIndex, grid: MusicalGrid) -> BarSpan`

Conversions integrate piecewise tempo, preserve monotonic boundaries and specify rounding. Audio sample offsets are derived from source seconds and actual sample rate, not from an assumed fixed samples-per-bar value.

### 3.2 First-version source/destination compatibility

Maintain a source grid per clip and a destination grid per request. Version 1 accepts exact musical-grid compatibility and checks audio-duration tolerance. MIDI can map source ticks to destination ticks when bar spans and grouping agree. Audio cannot change tempo merely by changing metadata; without an enabled time-stretch backend, require compatible durations.

Meter changes and pickups are represented in contracts. Preparation initially filters to complete supported bars and records exclusions rather than pretending all inputs are 4/4. The grid implementation should be general even if the first curated training experiment is restricted to 4/4.

Do not truncate all channels to the shortest clip. A short clip remains available for repeat decisions while longer clips continue. Incomplete source bars must be rejected or explicitly masked in the first version.

### 3.3 Notes crossing a cut

A MIDI note is an interval `[onset_tick, offset_tick)`. The offset is its end time, not an activity flag. Activity, onset and ending are different concepts.

MIDI rendering/export needs two distinct behaviours:

- Consecutive KEEP bars are coalesced into one continuous source interval, preserving notes crossing internal bar boundaries without retriggering them.
- A new START into a sustained note includes and retriggers the intersecting note at the new crop start; a crop end clamps/releases notes that cross its boundary under the version-1 hard-cut policy.

Use a deterministic policy for sustain pedal, program events and other supported controllers. Replay controller state at crop entry and release it at crop exit. Initially reject unsupported pitch-bend/controller cases or preserve them with explicit boundary state; never silently drop them while claiming faithful rendering.

Audio KEEP intervals are also concatenated from the original continuous source without per-bar fades. Apply short boundary fades only to actual discontinuities. Dry-source truncation and effect-tail overlap are different policies; version 1 chooses and records the former rather than letting plugin tails redefine labels.

## 4. Existing work we build on

This is a task-specific extension using established feature extractors and data tooling, not a reproduction of one arrangement paper. Do not label the proposed method novel or state of the art without a dedicated comparison.

| Component | Starting point | Reuse boundary |
|---|---|---|
| Native MIDI representations | [MusicBERT, ACL 2021](https://arxiv.org/abs/2106.05630) and [official code/checkpoints](https://github.com/microsoft/muzic/tree/main/musicbert) | Adapt its actual preprocessing and hidden-state interface. Its accompaniment-suggestion task is related; it does not already implement this crop/repeat scheduler. |
| Audio representations | [MERT paper](https://arxiv.org/abs/2306.00107) and [95M checkpoint](https://huggingface.co/m-a-p/MERT-v1-95M) | Frozen musical features. The checkpoint specifies 24 kHz and 768-dimensional hidden states; derive runtime shapes from the loaded configuration. |
| MIDI parsing | [pretty_midi](https://craffel.github.io/pretty-midi/) plus explicit tick/tempo-map handling | Reuse note/instrument parsing. A parser/tokenizer is not a pretrained encoder. |
| Paired MIDI/audio | [Slakh2100-redux](https://zenodo.org/records/4599666) | Use aligned instrument sources and provenance; do not confuse render alignment with DAW placement labels. |
| Future structural codes | [Structured Arrangement, NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/b95cb2d3f647dae571203bab285077e7-Paper-Conference.pdf) | Inspiration for rhythmic latent priors, not the initial implementation. It generates symbolic notes, not these supplied-clip decisions. |
| Explicit composition representation | [TOMI](https://arxiv.org/abs/2506.23094) | Related clip/section/track organisation. Its language/metadata planning is different from the learned action scorer. |

MusicBERT's official README links checkpoint downloads and notes fairseq compatibility requirements. First prove extraction on one real multitrack file in an isolated environment. Do not assume a maintained plug-and-play Hugging Face adapter. The upstream repository was archived in October 2026; freeze a tested revision and treat extraction as an offline boundary.

If native MIDI extraction proves impractical, an explicitly named `rendered_midi_mert` experiment can render MIDI and reuse MERT. That still accepts MIDI files but changes the representation path. It must not silently substitute for `native_musicbert` or claim separate native MIDI/audio encoders are implemented.

## 5. Download and local dataset layout

Download manually; data acquisition is separate from code refactoring.

1. [Lakh MIDI download page](https://colinraffel.com/projects/lmd/): choose **LMD-full** for volume. It contains 176,581 MIDI files, including invalid files that need filtering. MIDI parts are inside each file; they are not downloaded as separate audio stems. The original source link is [lmd_full.tar.gz](http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz). The endpoint was not successfully fetched during planning; use the official page for mirrors if it fails.
2. The same page offers a **Clean MIDI subset** for a smaller first experiment. It is not a guarantee of clean multitrack structure or absence of composition duplicates.
3. [Slakh2100-redux on Zenodo](https://zenodo.org/records/4599666): paired per-instrument MIDI/FLAC; the listed archive is 104.3 GB. Keep audio compressed. Exclude the `omitted` duplicate group and audit overlap with Lakh across splits.
4. Download MusicBERT weights through the updated links in its [official README](https://github.com/microsoft/muzic/tree/main/musicbert). MERT weights are configured by model ID/revision or a local snapshot.

Suggested configurable layout:

```text
data/
  raw/lakh/lmd_full/              # original extracted MIDI tree
  raw/slakh/                     # original train/validation/test layout
  manifests/                     # validated source records and split groups
  prepared/                      # clip libraries and decision-label episodes
  features/                      # frozen extractor outputs
checkpoints/
runs/
```

Paths can point to another drive. No loader assumes this machine's username or that all data fits in memory. Keep assets/caches out of Git and keep this plan tracked: the existing `.gitignore` excludes `docs/`, which is why the detailed specification lives at repository root.

## 6. Data records and type contracts

Use frozen dataclasses for immutable records, validated at the preparation boundary. Use explicit mutable state only in inference. Large payloads are file references until extraction/rendering requires them.

Suggested aliases:

```python
ClipId = NewType("ClipId", str)
ChannelId = NewType("ChannelId", str)
CompositionId = NewType("CompositionId", str)
SourceBarIndex = NewType("SourceBarIndex", int)
DestinationBarIndex = NewType("DestinationBarIndex", int)
Tick = NewType("Tick", int)
FloatArray = NDArray[np.float32]
```

These remain ordinary runtime values; boundary validators enforce ranges and invariants. Do not pretend `Tensor` annotations alone enforce dimensions or dtypes.

| Record | Required fields and invariants |
|---|---|
| `Meter` | Positive numerator/denominator, optional beat groups; validate their sum and interpretation. |
| `TempoChange` / `TempoMap` | Ordered tick/BPM entries, positive BPM, explicit starting tempo. |
| `BarSpan` / `MusicalGrid` | Ordered tick boundaries, meter, optional source-second boundaries; no negative or overlapping spans. |
| `MidiAsset` | Path, part selector, source PPQ, parser provenance; separate distinct parts/program changes. |
| `AudioAsset` | Path, actual rate/channel layout, grid alignment; no anonymous float32 byte assumption. |
| `Clip` | ID, composition group, source channel, role metadata, source grid, optional MIDI/audio assets, complete playable-bar count. Paired modalities describe one object. |
| `Channel` | Request-local identity, assigned clip ID, role/instrument metadata, optional fixed/seed actions. |
| `ClipLibrary` | Mapping of IDs to immutable clips; channels reference members. |
| `SourceDestination` | SILENCE or START with clip/source-bar reference; local candidate index is only a batch index. |
| `PlaybackAction` | Tagged KEEP/SILENCE/START action; START includes a valid source reference. |
| `ChannelPlaybackState` | Silent or playing with next source cursor; no invalid combinations of fields. |
| `BarPlayback` | Destination bar, channel, selected source span or silence, action, decision confidence. |
| `Placement` | Coalesced destination/source intervals; preserve starts/restarts even if adjacent content sounds identical. |
| `ArrangementRequest` | Channels, clip library, destination grid, length, optional seed/section hints and reproducible decode seed. |
| `ArrangementResult` | Bar decisions, placements, timing, asset references, model/feature provenance and export metadata. |
| `LabelProvenance` | Observed project, inferred MIDI recurrence, constructed schedule, or activity-only; confidence/ambiguity mask. |
| `DecisionTargets` | Binary change label/mask, acceptable destinations/mask, auxiliary played/not-played flag and provenance. |

Represent tagged actions with an enum plus validated payload, or a union of small frozen dataclasses. Select one pattern consistently. Avoid a nullable field collection allowing `SILENCE` with an arbitrary source position.

Do not pass training targets inside model input dataclasses. Metadata needed for rendering must not become a model feature merely because it appears in the record.

## 7. Pretrained extraction and feature caching

### 7.1 Frozen offline extraction

Separate heavy extraction from arrangement training. `model.eval()` and inference mode prevent dropout/gradients. Cache raw frozen features before trainable projection; caching projected values would freeze the projection unintentionally.

Encode clips independently from the target arrangement. A bidirectional extractor can inspect a supplied source clip because the entire clip is available at inference. It must not inspect an actual future arranged mix while creating a history token.

MERT adapter responsibilities: explicit mono conversion for feature extraction, resample to checkpoint rate, bounded-duration chunks, retain frame timestamps and valid lengths, fixed layer selection, deterministic handling of chunk overlap and boundaries. Its hidden states are frame features, not guaranteed exact rhythm or pitch labels.

MusicBERT adapter responsibilities: official checkpoint-compatible OctupleMIDI preprocessing, token-to-note/time alignment, length limits/chunking, deterministic empty-bar handling and supported instrument/meter values. Confirm the checkpoint's actual token-level output granularity rather than assuming every field is a separate note embedding.

Preserve entering sustains and ending notes in MIDI playback metadata. A note-onset-based tokenisation may not supply embeddings in a sustain-only bar. Reuse relevant spanning-note features where supported and add an explicit learned empty/sustain state; do not fabricate an onset just to satisfy the encoder.

### 7.2 Temporal representation

Store raw temporal features and their musical coordinates. For the compact baseline, resample/pool them into a configurable number P of within-bar phase bins, for example P = 16. Native note/sample timing remains intact for export. Model bins are a compression choice, not a claim that all music is on a sixteen-step grid.

Also cache a whole-clip summary to let candidate scoring inspect broader phrase context. Each destination candidate uses the source bar it would play now, plus clip-level context, source position and remaining length. Avoid relying on a whole-clip mean alone.

The feature cache key includes content digest, asset/part selector, crop/context policy, extractor ID/checkpoint revision, preprocessing version, layer selection, sample rate, timing map, pool resolution and dtype. Rendered MIDI adds renderer/preset/version/seed provenance. Revalidate metadata and write atomically; a renamed source path must not cause collisions.

### 7.3 Simple modality bridge

Start with independent trainable linear projections plus normalisation, frozen extractors, and learned MIDI/AUDIO embeddings. No role prediction network or shared-codebook alignment is required to make the first forward pass work.

Train MIDI-only first. Add paired Slakh batches with the same action schedule rendered through either modality. Randomly replace per-channel inputs with the paired modality so a single request can be interleaved. Initially choose one modality per paired clip; fusion is deferred.

Optional alignment compares projected structural summaries for paired bars with cosine/MSE or a contrastive loss. Use only documented paired examples and test whether alignment helps. Do not force acoustic timbre to equal symbolic structure. Ground-truth silence and editorial actions are not automatically inferable from sample-bank audio energy.

Mode-specific feature distributions, real-audio domain shift and class imbalance require separate validation. The MIDI-only checkpoint is not an audio-capable result until audio projections and mixed-context decisions have been trained/evaluated.

## 8. Obtaining supervision without inventing editorial ground truth

This is the main research dependency. Lakh and Slakh generally give complete performances, not a producer's original reusable clips and edit history. A note-free bar or silent source segment does not prove a clip was muted in the original DAW.

Use three explicit preparation regimes through the same output records:

### 8.1 Observed placements

Best labels come from project exports or curated manifests containing original clips, destination spans and source offsets. Convert each channel's placements into bar actions. Keep IDs and repeated source references, and allow adjacent placements with genuine retriggers to remain distinct.

If a placement/cut is off-grid under the chosen policy, exclude that decision with a recorded reason or defer the whole example. Do not round cuts silently and assert an exact label.

### 8.2 MIDI-derived arrangement episodes

Build reusable source clips from complete, audible note phrases and repeated MIDI passages, then map their occurrences back to destination bars. Fingerprints can use symbolic note intervals, duration and relative timing because these are labels/normalisation, not the manual acoustic feature extractor being replaced.

Prefer exact matches first. Represent sustained-note carry and controller context in matching. A rest inside a phrase remains an intentional source rest; it is not automatically a channel-level SILENCE action. Isolate stretches of no musical activity as activity evidence, and use phrase boundaries only where the construction supports them.

Construct channels from MIDI parts, not from role classes. Two guitars remain two channels. Split program changes only under a declared policy. Exclude malformed, excessively long, single-instrument-only or unsupported examples according to logged preparation filters rather than swallowing all exceptions.

Canonicalise equivalent actions: choose KEEP when source progression is natural; label START when occurrence mapping jumps/repeats; SILENCE stops a run. If several source starts reproduce the same target bar and context, store an acceptable destination set rather than training contradictory unique labels.

Reject or downweight episodes where the whole song becomes the supplied source clip and the target is always 'read the next original bar'. Hide original absolute song offsets from model features and reindex reusable clips locally. This regime is inferred supervision, not a claim to recover actual DAW edits.

### 8.3 Constructed schedules

Generate known crop/repeat/silence schedules from small libraries for state-machine verification and bootstrap training. Use recorded seeds and explicit construction rules. These labels teach execution mechanics and may encode the generator's preferences; they do not establish human arrangement quality.

Keep this evaluation separate from observed and inferred data. A model that memorises synthetic schedule rules has not demonstrated musical arrangement ability.

### 8.4 Activity-only examples

Where data only supports placed/not-placed supervision, mark gate/destination labels unknown and use a derived presence likelihood rather than inventing a source pointer:

    P(present) = P(KEEP) * I(KEEP plays)
               + sum_j P(destination j) * I(destination j plays)

Here 'plays' means a source interval is assigned; a source interval containing an intentional rest can still be placed. Audible activity is a different target. If only note/audio activity labels exist, replace the indicators with known candidate-activity indicators over the bar, including entering sustains, and name that objective activity prediction. Do not equate an inaudible source bar with an absent placement.

Apply a binary presence/activity loss only where the corresponding observational label is meaningful. Do not substitute detector confidence for editorial certainty; log label type and confidence. Activity-only training can help the model learn layering but cannot by itself teach crop source choices or repeats.

### 8.5 Splits and sampling

Split by composition before windows, clip mining, alternate renders or pair creation. Exact file hashes alone do not group different MIDI versions of the same composition; use available metadata, symbolic similarity checks and a recorded deduplication policy. Audit Slakh/Lakh parent overlap.

Never use validation/test clips as training negatives or donor material. Per-song bar windows do not count as independent songs. Weight sampling to prevent long arrangements, many-track compositions or MIDI volume from dominating all updates. Track effective counts and score each modality/provenance regime separately.

## 9. Dataloaders and preparation services

Use small interchangeable source adapters, a preparation pipeline and one cached decision dataset. A loader loads source records; it does not extract foundation-model features, train models or choose playback actions.

### 9.1 Source adapters

`LakhSource` scans local MIDI trees and returns composition records lazily. It uses stable content/group IDs, preserves original track names, tempo/meter/controller data and records typed parsing failures. It does not need Hugging Face or a custom byte format.

`SlakhSource` pairs source-level MIDI/FLAC through the dataset's actual metadata and split directories. Preserve source IDs and paired identities. Ignore `omitted` unless a documented deduplication experiment says otherwise. Timing comes from paired MIDI/alignment, not per-stem independent beat detection.

`ManifestSource` handles user-supplied clips, grids, channels and optional observed placements from a validated JSON/JSONL schema. Inference can load a library from the same clip schema without a song record.

`MidiReader` uses established MIDI parsing and retains tick-accurate semantics. `AudioReader` reads actual sample rates, keeps a consistent samples-by-channels convention for rendering and creates mono extraction views deliberately.

### 9.2 Preparation

`EpisodeBuilder` composes clip extraction/matching, action canonicalisation and provenance. A `PreparationConfig` names the regime and filters. Version 1 implements exact MIDI matching plus explicit observed manifests; approximate recurrence and automatic section discovery are separate experiments.

`PreparedEpisode` includes channels, library references, destination grid, canonical bar actions or partial labels, composition group and provenance. No embeddings are needed to inspect or render it.

`FeaturePreparer` batches referenced clips by extractor and writes versioned features. It reports skipped unsupported assets and never silently reduces the dataset to zero. Extraction should run outside DataLoader workers, especially on CUDA/Windows.

### 9.3 Cached decision dataset

Use `ArrangementDecisionDataset(Dataset[DecisionExample])` as a map-style dataset over a prepared index. Each example is one target channel decision at one destination bar, with an H-bar history window and the current within-bar decision prefix.

Construct complete bar orders once per episode/epoch so the teacher-forced prefix is consistent across channels. Seed ordering deterministically using run seed, epoch and episode ID. Do not let mutable worker-local random state change the targets unpredictably.

The collator pads history/channel/source-candidate axes independently, packs raw feature blocks by extractor, and returns separate `ModelInputs` and `DecisionTargets`. Padded entries are invalid and have no loss. Stable IDs are retained outside tensors to decode predictions.

`DataLoader` receives a seeded generator, explicit worker initialisation, finite batch sizing and a top-level pickleable collator. Test spawn-based Windows workers. Do not capture CUDA models or open non-pickleable reader handles in worker closures. Use lazy feature reads and optional bounded CPU caching.

### 9.4 Public preparation interfaces

```python
class ArrangementSource(Protocol):
    def iter_compositions(self) -> Iterator[CompositionRecord]: ...

class FrozenFeatureExtractor(Protocol):
    @property
    def spec(self) -> ExtractorSpec: ...

    def encode(self, clips: Sequence[Clip]) -> list[ClipFeatures]: ...

class LakhSource:
    def __init__(self, root: Path, config: SourceConfig) -> None: ...
    def iter_compositions(self) -> Iterator[CompositionRecord]: ...

class SlakhSource:
    def __init__(self, root: Path, split: DatasetSplit) -> None: ...
    def iter_compositions(self) -> Iterator[CompositionRecord]: ...

class EpisodeBuilder:
    def build(self, composition: CompositionRecord) -> list[PreparedEpisode]: ...

class ArrangementDecisionDataset(Dataset[DecisionExample]):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> DecisionExample: ...
```

These are proposed contracts; ellipses in this plan are not permission to leave implementation stubs in the completed refactor.

## 10. The small trainable model

### 10.1 Feature projection and bar encoding

`FeatureProjector` has a projection/normalisation per extractor output space. It converts raw bins to common D-dimensional tokens and adds modality, role and musical-coordinate embeddings. Suggested initial D = 128 or 256, configured rather than hardcoded through modules.

`SourceBarEncoder` pools the P temporal bins through a small attention/pooling block and learned summary query. Preserve phase order through positional coordinates. It returns one D-vector per source bar for efficient baseline scoring; raw/cropped temporal bins remain available for later detailed compatibility attention.

For a source jump, `DestinationEncoder` combines the source-bar summary, whole-clip summary, source location/remaining length and action-type embedding. SILENCE uses a learned representation. Track identity is request-local metadata; role/instrument embeddings carry reusable semantics. Never learn absolute clip IDs.

### 10.2 Completed history

`BarContextEncoder` attends over the actual committed channel content for each completed destination bar. Channels form a set; within-clip phase positions do not. Its learned summary token produces one global bar vector. Preserve silent-channel states as state tokens so the model can learn absences as well as sounding content.

The target channel's own previous playback summaries are also retained. A single pooled global vector must not erase which source material the queried channel was using. `HistoryEncoder` receives global and target-channel history streams, with bar positions and stream-type embeddings, and applies causal self-attention.

For BOS, use an explicit learned BOS token with a valid mask. For truncated windows, include absolute destination position and playback state; do not mislabel every window's first bar as the beginning of the piece. Keep H configurable, for example 16 or 32 completed bars.

Only bars before t may contribute to the historical memory for a decision at t. Candidate source content can be available in full because it is an inference input, not a future arranged answer.

### 10.3 Current other channels

Generate a random channel decision order per training bar. At inference use a reproducible order, optionally sampled with the request seed. Later decide whether a fixed musical-role order is preferable using validation; do not assume it is inherently correct.

For the queried channel:

- Already decided channels expose their committed current-bar content or silence.
- Fixed user selections expose their actual current-bar content.
- Undecided channels expose state and a provisional KEEP projection derived solely from previous state, explicitly marked `UNDECIDED`. If KEEP is invalid, expose that fact and a null content token.
- The query channel does not expose its target action/current ground-truth content.

`CurrentContextEncoder` treats these other-channel tokens as a set and attends over them, including decided/undecided status. It never reads current-bar ground-truth choices for channels that have not yet been committed in the chosen order.

Random ordering is augmentation; sequential decoding can still produce order-dependent joint arrangements. Do not claim global channel permutation invariance. The encoder/scorer should be equivariant to reordering an already-specified input set, while the complete policy's ordering is measured separately.

### 10.4 Query and heads

`ArrangementModel` creates a target-channel query from:

- Role/instrument metadata and modality.
- Silent/playing state, next-source cursor and remaining source length.
- The musical content KEEP would play, if valid.
- Destination bar position, meter, tempo and optional section/phrase hints.
- Attention to historical memory and current other-channel context.

`ChangeHead` is a small linear/MLP scalar head. `DestinationScorer` computes one scalar per proposed destination using the same weights for all proposals, for example an MLP over query, destination, their elementwise product and difference. Candidate count affects tensor length, not parameter count.

No independent presence head is required. Presence follows from the decoded action distribution and can supply an auxiliary loss where labels are available.

### 10.5 Minimal class inheritance

```text
torch.nn.Module
├── FeatureProjector
├── SourceBarEncoder
├── BarContextEncoder
├── HistoryEncoder
├── CurrentContextEncoder
├── DestinationEncoder
├── ChangeHead
├── DestinationScorer
├── ArrangementLoss
└── ArrangementModel             # composes the modules above

torch.utils.data.Dataset[DecisionExample]
└── ArrangementDecisionDataset

typing.Protocol
├── ArrangementSource           # LakhSource, SlakhSource, ManifestSource satisfy it
├── FrozenFeatureExtractor      # MusicBertExtractor, MertExtractor satisfy it
└── MidiRenderer                # optional rendering adapters satisfy it

Plain dataclasses/services
├── data and tensor contracts
├── EpisodeBuilder / FeaturePreparer / FeatureStore
├── PlaybackStateMachine / CandidateBuilder / DecodePolicy
├── ArrangementGenerator
└── Trainer / Evaluator / CheckpointManager
```

Protocol implementers use structural typing and need not inherit the protocol. Do not add a common 'BaseEverything' class. Trainer regimes are configuration and objectives, not separate trainer subclasses. The generator owns inference orchestration and contains the model; it does not inherit Trainer or load checkpoints through a training object.

## 11. Tensor contracts and exact model boundaries

Use one query per sample initially. B = decision samples, H = completed history bars, C = padded other channels, J = change destinations, P = temporal phase bins, D = trainable model width, F_m = raw width for extractor m.

The collator deduplicates raw feature references within a batch:

| Field | Dtype / shape | Meaning |
|---|---|---|
| `FeatureBlock.values` | float `[N_m, P, F_m]` | Raw bins grouped by extractor |
| `FeatureBlock.valid` | bool `[N_m, P]` | Real temporal bins |
| `FeatureBlock.phase` | float `[N_m, P]` | Within-bar musical positions |
| Global feature references | int64 indexes into a batch feature table | Join blocks after projection |
| History content references | int64 `[B, H, C]` with separate valid flags | Actual previous channel/source spans |
| Target history references | int64 `[B, H]` with state flags | Queried channel's previous playback |
| Current other references | int64 `[B, C]` with status flags | Committed/provisional current content |
| KEEP content reference | int64 `[B]` plus `keep_valid` | Natural next source span or silent state |
| Destination references | int64 `[B, J]` plus type/position metadata | SILENCE or source starts |
| `destination_valid` | bool `[B, J]` | Legal candidate choices |
| `change_logit` | float `[B]` | Change gate |
| `destination_logits` | float `[B, J]` | Raw destination scores |
| `gate_target`, `gate_valid` | float/bool `[B]` | Binary supervision where meaningful |
| `acceptable_destinations` | bool `[B, J]` | One or several equivalent correct destinations |
| `destination_target_valid` | bool `[B]` | Whether destination supervision exists |

Absent feature references use explicit null flags and learned state embeddings. Never accidentally gather padded `-1` indexes as the final real feature. All masks have one public convention: `True` means valid. Convert explicitly at PyTorch attention boundaries if its API expects `True` to mean blocked.

Padded raw bins are excluded from pooling. Empty/sustain-only MIDI spans get declared state handling rather than an all-masked attention row. Histories use BOS when empty; C = 0 uses an empty-context token. Shapes must work with B = 1 and J = 1, without an unconstrained `squeeze()` dropping batch dimensions.

The collator may pad J to at least one for tensor convenience even when a particular sample has no change destinations. Its valid mask stays false. Branch on action validity before softmax/attention; do not normalise an all-masked row and then multiply its NaNs by a zero mask.

Proposed model methods:

```python
class SourceBarEncoder(nn.Module):
    # Input: projected features float[N, P, D], valid bool[N, P], phase float[N, P].
    # Output: source-bar embeddings float[N, D].
    def forward(self, features: Tensor, valid: Tensor, phase: Tensor) -> Tensor: ...

class HistoryEncoder(nn.Module):
    # Input: global/target histories float[B, H, D], valid bool[B, H].
    # Output: causal history memory float[B, L, D], valid bool[B, L].
    def forward(self, inputs: HistoryInputs) -> EncodedMemory: ...

class DestinationScorer(nn.Module):
    # Input: query float[B, D], candidates float[B, J, D].
    # Output: destination logits float[B, J].
    def forward(self, query: Tensor, candidates: Tensor) -> Tensor: ...

class ArrangementModel(nn.Module):
    # Input: ModelInputs with ragged feature blocks and playback context; no targets.
    # Output: DecisionLogits with change float[B], destination float[B, J].
    def forward(self, inputs: ModelInputs) -> DecisionLogits: ...
```

When trainable projections update, history summaries must be recomputed from cached raw features during training. During inference, projected source bars can be cached once because model weights are fixed. Do not silently cache stale trainable bar/history encodings across optimisation steps.

## 12. Training losses

Let g_i be the gate logit, y_i the binary change target and v_i indicate a genuinely supervised choice where both KEEP and a change are possible. Let u_i indicate a supervised change destination. Let A_i be the acceptable destination set after equivalence handling, contained in the legal candidates.

### 12.1 Gate loss

    L_gate = sum_i v_i * BCEWithLogits(g_i, y_i) / sum_i v_i

If no gate decisions are supervised in a batch, return a graph-connected zero rather than divide by zero. Forced actions are not informative gate examples. An all-silent start still has a supervised gate choice when valid START destinations exist.

Compute BCE/CE on selected finite supervised rows before reduction. A zero label mask is not sufficient protection if an earlier all-masked softmax already created NaNs. Build graph-connected zeros from finite raw outputs, not logits already filled with negative infinity.

KEEP may be silent or sounding; the gate learns change, not presence. Report entrance, exit, repeat/jump and continued-play cases separately so a high aggregate KEEP accuracy cannot hide poor editing decisions.

### 12.2 Conditional destination loss

For a unique correct destination, use cross-entropy over legal destinations. For equivalent correct destinations use:

    L_dest_i = -log(sum_{j in A_i} softmax(masked_scores_i)[j])
    L_dest = sum_i u_i * L_dest_i / sum_i u_i

u_i is true only for known change destinations, including forced changes at source ends. It is false on KEEP and ambiguous/unknown destination examples. The acceptable set must intersect the legal set; otherwise preparation fails with a diagnostic instead of hiding the label behind an invalid-action mask.

Do not apply the gate probability inside conditional CE: it is already trained by the gate loss. Forced-change rows omit the gate factor from their action likelihood. Keep metrics for the full joint action distribution as well as each head.

### 12.3 Optional supervision

    L = lambda_gate * L_gate + lambda_dest * L_dest
      + lambda_presence * L_presence
      + lambda_alignment * L_alignment

Only enable losses backed by available labels/pairs. For activity-only labels, derive presence from the joint action distribution as in Section 8.4. Compute its binary likelihood stably and keep it distinct from gate labels. Alignment starts disabled until basic extraction and arrangement training work.

The loss object returns a typed `LossBreakdown` with differentiable total/components and counts. Epoch logging aggregates sums/counts, not an unweighted mean of variable-size batch means. The configured normalisation and lambda choices must be saved so experiments are reproducible.

Heavy class reweighting/focal loss changes calibration; do not add it blindly. Start with logged class ratios, balanced episode sampling and ordinary BCE/CE. If weights are needed, calibrate the resulting joint decode distribution on validation data.

## 13. Training stages and loop

### 13.1 Preparation gate

Before long training, prove:

1. One Lakh file parses into distinct channels with reliable timing.
2. One prepared crop/repeat episode round-trips into actions and back into placements.
3. MusicBERT extraction produces finite features with documented alignment and deterministic masks.
4. A small paired Slakh sample supplies corresponding MIDI/audio source spans.
5. A decision batch yields finite logits and losses, including forced-action and silence cases.

An extractor compatibility failure is a boundary failure, not an excuse to replace pretrained features with random embeddings and call the pipeline complete.

### 13.2 MIDI-first arranger training

Frozen MusicBERT features feed trainable projection/bar encoders, history/current context, gate and destination scorer. Train on prepared native MIDI episodes. This is pretraining the arranger on symbolic data, not training MusicBERT again from scratch.

Initial loop:

```text
for epoch:
    set epoch seed and episode/channel-order sampling
    for inputs, targets in training loader:
        move only tensor fields to device
        project cached frozen features
        encode actual historical bars and current decision prefix
        compute gate and candidate logits
        compute masked losses
        backpropagate, clip gradients, update optimiser
        record sums/counts and invalid-data diagnostics
    evaluate held-out teacher-forced decisions
    periodically run complete generated arrangements
    save last checkpoint; update best checkpoint by declared metric
```

Use AdamW, configurable learning rate/warmup, optional mixed precision supported by the device, gradient clipping and reproducible seeds. Defaults are experiment settings, not claimed optimal hyperparameters. Test a real small overfit case before launching the full corpus.

### 13.3 Paired and mixed-modality adaptation

Initialise shared model weights from the MIDI checkpoint. Add/train the audio projection using frozen MERT features. Mix MIDI-only, audio-only paired renditions and interleaved channel inputs with a controlled batch sampler. The underlying placement schedule is identical for paired variants.

First train the new projection and optionally gate/scorer while shared history modules are fixed; then unfreeze the small arrangement network at a lower learning rate if validation supports it. This schedule is configurable, not a requirement to add several new encoder networks.

Do not let paired examples cross composition splits. Keep pretrained-extractor revisions fixed, and record if their original pretraining datasets may overlap evaluation material; arranger-level split hygiene alone does not establish a fully unseen foundation-model benchmark.

### 13.4 Real-audio training

Use observed manifests or explicitly qualified inferred/audio-activity labels. Real stems cannot be assumed to have MIDI-equivalent note labels. Train available objectives, compare against paired synthetic audio and validate transfer on held-out real clips.

If full real-audio crop/repeat supervision is missing, clearly report that only presence/activity adaptation was trained. Do not manufacture exact source-jump labels from a silent/full-length WAV timeline.

### 13.5 Teacher forcing and rollouts

Teacher forcing supplies actual past actions and already-decided current channels. Inference supplies generated history. Evaluate both from the outset.

Do not immediately apply supervised targets to arbitrary corrupted/generated states: the canonical next action can become invalid or no longer be correct. Scheduled sampling requires recomputed reachable targets, curated correction episodes or an explicitly different objective. Defer it until basic state transitions and labels are reliable.

History windows carry the true playback state at their first decision. Derive that state from the preceding schedule, not from resetting each window. Record window boundaries and preserve prior motif information available within H; long-range phrase memory is optional future work.

## 14. Inference and decoding

The generator uses the exact same `PlaybackStateMachine`, `CandidateBuilder`, timing metadata and context builder as preparation/training. A separate implementation of cursor updates is a preventable source of train/inference mismatch.

### 14.1 Algorithm

```text
validate request and source/destination grids
load or extract features for supplied clips
project/encode source bars once with fixed model weights
initialise silent channel states or validated seed states
initialise BOS and completed-bar history

for destination bar t:
    choose reproducible channel order
    current decisions = fixed user decisions for t
    for each undecided channel k:
        derive KEEP content/validity from channel state
        enumerate legal changes for this channel's clip
        build context from completed history and current prefix
        score change gate and destinations
        combine legal action log-probabilities
        choose action with DecodePolicy
        commit this channel's BarPlayback and update cursor
    summarise the fully committed bar
    append summary/target-channel records to history

coalesce consecutive KEEP runs into placements
export typed arrangement JSON and per-channel MIDI/audio outputs
```

Run `.eval()` and inference mode. Generation never needs `Trainer` or reconstructs a training dataset. Seed both channel order and stochastic action sampling. Keep deterministic argmax as the first policy.

Do not resample another channel's completed decision just because a later channel finds it incompatible. Global search/refinement is a future policy. Record channel order to make sequential decisions reviewable.

### 14.2 Candidate construction

Each channel supplies its legal source starts plus SILENCE if useful. Include role/meter eligibility, complete-bar masks and optional source-start restrictions. Mask the natural next position if it is action-equivalent to KEEP. Do not mask an audible rest just because it has no onsets; sustain-only and intentionally empty source material have different metadata.

For very long clips, configurable candidate restriction can later retain phrase/bar boundaries or retrieved top candidates. Start exhaustive over a capped supported clip length, with an explicit input limit. Never drop the correct training destination while subsampling negatives.

Clip end is represented by a cursor equal to playable length, making KEEP invalid. There is no automatic modulo loop; restarting source bar 0 is a scored destination. Silence can be selected at every eligible cut boundary.

### 14.3 Decode policy

`DecodePolicy` consumes log probabilities and legal actions. Default argmax. Optional seeded sampling uses the joint action distribution with explicitly documented temperature semantics. Applying one temperature to full joint action logits differs from separately changing gate/candidate temperatures; record the chosen method.

No post-hoc 'always drums/bass' rules in the default policy. User-fixed selections and explicit constraints are inputs. Early all-silent collapse is a training/calibration issue to diagnose, not hide with an undocumented fallback.

### 14.4 Output and coalescing

The JSON trace stores every bar action, source choice, legal candidates or their IDs, probability and provenance. A compact placement export combines adjacent compatible KEEP bars but preserves START boundaries. The source span and destination span remain explicit, so repeated internal sections are renderable.

MIDI output contains separate channel tracks with re-based note/controller times and the destination tempo/meter map. Audio output contains separate rendered channels and an optional mix. Preserving modality in output does not require rendering MIDI for action scoring.

Rendering has a pluggable instrument backend when MIDI audio is requested. Baseline MIDI export works without commercial plugins. Heavy FL Studio/VST automation is separate from the arrangement model and should not block action-model development.

## 15. Services, signatures and ownership

Selected public APIs:

```python
class PlaybackStateMachine:
    def keep_is_valid(
        self, state: ChannelPlaybackState, clip: Clip, target: BarSpan
    ) -> bool: ...

    def apply(
        self,
        state: ChannelPlaybackState,
        action: PlaybackAction,
        clip: Clip,
        target: BarSpan,
    ) -> tuple[ChannelPlaybackState, BarPlayback]: ...

class CandidateBuilder:
    def build(
        self, channel: Channel, state: ChannelPlaybackState, target: BarSpan
    ) -> CandidateSet: ...

class DecodePolicy:
    def choose(
        self,
        scores: SingleDecisionScores,
        candidates: CandidateSet,
        keep_valid: bool,
        generator: torch.Generator | None = None,
    ) -> PlaybackAction: ...

class ArrangementGenerator:
    def generate(self, request: ArrangementRequest) -> ArrangementResult: ...

class Trainer:
    def fit(
        self,
        train_batches: Iterable[DecisionBatch],
        validation_batches: Iterable[DecisionBatch],
    ) -> TrainingResult: ...

class ArrangementLoss(nn.Module):
    # Input: DecisionLogits and masked DecisionTargets for B decisions.
    # Output: scalar differentiable loss and typed component sums/counts.
    def forward(
        self, logits: DecisionLogits, targets: DecisionTargets, masks: ActionMasks
    ) -> LossBreakdown: ...

class FeatureStore:
    def load(self, key: FeatureKey) -> ClipFeatures: ...
    def save(self, key: FeatureKey, features: ClipFeatures) -> None: ...

class CheckpointManager:
    def save(self, path: Path, state: TrainingCheckpoint) -> None: ...
    def load(self, path: Path, device: torch.device) -> TrainingCheckpoint: ...
```

The Dataset/DataLoader generic is the dataset sample type. A custom collator returns `DecisionBatch`, which may not be expressible by PyTorch's installed stubs. Centralise that typing mismatch in the loader factory/iteration adapter and expose `Iterator[DecisionBatch]` to the trainer if needed; do not scatter untyped casts throughout training.

`SingleDecisionScores` contains one scalar change logit and J destination scores, obtained by explicitly selecting one row from `DecisionLogits`. This keeps scalar playback states out of batched tensor APIs and prevents ambiguous `choose` calls.

CandidateBuilder and the state machine are stateless services; no database/model inheritance. Generator owns mutable playback state. Trainer owns optimiser/scheduler and evaluation scheduling. FeatureStore owns feature persistence, not neural-network construction. CheckpointManager stores model/config state, not live extractor objects or datasets.

## 16. Checkpoints, configuration and experiment reproducibility

Use typed dataclass configuration with separate data, feature, model, objective, training and decoding sections. Validate unknown keys and incompatible combinations. Config parsing happens once at the entry point; no import-time model construction or global mutable training settings.

Save:

- Schema/architecture version and plain serialisable configuration.
- Trainable model state dict; frozen extractor IDs/revisions and feature schema.
- Optimiser/scheduler/scaler state, epoch/global step and best metric.
- Python/NumPy/PyTorch and CUDA RNG state when resuming.
- Dataset/split/label-policy digest, candidate policy and source-crop/render semantics.
- Role/modality mappings, PPQ/meter assumptions and decode/calibration settings.

Resume and inference loading are separate factory functions. A checkpoint stores no auto-selected device assumption; the caller selects CPU/CUDA explicitly. Validate feature-width/preprocessing mismatches instead of silently accepting a different encoder layer/cache.

Write checkpoints atomically and support last/best names. Old 25-dimensional VQ/whole-bundle checkpoints fail with a clear incompatible-version message; there is no meaningful automatic migration to new action labels.

Record source revisions and environment locks for extraction and arrangement training. Keep legacy fairseq extraction isolated if its dependencies conflict with the main runtime. This is preferable to copying a large research repository into `src` with hidden version assumptions.

## 17. Evaluation and meaningful tests

### 17.1 Baselines

- Always KEEP until source end, then restart at 0.
- Fixed loop schedules with supplied clip lengths.
- Independent channel scorer without other-channel context.
- Shared scorer without historical context.
- Whole-clip mean versus temporal source-bar features.

Report legal-action rate and label quality before musical comparisons. Fixed loops can be musically competent, so a complex model must show useful editing beyond mechanics.

### 17.2 Metrics

- Gate log loss, precision/recall for changes and calibration.
- Joint action NLL/accuracy and acceptable-set destination accuracy.
- Presence F1 only where labels genuinely support presence.
- Entrance/exit/repeat source accuracy, crop-run length distribution and unnecessary-cut rate.
- Per-modality/provenance, unseen-composition and mixed-input slices.
- Rollout silence ratio, source utilisation, repetition/continuity and constraint violations.
- Listening assessment of fit between channels, groove, crop boundaries, development and recurring ideas.

Structural similarity alone can reward repetitive/static output. Repetition metrics and activity labels do not substitute for listening evaluation. Avoid claims of better arrangement quality based solely on synthetic exact-action accuracy.

### 17.3 Essential tests

1. State transitions: silent KEEP, entrance, continuation, stop, restart, source jump and source end.
2. Variable lengths: 1/2/8-bar channels in one request; masks and cursors remain correct.
3. Timing: piecewise tempo, 3/4 versus 6/8 grouping, bar boundaries and unsupported partial bars.
4. Sustain/crop semantics: a note begins before a crop, ends inside it, crosses a KEEP boundary or is cut on restart.
5. Probability algebra: combined distribution sums to one; forced actions and all-invalid destinations never yield NaNs.
6. Loss masks: destination loss only applies to labelled changes; all-zero label masks produce finite graph-connected zero.
7. Causality: changing future committed targets leaves current logits unchanged. Test the actual model path, not only a mask helper.
8. Hidden-label isolation: changing the queried action or an undecided channel's target does not alter input tensors.
9. Candidate permutation: logits reorder consistently with candidate inputs; padded candidates are never selected.
10. Channel set permutation: context encoder output is invariant/equivariant under its declared set interface; decode-order effects are evaluated separately.
11. Padding/nulls: B = 1, no other channels, empty history, silent/sustain-only bars and finite masks.
12. Training/inference parity: replay a labelled schedule through the generator's state logic and recover the same context/actions.
13. Checkpoint round trip: identical evaluation logits, resume state and clear version failures.
14. Feature provenance: changed encoder/crop/grid metadata invalidates cache keys.
15. Loader split audits, malformed source reporting and Windows worker smoke test.
16. Small end-to-end fixture: prepare -> cached features -> one update -> save/load -> generated decisions -> MIDI export/audio boundary rendering.

Use deterministic minimal MIDI/audio fixtures for mechanics. Tests may inject small fixed extractor fixtures to avoid heavyweight downloads, but real extractor smoke checks remain a separate required integration check. Do not call fake/random features pretrained musical extraction.

## 18. Code standard

The implementation should be compact, typed and inspectable enough to support research experiments without becoming a collection of loosely coupled scripts.

- No file-header docstrings, heading docstrings, section banners or decorative separators in Python files.
- If a docstring helps, use one succinct line describing the contract or purpose. Do not reproduce the type signature in prose.
- Place concise input/output comments beside each model's forward method. Include tensor dtype, dimensions, mask meaning and relevant state semantics.
- Fully annotate public and private functions, parameters, constructors, dataclass fields and return values. Use built-in generics, unions, `Sequence`/`Mapping` for read-only interfaces and concrete lists/dicts where mutation is intended.
- Use `NDArray[np.float32]` for audio/features, explicit Tensor shape comments and runtime validation at boundaries. Avoid broad `Any`, bare `dict`/`tuple`, type-ignore proliferation and invalid shape annotations pretending tensors are statically enforced.
- If framework return types require a cast, contain it in the adapter with a short reason and validate the runtime interface.
- Follow one pinned Python version; choose it after the extractor smoke checks. Do not force old fairseq dependencies into the main environment just to keep one Python version everywhere.
- Prefer composition. Protocols cover meaningful interchangeable readers/extractors/renderers; `nn.Module` and Dataset are the principal concrete base classes. No inheritance hierarchy solely to share a few helper lines.
- Keep pure timing/state/candidate logic independent of GPU, training and storage.
- Do not add empty placeholder implementations, dead compatibility code, mutable import-time configs or global model/device selection.
- Use logging for progress and typed failure reports for invalid data. Catch expected parsing/I/O exceptions at source boundaries; do not suppress programming errors with broad catch-and-continue blocks.
- Use descriptive musical names: source bar, destination bar, onset, offset, continuation and restart. Avoid calling the gate presence when it predicts change.
- Keep comments for reasoning, masking and musical edge cases. Move architecture exposition to this plan and runnable workflow docs.

Style example:

```python
class DestinationScorer(nn.Module):
    """Score legal source destinations against the channel context."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(model_dim * 4, 1)

    # Input: query float[B, D], candidates float[B, J, D].
    # Output: raw logits float[B, J]; legality is applied by the loss/policy.
    def forward(self, query: Tensor, candidates: Tensor) -> Tensor:
        context = query.unsqueeze(1).expand(-1, candidates.size(1), -1)
        features = torch.cat(
            (context, candidates, context * candidates, context - candidates), dim=-1
        )
        return self.score(features).squeeze(-1)
```

This is an illustrative signature/style, not a commitment that a linear scorer is sufficiently expressive. `expand` shares query storage; concatenation allocates the combined features. Do not mutate expanded views in place.

For tiny heads, an `nn.Linear`/`nn.Sequential` member can replace a named wrapper class. The semantic components in the diagram need not each become a separate file or an inheritance layer.

## 19. Proposed package layout and dependency direction

Keep `src` during the refactor to avoid unnecessary packaging churn. Group classes by responsibility; create a new file when responsibilities or dependencies justify it, not per class.

```text
src/
  contracts.py                 # domain records, enums and identity types
  config.py                    # validated typed experiment configuration
  timing.py                    # grids, tempo maps, ticks and sample conversion
  data/
    sources.py                 # source protocol and local adapters
    midi.py                    # parser/normalisation bridge
    preparation.py             # episode labels, matching and provenance
    dataset.py                 # decision dataset and window/order indexing
    batch.py                   # tensor records, collation and device transfer
  features/
    interfaces.py              # feature records/spec/protocol
    musicbert.py               # native MIDI extraction adapter
    mert.py                    # frozen audio extraction adapter
    cache.py                   # keys, feature persistence and preparation
  models/
    encoders.py                # projection, temporal/bar/current/history encoders
    arrangement.py             # model orchestration, query, gate and scorer
    losses.py                  # masked objectives and probability algebra
  training/
    trainer.py                 # one optimisation/evaluation loop
    evaluation.py              # teacher forcing, rollouts and baselines
    checkpoints.py             # state serialisation and model factories
  inference/
    state.py                   # state transitions and placement coalescing
    candidates.py              # candidate/mask/context construction
    generator.py               # rollout and joint-action decode policy
  rendering/
    midi.py                    # interval crops, controller handling and export
    audio.py                   # continuous source reads, boundary fades and mix
  cli/
    prepare.py
    extract.py
    train.py
    generate.py
    evaluate.py
tests/
configs/
```

Dependency direction:

```text
contracts / timing / config
           ↓
data / features / inference state primitives
           ↓
model input construction / neural models / objectives
           ↓
training or generation
           ↓
rendering/export and CLI orchestration
```

Renderer only needs placements/assets/timing, not trainable model objects. State primitives can be imported by data preparation without importing the full generator. Neural model code must not import Trainer. Contracts must not import implementations that use them.

The current action masks/probability algebra should have one owner, shared by loss and decoding. Place these small pure tensor functions in a shared decision module or alongside typed action contracts when the dependency structure is clear; do not duplicate them in trainer/generator.

## 20. Existing code migration map

| Existing path | Refactor treatment |
|---|---|
| `src/song_data.py` | Replace anonymous nested arrays with explicit contracts; retain a temporary conversion only if a working loader needs it. |
| `src/dataloaders/stem_processor.py` | Move validated timing/channel handling into readers/timing; preserve identity and complete-bar metadata. |
| `src/dataloaders/manifest_loader.py` | Replace with typed manifest adapter using real sample rates and safe cache keys. |
| `src/dataloaders/hf_streaming_loader.py` | Defer HF transport until local schemas work; avoid one dynamic audio column per stem as the internal contract. |
| `src/dataloaders/music_dataset.py` | Replace one-shot token iterator with prepared decision indexing/collation. |
| `src/model_preprocessing/feature_vector_segmentation.py` | Remove from active path; replace hand-designed acoustic descriptors with pretrained extraction. |
| `src/model_preprocessing/quantization/` | Retire from active path; later structural VQ is a separately evaluated module, not a dimension change to this model. |
| `src/main_model/transformers_improved.py` | Replace with a properly causal history encoder and legitimate cross-attention to supplied inputs. Do not retain the self-as-unmasked-memory leakage. |
| `src/training_code/arrangement_trainer.py` | Replace orchestrator/config coupling with Trainer and checkpoint factories; preserve the user's current edits before removal. |
| `src/inference/stem_library.py` | Replace bundle/code storage with channel clip libraries and source-span candidates. |
| `src/inference/arrangement_generator.py` | Replace token generation/random retrieval with explicit stateful action decoding. |
| `src/latent_preprocessing/` | Audit imports, remove obsolete manual role/harmony heuristics from active paths and delete only after migration. |
| `src/audio_training_breakdown_automation/audio_breakdown.py` | Retain useful audio utilities only after boundary/tempo/downbeat corrections. |
| `main.py`, `inference.py` | Thin wrappers or replaced documented CLI modules; no synthetic-noise default training. |
| `scripts/create_hf_dataset.py` | Adapt only after new schemas are stable; no external upload is part of this refactor. |
| `scripts/unzip_all.py` | Pre-existing user file; leave unrelated behaviour intact. |
| Existing docs | Update active workflows; mark obsolete architecture documents historical instead of leaving contradictory instructions. |

Known implementation defects addressed as part of this work: stereo axis confusion, discarded file rates, assumed first-sample downbeat, shortest-stem truncation, missing runtime dependencies, empty-data/zero-batch crashes, iterator exhaustion, future leakage, incomplete validation/checkpoint state and inference/training coupling.

No migration should overwrite unrelated local changes. At planning time the worktree has a modified `arrangement_trainer.py` and an untracked `scripts/unzip_all.py`. Recheck before implementation; do not treat this snapshot as permanent.

## 21. Entry points and concrete workflow

Proposed commands, to become real supported entry points during implementation:

```text
python -m src.cli.prepare --config configs/prepare_lakh.yaml
python -m src.cli.extract --config configs/features_musicbert.yaml
python -m src.cli.train --config configs/train_midi.yaml
python -m src.cli.evaluate --config configs/evaluate_midi.yaml --checkpoint <path>
python -m src.cli.generate --checkpoint <path> --request <clip_manifest.json> --output <dir>
```

Equivalent Slakh/MERT/mixed experiment configs reuse the same commands. All roots and checkpoints come from CLI/config, not hardcoded paths. Smoke fixtures are explicitly selected, not substituted for missing training data.

Preparation writes schema/split/label reports before extraction. Extraction validates the prepared feature inventory before training. Training refuses an empty effective dataset or missing required features and prints actionable counts. Generation writes an arrangement JSON even if audio rendering is not requested.

Use a small direct-dependency `pyproject.toml` or equivalent tested requirements with optional groups for native MIDI extraction, audio extraction and rendering. Add formatting/linting, type checking and pytest configuration. Choose pins from an installation/smoke check; don't preserve the current list of incidental transitive dependencies as the design.

## 22. Implementation sequence for the later refactor

These are ordered deliverables inside the intended comprehensive refactor, not requests for repeated approval.

### Milestone A: contracts and playback mechanics

- Implement typed timing/domain/action records and validators.
- Implement shared state transitions/candidates/joint probability functions.
- Build a meaningful MIDI fixture with variable clip lengths, internal crops, sustains and repeats.
- Export its labelled placements correctly without any learned model.

Completion: state/timing/mask/export checks pass and action semantics are unambiguous.

### Milestone B: source data and labels

- Implement local Lakh, Slakh and observed-manifest adapters.
- Implement exact MIDI-derived episode construction and explicit synthetic fixtures.
- Group/split compositions, record exclusions/ambiguity and validate action reachability.
- Implement cached decision indexing, windows/order prefixes and collator.

Completion: a small real MIDI subset yields inspectable episodes with honest label provenance; original song playback is not the only task.

### Milestone C: pretrained extraction

- Prove MusicBERT checkpoint preprocessing/output alignment in a tested environment.
- Prove MERT extraction on a paired audio stem with revision-pinned preprocessing.
- Implement immutable raw-feature cache, bounded extraction and empty/sustain policies.
- Keep the explicitly named rendered-MIDI fallback independent if needed.

Completion: finite deterministic real features, correct timing and valid cache invalidation. A mock extractor does not satisfy this milestone.

### Milestone D: MIDI baseline model and training

- Implement small continuous encoders/history/current attention and the two action heads.
- Implement masked BCE/conditional destination CE, optional partial-label presence likelihood.
- Add trainer, checkpoints, validation and simple baselines.
- Overfit a meaningful fixture and run a small held-out native-MIDI experiment.

Completion: the model learns nontrivial transitions, has no future/target leakage, and generates legal arrangements from clip-only requests.

### Milestone E: audio and mixed inputs

- Train audio projection and mixed-context adaptation using paired examples.
- Validate unseen clips, MIDI/audio/mixed requests and per-modality metrics.
- Export MIDI and prerendered audio placements using the same schedule.
- Document any missing real-audio editorial labels and evaluate available objectives honestly.

Completion: both modalities are exercised by a trained checkpoint, not merely accepted by an untrained branch.

### Milestone F: cleanup and research readiness

- Remove obsolete active pipelines, stale imports and duplicated code.
- Apply code simplification after nontrivial implementation, preserving contracts and behaviour.
- Complete appropriate tests, lint/type checks and runnable workflow docs.
- Provide migration notes, limitations and experiment manifests.

Completion: the refactored code is reviewable and runnable. Large-corpus training quality still depends on data and compute; do not confuse code readiness with a musically competent final model.

## 23. Deferred research extensions

After the baseline is working and evaluated:

- Multiple source clips per channel, using the same variable-candidate scorer.
- Beat/sub-bar cut decisions and duration-aware partial source bars.
- Explicit tempo adaptation/transposition with corresponding feature recomputation.
- Phrase/section conditioning with verified labels or user plans.
- Discrete bar/phrase summaries using a tested optional VQ bottleneck.
- Learned stopping/refinement/search for channel-order dependence.
- Native MIDI/audio paired fusion rather than modality selection.
- Long-range motif retrieval/memory beyond the fixed history window.
- Role inference when metadata is unavailable.
- Improved observed arrangement datasets and evaluation against explicit sample-planning systems.

Do not implement these as empty frameworks during the first refactor. Stable contracts and the shared candidate scorer already leave extension points.

## 24. Remaining decisions and defaults

| Decision | Proposed first default | What changes it |
|---|---|---|
| Clip ownership | One clip per channel | A concrete need for alternative clips in one channel |
| Decision interval | Complete destination bars | Sub-bar placement requirements with labels |
| Silence behaviour | Ends playback run; explicit source choice on entrance | A requested pause/resume musical policy |
| Gate meaning | KEEP versus CHANGE | If the user explicitly prefers independent presence and playback heads |
| Destination space | SILENCE and legal source starts | Multiple clips or transforms |
| MIDI extractor | Frozen MusicBERT, feasibility checked first | Checkpoint/dependency failure; named rendered-MIDI baseline |
| Audio extractor | Frozen MERT-95M | Task-specific validation or deployment constraints |
| Quantization | Disabled | Evidence continuous summaries need a code bottleneck |
| Channel decisions | Sequential, seeded order | Validation favouring a fixed order or joint/refinement policy |
| Feature phases/history | Configurable small P/H, start 16/16 or 16/32 | Measured rhythm fidelity, memory and rollout quality |
| Training labels | Observed when available, exact inferred MIDI plus qualified constructed episodes | Additional DAW placement data |
| Real-audio schedule labels | Partial only unless observed/inferred reliably | Annotated crop/repeat examples |

The next implementation can proceed under these defaults. The model definitions, losses and tests must use the same selected semantics; do not preserve mutually contradictory choices from the brainstorming as simultaneous requirements.

## 25. Research risk register

| Risk | Evidence to gather | Planned response |
|---|---|---|
| No reliable crop/repeat labels in the downloaded corpus | Inspect prepared episodes and provenance; compare exact recurrence with actual project exports | Keep inferred/constructed/observed regimes separate; prioritise better labels over a larger model |
| The arranger copies original source progression | Measure jump/repeat learning and evaluate reindexed short clip libraries | Remove absolute-origin shortcuts and trivial full-song-only episodes |
| All-silent or always-KEEP collapse | Entrance/change recall, joint calibration and rollout inspection | Improve sampling/labels and validate thresholds; keep baselines visible |
| Native MIDI extraction integration fails | One-file checkpoint/preprocessing smoke check | Isolate adapter dependencies or run the explicitly named rendered-MIDI experiment |
| Audio projection never learns the MIDI-trained context space | Paired and mixed-input validation | Adapt with balanced pairs, optional alignment and small-network unfreezing |
| Rhythm/sustains lost in pooled features | Temporal-feature ablations and crop listening tests | Increase phase resolution or add detailed candidate/context attention where justified |
| Sequential ordering produces inconsistent combinations | Repeat rollouts under controlled channel permutations | Measure the effect; later add refinement/joint decoding if it materially helps |
| Boundary rendering makes correct decisions sound wrong | Round-trip source spans and sustain/tail fixtures | Preserve KEEP continuity and explicitly handle actual cuts |
| Apparently strong results come from duplicate compositions | Split audit across Lakh, Slakh and rendered variants | Group parents/near duplicates and re-run clean evaluation |

Resolve these with small inspectable experiments before scaling corpus size or architecture complexity. The intended first contribution is a correct learned playback/placement policy over supplied clips, not a pile of additional encoder stages.

# Handoff: Speech-to-Singing Voice Conversion

## Project Goal

Build a local-first desktop application that converts any song's vocal to sound like a target user's voice, using only a speech recording as reference (no singing required from the user).

**Input**: song file + user speech recording
**Output**: final song with vocals replaced by user's voice timbre, preserving original pitch, timing, lyrics, and expression

## Architecture

```
Song.wav ──> [Demucs Separator] ──> Vocals + Instrumental
                                        │
User Speech ──> [SVC Backend] <─────────┘
                      │
                Converted Vocal ──> [Quality Judge] ──> [Exporter] ──> Final Mix
```

### Key Components

| Module | Path | Description |
|--------|------|-------------|
| Pipeline | `src/voiceconv/core/pipeline.py` | Orchestrates separation → conversion → judge → export |
| Types | `src/voiceconv/core/types.py` | AudioClip, ConversionJob, ConversionResult dataclasses |
| Demucs | `src/voiceconv/separation/demucs_adapter.py` | `htdemucs_ft` model, shifts=3, overlap=0.5 |
| SVC Base | `src/voiceconv/svc/base.py` | Abstract SVCBackend interface |
| Seed-VC | `src/voiceconv/svc/seedvc_svc.py` | 22kHz zero-shot diffusion transformer |
| OpenVoice | `src/voiceconv/svc/openvoice_svc.py` | VITS-based tone color converter |
| WORLD | `src/voiceconv/svc/world_svc.py` | Traditional vocoder (MCEP transfer) |
| Fine-tune | `src/voiceconv/finetune/trainer.py` | Seed-VC checkpoint fine-tuning wrapper |
| GUI | `src/voiceconv/gui/main_window.py` | PySide6 desktop app (Convert + Fine-Tune tabs) |
| Judge | `src/voiceconv/judge/quality.py` | F0 correlation + speaker similarity checks |
| Compliance | `src/voiceconv/compliance/provenance.py` | Consent + provenance tracking |
| A/B Script | `scripts/ab_compare.py` | Zero-shot vs few-shot comparison |

## Benchmark Results (2026-04-13)

Test song: `Solo Author.wav` (149.7s), 60s vocal segment from middle.
Reference voice: `yj_voice.m4a` (17 min speech recording).
Metric: resemblyzer cosine similarity (higher = more like target).

| Backend | Sim → You | Sim → Orig Singer | Delta | Inference (60s) | Training Time |
|---------|:---------:|:-----------------:|:-----:|:---------------:|:-------------:|
| **so-vits-svc 500ep** | **0.8344** | 0.7275 | **+0.107** | 9s | ~22h CPU |
| Applio RVC 100ep | 0.8087 | 0.7979 | +0.011 | 13s | ~15h CPU |
| Seed-VC zero-shot | 0.7918 | 0.7503 | +0.042 | ~2h CPU | None |
| OpenVoice V2 | 0.797 | — | — | ~30s | None |
| WORLD vocoder | 0.687 | — | — | ~5s | None |

**Conclusion**: so-vits-svc with sufficient training epochs is currently the best backend for speaker similarity. Applio RVC needs more epochs (only 100 of 200+ target). Seed-VC is impractical on CPU (~2h for 60s).

## Training Artifacts (NOT in git)

These are large files that need to be recreated on the new machine:

| Artifact | Path | Size | Description |
|----------|------|------|-------------|
| so-vits-svc model | `sovits_workspace/logs/44k/G_500.pth` | 523MB | Best model, 500 epochs |
| so-vits-svc config | `sovits_workspace/logs/44k/config.json` | — | Model config |
| Applio RVC model | External: `Applio/logs/yj_voice/yj_voice_100e_9800s.pth` | 53MB | 100 epochs, needs more |
| Applio RVC index | External: `Applio/logs/yj_voice/yj_voice.index` | 146MB | FAISS speaker index |
| Training data | `training_data/` | 98MB | 117 segments from yj_voice.m4a |
| Seed-VC checkpoints | `checkpoints/` | 4.7GB | Auto-downloaded pretrained models |
| Demucs models | Auto-downloaded | ~320MB | htdemucs_ft ensemble |

## Known Issues

1. **Applio RVC segfault with FAISS index**: On macOS ARM, using `index_rate > 0` with the FAISS index causes SIGSEGV on audio > 30s. Workaround: set `index_rate=0`. Root cause likely faiss + torch memory conflict.

2. **Seed-VC 44kHz model impractical on CPU**: The F0-conditioned singing model (BigVGAN vocoder) takes ~26s per diffusion step. Only viable with GPU.

3. **Applio RVC undertrained**: Only 100 epochs completed (target: 200+). Speaker similarity delta is marginal (+0.011). Needs continued training.

4. **fairseq incompatible with Python 3.11**: Applio uses `transformers.HubertModel` instead (already patched in the Applio install). The project venv does not use fairseq.

5. **numpy version conflict**: `openvoice-cli` installs numpy 2.x, breaking `<2.0` constraint. Pin `numpy==1.26.4` after installing openvoice.

## Setup on New Machine

### 1. Clone and create venv

```bash
git clone <repo-url>
cd million-dollar-voice-convertor
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install numpy==1.26.4  # pin after openvoice installs numpy 2.x
```

### 2. Install external backends

```bash
# so-vits-svc-fork (best performing backend)
pip install so-vits-svc-fork

# Applio RVC (clone separately)
cd /path/to/workspace
git clone https://github.com/IAHispano/Applio.git
cd Applio
pip install -r requirements.txt
# Download pretrained models:
python -c "from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline; prequisites_download_pipeline(True, True, False)"
```

### 3. Prepare training data

Place your 17-min voice recording at `~/Downloads/yj_voice.m4a`, then:

```bash
# Split into segments for so-vits-svc
python -c "
import librosa, soundfile as sf
from pathlib import Path
audio, sr = librosa.load('~/Downloads/yj_voice.m4a', sr=44100, mono=True)
out = Path('training_data')
out.mkdir(exist_ok=True)
seg_len = 10 * sr  # 10s segments
for i in range(0, len(audio) - seg_len, seg_len):
    sf.write(str(out / f'seg_{i//seg_len:04d}.wav'), audio[i:i+seg_len], sr)
"
```

### 4. Train so-vits-svc (recommended)

```bash
# Preprocess
svc pre-resample -i training_data/ -o sovits_workspace/dataset_raw/yj/
svc pre-config -i sovits_workspace/
svc pre-hubert -i sovits_workspace/

# Train (500 epochs took ~22h on M-series CPU)
svc train -m sovits_workspace/ --max-epochs 500
```

### 5. Train Applio RVC

```bash
cd /path/to/Applio
# Use run_benchmark.py or Applio's GUI for:
# 1. Preprocess  2. Extract features  3. Train (200+ epochs)
```

### 6. Run tests

```bash
pytest tests/ -v --ignore=tests/test_gui.py
```

## Pending Work (User's 7-Point Strategy)

The user provided a detailed strategy document that has NOT been executed yet:

1. **Targeted dataset supplementation**: Add coverage for low-register, rap, bilingual segments (not just more duration)
2. **A/B across pretrained bases**: Test SingerPreTrain 32k, TITAN 40k/48k, SnowieV3.1 40k/48k as RVC base models
3. **Phrase-level chunking**: Per-section parameter sweeps (index_rate, protect, f0_method per section type)
4. **Separator A/B**: Compare htdemucs vs htdemucs_ft vs other models
5. **Section-level benchmarks**: Metrics per section type (rap_low_similarity, melodic_similarity, melodic_pitch_corr, artifact_score, longform_stability)
6. **No manual post-processing dependency**: All quality must come from the pipeline
7. **YingMusic-SVC**: Exploratory only (https://github.com/YingMusic-SVC)

## File Inventory

### Source code (committed)
- `src/voiceconv/` - All application code
- `scripts/` - Build and comparison scripts
- `tests/` - Pytest test suite
- `pyproject.toml` - Package config
- `requirements.txt` - Dependencies

### Large artifacts (gitignored, recreate on new machine)
- `sovits_workspace/` - so-vits-svc training workspace (8.1GB)
- `checkpoints/` - Seed-VC pretrained models (4.7GB)
- `training_data/` - Audio segments from yj_voice.m4a (98MB)
- `test_output_solo/` - Benchmark outputs for Solo Author
- `test_output_final/` - Benchmark outputs for Duck The Rope
- `.venv/` - Python virtual environment

### External (not in this repo)
- `~/Downloads/yj_voice.m4a` - 17-min reference voice recording
- `~/Downloads/Solo Author.wav` - Test song
- `/path/to/Applio/` - Applio RVC installation + trained model

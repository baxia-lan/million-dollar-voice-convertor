# VoiceConv

**Zero-shot speech-to-singing voice conversion.** Preserves pitch, timing, lyrics, and expression — only changes vocal timbre.

## What It Does

- **Input**: A song + a speech sample (you don't need to sing)
- **Output**: The same song with your voice timbre
- Separates vocals from instrumentals automatically
- Converts vocal timbre while preserving the original performance
- Mixes back and exports with full provenance tracking

## Quick Start

### Prerequisites

- Python 3.11+
- (Optional) FFmpeg for MP3/M4A support

### Install

```bash
# Clone
git clone https://github.com/baxia-lan/million-dollar-voice-convertor.git
cd million-dollar-voice-convertor

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Run

```bash
# Launch the GUI
python scripts/run.py

# Or via installed entry point
voiceconv
```

### Test

```bash
# Run all tests (skips slow tests by default)
pytest

# Run all tests including slow ones
pytest -m "" --tb=long

# Run with coverage
pytest --cov=voiceconv --cov-report=term-missing
```

### Package

```bash
# Install build dependencies
pip install -e ".[build]"

# Build standalone executable
python scripts/build.py
# Output: dist/VoiceConv/
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for full details.

### Pipeline

```
Song → Separator → Vocals → SVC Backend → Converted Vocal → Mixer → Output
           │                      ↑                            ↑
           └── Instrumental ──────┼────────────────────────────┘
                                  │
Speech Reference ─────────────────┘
```

### Adapter Pattern

Heavy backends are behind abstract interfaces — swap implementations without changing the pipeline:

| Component | Interface | Default Implementation |
|-----------|-----------|----------------------|
| Separator | `SeparatorBackend` | Demucs (htdemucs) |
| Voice Conversion | `SVCBackend` | WORLD vocoder-based |

### Preservation Guarantees

| Property | Preserved? | Source |
|----------|-----------|--------|
| F0 (pitch) | ✓ | Source vocal |
| Duration | ✓ | Source vocal |
| Lyrics | ✓ | Source vocal |
| Expression | ✓ | Source vocal |
| Timbre | Replaced | Reference speech |

## Project Structure

```
million-dollar-voice-convertor/
├── src/voiceconv/
│   ├── app.py                  # Entry point
│   ├── logging_config.py       # Logging
│   ├── core/
│   │   ├── types.py            # Data types
│   │   ├── audio_io.py         # Audio I/O
│   │   ├── preprocessing.py    # Validation, normalization
│   │   └── pipeline.py         # Pipeline orchestrator
│   ├── separation/
│   │   ├── base.py             # Separator ABC
│   │   └── demucs_adapter.py   # Demucs implementation
│   ├── svc/
│   │   ├── base.py             # SVC ABC
│   │   └── world_svc.py        # WORLD-based VC
│   ├── judge/
│   │   └── quality.py          # Quality assessment
│   ├── compliance/
│   │   └── provenance.py       # Provenance tracking
│   ├── export/
│   │   └── exporter.py         # Mixing & export
│   └── gui/
│       ├── main_window.py      # PySide6 main window
│       └── worker.py           # Background worker thread
├── tests/                      # pytest test suite
├── scripts/
│   ├── run.py                  # Launch app
│   └── build.py                # PyInstaller build
├── docs/
│   └── architecture.md         # Architecture documentation
├── pyproject.toml              # Project config
├── requirements.txt            # Pinned dependencies
└── LICENSE                     # MIT
```

## Compliance

Every conversion produces a `provenance_<job_id>.json` containing:
- Input file hashes (SHA-256, non-reversible)
- Backend identifiers
- Preservation guarantees (F0, duration, lyrics, expression)
- User consent acknowledgement
- Quality judgement scores
- Timestamps

## Adding a New Backend

### New Separator

```python
from voiceconv.separation.base import SeparatorBackend

class MySeparator(SeparatorBackend):
    @property
    def name(self) -> str:
        return "my-separator"

    def is_available(self) -> bool:
        # Check if dependencies are installed
        ...

    def separate(self, mix: AudioClip) -> SeparationResult:
        # Implement separation
        ...
```

### New SVC Backend

```python
from voiceconv.svc.base import SVCBackend

class MySVC(SVCBackend):
    @property
    def name(self) -> str:
        return "my-svc"

    def is_available(self) -> bool:
        ...

    def convert(self, source_vocal, reference_speech) -> ConversionResult:
        # MUST preserve: F0, duration, lyrics, expression
        # ONLY change: timbre (speaker identity)
        ...
```

## License

MIT

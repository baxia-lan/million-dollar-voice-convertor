# Architecture

## Overview

VoiceConv is a local-first desktop application for **zero-shot speech-to-singing voice conversion**. It takes a song and a speech reference, then produces a new version of the song where the vocal timbre matches the speech reference while preserving the original vocal performance (pitch, timing, lyrics, expression).

## Pipeline

```
Song ──► Separator ──► Vocals ──► SVC Backend ──► Converted Vocal ──► Mixer ──► Final Output
              │                        ▲                                  ▲
              └─ Instrumental ─────────┼──────────────────────────────────┘
                                       │
Speech Reference ──────────────────────┘
```

### Stages

1. **Load & Validate** — Audio I/O with format detection, resampling to 44.1kHz mono
2. **Separate** — Demucs splits the song into vocals + instrumental
3. **Convert** — SVC backend transforms vocal timbre using speech reference
4. **Judge** — Quality assessment (speaker similarity, F0 correlation, SNR)
5. **Export** — Mix converted vocal with instrumental, save all stems + provenance

## Adapter Architecture

All heavy backends are behind abstract interfaces:

- `SeparatorBackend` — vocal/instrumental separation
- `SVCBackend` — singing voice conversion

This allows swapping implementations without changing the pipeline:

| Slot | Default | Notes |
|------|---------|-------|
| Separator | Demucs (htdemucs) | GPU-accelerated when available |
| SVC | WORLD-based | CPU-only, no model downloads |
| SVC (advanced) | (adapter slot) | For neural backends like FreeVC, kNN-VC |

## WORLD-Based SVC

The default backend uses the WORLD vocoder:

1. Decompose source vocal → F0, spectral envelope (SP), aperiodicity (AP)
2. Decompose reference speech → F0_ref, SP_ref, AP_ref
3. Convert SP to mel-cepstral coefficients (MCEPs)
4. Global variance normalization: shift source MCEP distribution to match target
5. Optional frequency warping for vocal tract length normalization
6. Resynthesize with **original F0** + converted SP + **original AP**

Key guarantees:
- `preserve_f0 = True` — pitch contour comes from source
- `preserve_duration = True` — output length matches source
- `preserve_lyrics = True` — phonetic content from source
- `preserve_expression = True` — vibrato, dynamics, articulation from source

## Compliance

Every conversion produces a `provenance_<job_id>.json` file containing:
- SHA-256 hashes of input files
- Backend identifiers
- Preservation guarantees
- User consent acknowledgement
- Quality judgement scores
- Timestamps

## Module Map

```
src/voiceconv/
├── app.py              # Entry point
├── logging_config.py   # Logging setup
├── core/
│   ├── types.py        # Data types (AudioClip, ConversionJob, etc.)
│   ├── audio_io.py     # Load/save audio with format detection
│   ├── preprocessing.py # Validation, trim silence, normalize
│   └── pipeline.py     # Main orchestrator
├── separation/
│   ├── base.py         # SeparatorBackend ABC
│   └── demucs_adapter.py
├── svc/
│   ├── base.py         # SVCBackend ABC
│   └── world_svc.py    # WORLD vocoder-based VC
├── judge/
│   └── quality.py      # Speaker similarity, F0 correlation, SNR
├── compliance/
│   └── provenance.py   # Consent, provenance records
├── export/
│   └── exporter.py     # Mixing, stem export
└── gui/
    ├── main_window.py  # PySide6 main window
    └── worker.py       # Background QThread for pipeline
```

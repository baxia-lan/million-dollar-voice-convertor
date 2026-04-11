# VoiceReplace Studio / GEMINI.md

## Mission
Ship a local-first desktop application for zero-shot speech-to-singing voice conversion.

## Hard Rules
- Do not turn this into a "user sings first" workflow.
- Do not solve with pitch correction + retiming.
- Preserve source vocal timing, pitch contour, lyrics, and expression.
- Use source vocal as the only performance carrier.
- User speech references are for timbre identity only.
- GUI is required.
- Tests are required.
- Compliance UI and provenance manifest are required.
- Avoid TODOs and stubs on the critical path.

## Default Stack
- Python 3.11
- PySide6
- PyTorch / torchaudio
- librosa / soundfile / scipy
- ffmpeg
- pydantic
- pytest

## Product Behavior
- One-click generate flow
- Inputs: song + speech reference
- Outputs: final_mix.wav, converted_vocal.wav, instrumental.wav, quality_report.json, provenance_manifest.json

## Compliance
- Require explicit consent attestation
- Block non-consensual third-party voice cloning
- Keep processing local by default
- Generate provenance manifest on export

## Delivery Standard
- Working code, not a concept
- Full repo changes
- Start/run/test/build commands documented

# CLAUDE.md

## Project Purpose
Build a local-first desktop product for zero-shot speech-to-singing voice conversion.

## Core Definition
Input: song + arbitrary speech reference audio.
Output: final song where source vocal performance is preserved and only singer timbre is replaced by target user voice identity.

## Hard Constraints
- No user singing required
- No timing alignment from target audio
- No pitch correction as main solution
- No full-mix direct conversion
- GUI required
- Compliance required
- Provenance required
- Tests required

## Engineering Constraints
- Python 3.11
- PySide6
- Adapter architecture
- At least one actually runnable SVC backend
- No critical-path stubs

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-04-13

### Added

- Comprehensive test suite with 187 tests and 95% coverage ([#30])
- Ruff linting and formatting with CI enforcement ([#30])
- Windows CI testing on Python 3.10 ([#30])
- Configurable `tolerance` and `clamp` parameters for ChunkResampler ([#34])
- Nodes table in README documenting all seven nodes ([#33])
- Stem mapping reference and troubleshooting guide in README ([#32])

### Fixed

- CPU device handling no longer silently skips separation ([#31], [#23])
- Float-to-int cast errors in audio processing ([#31], [#16], [#22])
- Corrupted model checkpoint error with actionable fix instructions ([#32], [#21])

### Changed

- Pinned librosa to `>=0.10.2,<1` to prevent breaking changes ([#31], [#20])
- Version bumped from 1.x to 2.0.0 ([#33])

[2.0.0]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/releases/tag/v2.0.0
[#4]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/issues/4
[#9]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/issues/9
[#11]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/issues/11
[#16]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/issues/16
[#20]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/issues/20
[#21]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/issues/21
[#22]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/issues/22
[#23]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/issues/23
[#30]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/pull/30
[#31]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/pull/31
[#32]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/pull/32
[#33]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/pull/33
[#34]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/pull/34
[#35]: https://github.com/christian-byrne/audio-separation-nodes-comfyui/pull/35

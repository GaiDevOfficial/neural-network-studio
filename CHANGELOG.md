# Changelog

All notable changes to Node Studio are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-10-21

### Fixed

- Pinned `.js`, `.mjs`, `.css`, `.json`, `.map`, `.svg` and `.wasm` MIME types in both the desktop host
  and the development server. Windows machines can map `.js` to `application/javascript` while others use
  `text/javascript`; the host now answers consistently and the CI MIME assertion is stable.
- The release helper now updates `package.json` to the tag version before committing.

### Changed

- CI installs its Python dependencies from `requirements-dev.txt`.

## [1.0.0] - 2026-10-21

### Added

- Scratch-style Node Studio canvas with draggable blocks, bezier wires, pan/zoom, undo and live graph
  compilation.
- Absolute Beginner Playground with the real MNIST 0-vs-1 brain, drawing pad, prediction probe and
  bounded online corrections.
- Beginner Assist and Advanced Manual tiers backed by the same graph compiler and worker protocol.
- Smart Assist repairs, ghost suggestions, auto-wiring and one-click presets for MNIST, XOR and Iris.
- Right-click wire menu with animated **Break connection**, plus drag-to-delete and drag-wire quick-add.
- Real training loop with held-out loss, accuracy, confusion matrix, loss sparkline and prediction probe.
- Built-in benchmarks, real MNIST, CSV upload, SQLite, local CSV and REST/JSON data sources.
- Graph JSON, model JSON, Python, JavaScript, NumPy `.npz` and ONNX exports.
- Ten step-by-step lessons, including three action-gated build missions (13, 14 and 18 steps).
- Portable Windows one-folder build, public ZIP and SHA-256 checksum.
- JavaScript, Python, real-WebView2 and packaged-launcher verification suites.

### Changed

- Node Studio is now the only application workspace. The retired classic forms-and-charts page is
  replaced by a minimal redirect shim, and the desktop launcher opens `graph_studio.html` directly.
- The desktop host and development server redirect `/` and `/index.html` to Node Studio.
- Release artifacts are now the portable folder, its ZIP and the checksum sidecar.

### Removed

- Classic visualizer landing page, dashboard controls, decision-boundary panel, prediction sandbox,
  metrics chart and their classic-only UI/styles.
- The one-file executable from the public distribution because of a Defender heuristic false positive.

### Security

- The local desktop host is bound to `127.0.0.1` only and does not upload data.
- The SQLite/CSV/REST Data API is read-only and served from the same local process.

### License

- Source code and documentation are released under **AGPL-3.0-or-later**.
- The Node Studio name and branding are protected by the trademark policy.
- Proprietary and commercial terms are available through a separate commercial license.

[1.0.1]: https://github.com/GaiDevOfficial/neural-network-studio/releases/tag/v1.0.1
[1.0.0]: https://github.com/GaiDevOfficial/neural-network-studio/releases/tag/v1.0.0

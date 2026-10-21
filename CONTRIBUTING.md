# Contributing to Node Studio

Thanks for helping improve Node Studio. The project values real measurements over impressive-looking
placeholders: every graph, metric, prediction and export must run through the production engine.

## Development setup

Requirements:

- Windows 10/11 for the desktop host and WebView2 smokes.
- Node.js 22 or newer.
- Python 3.11 or newer with NumPy.
- Development packages: `pip install -r requirements-dev.txt`.
- Optional for the packaged build: the same file installs PyInstaller and pywebview.

```powershell
npm test
npm run test:python
python tools/smoke_node_studio.py
```

The browser and desktop builds are served over HTTP; do not open `src/graph_studio.html` from `file://`.

## What makes a good change

- Keep the graph compiler, worker RPCs and exported artifacts numerically faithful to the engine.
- Add a test when you change a gate, a wire rule, a metric, a data path or an export format.
- Prefer exact block labels from the registry over invented names.
- Do not add simulated metrics, decorative "AI thinking" states or fake loading progress.
- Keep the classic landing page retired. Node Studio is the application.
- Do not distribute the one-file PyInstaller artifact; use the portable folder and ZIP.

## Pull requests

1. Fork the repository and create a focused branch.
2. Make the change and run the relevant suites.
3. Update `CHANGELOG.md` under `Unreleased` when the change is user-visible.
4. Open a pull request with a clear summary, test evidence and screenshots for UI changes.
5. Keep unrelated formatting changes out of the same pull request.

The CI workflow runs the JavaScript and Python suites on Windows. Real-WebView2 smokes are run locally
because they need a desktop session.

## Release workflow

Releases use annotated tags and the portable build. The repository keeps a `backup` branch at the
previous release and a local `backups/` ZIP of that version.

```powershell
# Full quality gate, build, commit, tag, backup and release:
powershell -ExecutionPolicy Bypass -File tools/push_with_backup.ps1 -Message "Describe the change"

# Explicit version instead of an automatic patch bump:
powershell -ExecutionPolicy Bypass -File tools/push_with_backup.ps1 `
  -Message "Add feature" -Version 1.1.0
```

Before a release:

1. Add release notes to `release-notes/vX.Y.Z.md` (the script falls back to commit history).
2. Rebuild with `python desktop/build_exe.py`.
3. Run `npm test`, `npm run test:python`, `python tools/smoke_node_studio.py` and
   `npm run smoke:packaged`.
4. Confirm the ZIP checksum in `dist/NeuralNetworkStudio-Windows-x64.sha256.txt`.
5. Publish the release with the ZIP and checksum attached.

## Reporting security issues

Please do not open a public issue for a vulnerability. Use GitHub's private security advisory flow
described in [SECURITY.md](SECURITY.md).

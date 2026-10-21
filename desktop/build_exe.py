"""
Automated Build Script for the Windows Desktop App and Web Artifacts
Neural Network Studio Platform

The default Windows build is a portable one-folder app:

    dist/NeuralNetworkStudio/NeuralNetworkStudio.exe

PyInstaller's one-file bootloader is currently quarantined by Windows Defender as
a heuristic false positive (Trojan:Win32/Bearfoos.A!ml) even though the same
executable and dependencies pass the real WebView2 boot smoke in one-folder mode.
The portable folder is therefore the only build intended for distribution.
`--onefile` remains a developer-only option; do not ship that artifact.

Usage:
  python desktop/build_exe.py             # full build: dist/web + portable app folder + public ZIP
  python desktop/build_exe.py --onefile   # developer-only one-file exe (not for distribution)
  python desktop/build_exe.py --web-only  # static web bundle only (GitHub Pages / Netlify / Vercel)
  python desktop/build_exe.py --exe-only  # desktop package only (skips re-copying dist/web)
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys


def build_web(project_root):
    src_dir = os.path.join(project_root, 'src')
    web_dist_dir = os.path.join(project_root, 'dist', 'web')
    print("[1/3] Assembling static web distribution in dist/web/...")
    if os.path.exists(web_dist_dir):
        shutil.rmtree(web_dist_dir)
    shutil.copytree(src_dir, web_dist_dir)
    print(f"  -> Successfully generated static web bundle at: {web_dist_dir}")
    return web_dist_dir


def build_exe(project_root, onefile=False):
    dist_dir = os.path.join(project_root, 'dist')
    if onefile:
        spec_name = 'NeuralNetworkStudio.spec'
        work_dir = os.path.join(dist_dir, 'build')
        label = 'one-file executable'
    else:
        spec_name = 'NeuralNetworkStudioOnedir.spec'
        work_dir = os.path.join(dist_dir, 'build-portable')
        label = 'portable app folder'

    spec_path = os.path.join(project_root, 'desktop', spec_name)
    print(f"\n[2/3] Compiling Windows {label} with PyInstaller...")
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
        "--distpath", dist_dir,
        "--workpath", work_dir,
        spec_path
    ]
    print(f"  Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        print("\n[ERROR] PyInstaller compilation failed!")
        sys.exit(result.returncode)
    return dist_dir


def _folder_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _write_portable_readme(app_dir):
    """Ship a short, honest note with the portable folder."""
    readme = os.path.join(app_dir, 'README.txt')
    text = """NEURAL NETWORK STUDIO - WINDOWS PORTABLE BUILD
================================================

1. Keep this whole folder together.
2. Double-click NeuralNetworkStudio.exe.
3. No installer, Python, Node.js or external runtime is required.

The app serves its own pages from a local-only server and stores data on this
machine. Nothing is uploaded by the desktop host.

WHY A FOLDER INSTEAD OF ONE EXE?
PyInstaller's self-extracting one-file launcher triggers a Windows Defender
machine-learning false positive (Trojan:Win32/Bearfoos.A!ml). This folder build
is the same application without that one-file wrapper, and it does not trigger
the same detection. Distribute this folder, or a ZIP of it, rather than the old
single-file build.

IF WINDOWS SMARTSCREEN WARNS
This build is not code-signed with a commercial certificate, so Windows may show
"Windows protected your PC" the first time. That is a reputation warning, not a
virus detection. Only run a copy you received from the publisher, and verify the
SHA-256 checksum supplied with the ZIP.

For a fully trusted, warning-free public release, sign NeuralNetworkStudio.exe
with a trusted code-signing certificate. That is the only way to remove the
unsigned-app reputation warning on other people's machines.
"""
    with open(readme, 'w', encoding='utf-8') as handle:
        handle.write(text)
    return readme


def _package_public_zip(dist_dir, app_dir):
    """Create the distributable ZIP from the portable folder and publish its SHA-256."""
    base = os.path.join(dist_dir, 'NeuralNetworkStudio-Windows-x64')
    archive = base + '.zip'
    checksum = base + '.sha256.txt'
    if os.path.exists(archive):
        os.remove(archive)
    shutil.make_archive(base, 'zip', root_dir=dist_dir, base_dir=os.path.basename(app_dir))
    digest = hashlib.sha256()
    with open(archive, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    with open(checksum, 'w', encoding='ascii') as handle:
        handle.write(f"{digest.hexdigest().upper()}  {os.path.basename(archive)}\n")
    return archive, digest.hexdigest().upper(), os.path.getsize(archive) / (1024 * 1024)


def verify(project_root, built_web, onefile=False):
    dist_dir = os.path.join(project_root, 'dist')
    web_dist_dir = os.path.join(dist_dir, 'web')
    if onefile:
        exe_path = os.path.join(dist_dir, "NeuralNetworkStudio.exe")
        app_dir = None
    else:
        app_dir = os.path.join(dist_dir, "NeuralNetworkStudio")
        exe_path = os.path.join(app_dir, "NeuralNetworkStudio.exe")
    print("\n[3/3] Verifying build outputs...")

    if built_web:
        node_studio = os.path.join(web_dist_dir, 'graph_studio.html')
        if os.path.exists(node_studio):
            print(f"  -> Node Studio page present in web bundle: {node_studio}")
        else:
            print("  [WARNING] graph_studio.html not found in dist/web — Node Studio missing from bundle!")

    if os.path.exists(exe_path):
        if app_dir:
            readme_path = _write_portable_readme(app_dir)
            zip_path, zip_sha, zip_mb = _package_public_zip(dist_dir, app_dir)
            size_mb = _folder_size(app_dir) / (1024 * 1024)
            print("\n==============================================================")
            print("   BUILD COMPLETE: PORTABLE DESKTOP APP GENERATED!           ")
            print("==============================================================")
            print(f"  Launcher    : {exe_path}")
            print(f"  App Folder  : {app_dir}  ({size_mb:.2f} MB)")
            print(f"  Shipping Note: {readme_path}")
            print(f"  Public ZIP  : {zip_path}  ({zip_mb:.2f} MB)")
            print(f"  SHA-256     : {zip_sha}")
            print(f"  Web Bundle  : {web_dist_dir}")
            print("  Run         : double-click NeuralNetworkStudio.exe in the app folder")
            print("==============================================================\n")
        else:
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print("\n==============================================================")
            print("   BUILD COMPLETE: STANDALONE EXECUTABLE GENERATED!          ")
            print("==============================================================")
            print(f"  Target File : {exe_path}")
            print(f"  Binary Size : {size_mb:.2f} MB")
            print(f"  Web Bundle  : {web_dist_dir}")
            print("==============================================================\n")
    else:
        print(f"\n[WARNING] Executable not found at expected location: {exe_path}")
        if not onefile:
            print("          Check the PyInstaller output above for the failure.")
        else:
            print("          Windows Defender may have quarantined the one-file build.")
            print("          Re-run without --onefile to produce the portable folder.")


def main():
    parser = argparse.ArgumentParser(description="Neural Network Studio build pipeline")
    parser.add_argument('--web-only', action='store_true', help='build the static web bundle only')
    parser.add_argument('--exe-only', action='store_true', help='build the desktop app only (skip dist/web copy)')
    parser.add_argument('--onefile', action='store_true',
                        help='developer-only single-file exe (Defender false positive; do not distribute)')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print("==============================================================")
    print("   NEURAL NETWORK STUDIO - AUTOMATED BUILD PIPELINE           ")
    print("==============================================================")
    print(f"Project Root : {project_root}")
    print(f"Dist Dir     : {os.path.join(project_root, 'dist')}\n")

    built_web = False
    if not args.exe_only:
        build_web(project_root)
        built_web = True
    if not args.web_only:
        build_exe(project_root, onefile=args.onefile)
    verify(project_root, built_web, onefile=args.onefile)


if __name__ == '__main__':
    main()

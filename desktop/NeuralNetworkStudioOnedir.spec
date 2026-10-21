# -*- mode: python ; coding: utf-8 -*-
# Portable one-folder desktop build (default).
# The one-file bootloader is currently flagged by a Windows Defender heuristic
# while this exact app and its dependencies pass the real WebView2 boot smoke.

import os

block_cipher = None
project_root = os.path.abspath(os.path.join(os.path.dirname(SPEC), '..'))
src_dir = os.path.join(project_root, 'src')
icon_file = os.path.join(src_dir, 'assets', 'icon.ico')

a = Analysis(
    ['desktop_app.py'],
    pathex=[project_root, os.path.join(project_root, 'desktop')],
    binaries=[],
    datas=[(src_dir, 'src')],
    hiddenimports=['webview', 'clr_loader', 'pythonnet', 'cffi', 'http.server', 'socket', 'threading', 'studio_data_api'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NeuralNetworkStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file if os.path.exists(icon_file) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='NeuralNetworkStudio',
)

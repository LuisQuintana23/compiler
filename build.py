import os
import sys

def build():
    src_path = os.path.abspath("src")
    sys.path.insert(0, src_path)
    
    os.system("poetry install")
    
    # create spec file
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/compiler/cli/cli.py'],
    pathex=['src'],
    binaries=[],
    datas=[('src/compiler', 'compiler')],
    hiddenimports=['compiler.cli.cli'],
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='unam.fi.compilers.g5.06',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    with open("compiler.spec", "w") as f:
        f.write(spec_content)
    
    os.system("pyinstaller --clean compiler.spec")

if __name__ == "__main__":
    build()

# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Recolecta datos/binaries/imports de librerías con assets estáticos
datas = []
binaries = []
hiddenimports = [
    "flask_caching",
    "dash",
    "dash_bootstrap_components",
    "plotly",
    "snowflake.connector"
]

for mod in (
    "dash",
    "dash_bootstrap_components",
    "plotly",
    # Si usas Snowflake en tu app, estas líneas ayudan al análisis:
    "snowflake",
    "snowflake.connector",
    # Certificados (por si alguna dependencia los necesita):
    "certifi",
):
    try:
        d, b, h = collect_all(mod)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        # No fallar si alguna no está instalada en tu entorno actual
        pass

a = Analysis(
    ['Analyzer_2.py'],
    pathex=[os.path.abspath('.')],   # ruta base
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ONEFILE (empaquetado en un único .exe)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Analyzer',                 # Nombre del ejecutable final
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,                   # Sin consola (windowed)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,                       # Pon aquí un .ico si quieres
)

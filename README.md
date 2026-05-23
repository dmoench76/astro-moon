# astro-moon

Python-Pipeline zur Verarbeitung von Mondaufnahmen (ZWO ASI662MC, SharpCap) in MP4-Videos.

## Skripte

| Skript | Beschreibung |
|--------|-------------|
| `moon-in-the-river.py` | Original FIT→MP4, Referenz |
| `ser2mp4.py` | v1: Basis SER→MP4, VAAPI |
| `ser2mp4v2.py` | v2: Globaler Stretch, 48 fps |
| `ser2mp4v3.py` | v3: Sky-Mask, CLAHE, USM, Fade-in/out |
| `ser2mp4v4.py` | v4: Timestamp-basiertes Frame-Mapping |
| `ser2mp4v5.py` | v5: Phase-Correlation-Interpolation, Color-Grading **(aktuell)** |

## Verwendung

```bash
python3 ser2mp4v5.py <datei.ser> <ausgabe.mp4>
```

## Kamera

ZWO ASI662MC, SharpCap, SER RAW16 RGGB (1920×1080)

## Hinweise

- SER-Dateien von SharpCap sind immer little-endian, unabhängig vom `LittleEndian`-Feld im Header.
- VAAPI (h264_vaapi) wird automatisch verwendet wenn `/dev/dri/renderD128` vorhanden ist, sonst libx264.

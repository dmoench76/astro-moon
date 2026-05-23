# astro-moon

Python-Pipeline zur Verarbeitung planetarer Mondaufnahmen in abspielbare MP4-Videos.

## Was macht das Skript?

Rohdaten einer Planetenkamera (SER RAW16, Bayer-Muster RGGB) werden in ein fertig prozessiertes,
farbnormiertes MP4-Video umgewandelt. Die Pipeline adressiert dabei typische Probleme
bei Lucky-Imaging-Aufnahmen des Mondes:

- **Debayering** – Bayer-Rohdaten (RGGB) werden in Farb-BGR-Frames umgerechnet.
- **Globale Streckung** – Helligkeit und Kontrast werden einmalig über Stichprobenframes
  kalibriert, nicht per Frame. Verhindert das typische Flimmern durch Auto-Exposure-Schwankungen.
- **Sky-Maske** – Der Himmelshintergrund wird anhand der Rohdaten maskiert und auf Schwarz
  gesetzt. Kein helles Grau oder Rauschen um den Mond herum.
- **CLAHE** – Adaptiver lokaler Kontrast (auf dem LAB-L-Kanal) hebt Krater und
  Oberflächendetails hervor, ohne helle Bereiche zu übersteuern.
- **Unsharp Masking** – Feines Schärfen (σ=1.5) für knackscharfe Kraterkanten.
- **Timestamp-Mapping** – SER-Dateien enthalten UTC-Timestamps pro Frame. Das Skript
  mappt Output-Frames gleichmäßig auf die echte Aufnahmedauer, nicht auf Frame-Nummern.
  Damit wird der Beschleunigungseffekt durch variable Auto-Exposure-Intervalle eliminiert.
- **Phase-Correlation-Interpolation** – Zwischen zwei Quell-Frames wird der
  Translationsvektor per FFT berechnet. Beide Frames werden auf die Zwischenposition
  gewarpt und geblendet. Das ergibt eine butterweiche, kinematische Mondbewegung.
- **Color-Grading** – Drei wählbare Farbpaletten (siehe unten).
- **Fade-in / Fade-out** – Je 2 Sekunden am Anfang und Ende.
- **Hardware-Encoding** – H.264 per VAAPI (GPU), Fallback auf libx264 (CPU).

## Eingabe

Das Skript erwartet **SER-Dateien** im Format RAW16 RGGB, wie sie SharpCap mit einer
ZWO-Farbkamera erzeugt.

**SER-Format-Besonderheit (SharpCap-Bug):**  
SharpCap schreibt SER-Dateien immer little-endian, setzt aber `LittleEndian=0` im Header.
Das Skript ignoriert dieses Flag und liest die Daten immer als `<u2` (little-endian 16-bit).

Getestet mit:
- Kamera: **ZWO ASI662MC** (1920×1080, 16-bit, Farbsensor RGGB)
- Software: **SharpCap** (SER-Aufnahme, RAW16-Modus)
- Bayer-Code im SER-Header: ColorID=8 → `cv2.COLOR_BayerRG2BGR`

Andere ZWO-Farbkameras mit RGGB-Sensor sollten ebenfalls funktionieren.
Für andere Bayer-Muster (GRBG, GBRG, BGGR) sind die entsprechenden ColorIDs
9, 10, 11 im Header bereits unterstützt.

## Verwendung

```bash
# Vollständiges Video (Standard: 48 fps, Farbpalette gold)
python3 ser2mp4v5.py aufnahme.ser ausgabe.mp4

# Optionen
python3 ser2mp4v5.py aufnahme.ser ausgabe.mp4 --fps_out=60 --color=marsian

# Vorschau-Clip (3 Sekunden aus der Mitte, kein Rendering des gesamten Materials)
python3 ser2mp4v5.py aufnahme.ser ausgabe.mp4 --sample_time=3s --sample_from=middle
```

### Optionen

| Option | Werte | Standard | Beschreibung |
|--------|-------|---------|-------------|
| `--fps_out` | Ganzzahl | `48` | Ausgabe-Framerate |
| `--color` | `pale`, `gold`, `marsian` | `gold` | Farbpalette |
| `--sample_time` | z.B. `3s`, `500ms` | – | Vorschau-Dauer; schreibt `*_sample.mp4` |
| `--sample_from` | `start`, `middle`, `end` | `middle` | Position des Vorschau-Clips |

### Farbpaletten

| Palette | B× | G× | R× | Wirkung |
|---------|-----|-----|-----|---------|
| `pale` | 0.92 | 1.02 | 1.02 | Fast neutral, minimaler warmer Hauch |
| `gold` | 0.65 | 1.10 | 1.08 | Gelb-golden, subtil — entspricht dem natürlichen Mondlicht |
| `marsian` | 0.42 | 0.98 | 1.22 | Orange-rot, deutlich — für dramatischen Look |

Der Schlüssel zu Gelbton statt Orange: **G anheben, nicht R**. R hoch → orange/Mars,
G hoch + B runter → gelb/gold.

## Systemvoraussetzungen

### Python-Pakete

```bash
sudo apt install python3-numpy python3-opencv
```

Kein pip, kein venv nötig — beide Pakete sind als APT-Pakete verfügbar.

### ffmpeg

```bash
sudo apt install ffmpeg
```

### GPU-Encoding (VAAPI)

Das Skript erkennt automatisch, ob VAAPI verfügbar ist, und wählt den passenden Encoder:

| Situation | Encoder | Qualitätssetting |
|-----------|---------|-----------------|
| `/dev/dri/renderD128` vorhanden | `h264_vaapi` (GPU) | `-qp 20` |
| Kein VAAPI | `libx264` (CPU) | `-crf 20 -preset fast` |

**VAAPI** läuft auf Systemen mit Intel- oder AMD-Grafikeinheit unter Linux
(i915, amdgpu, Mesa). Das Gerät `/dev/dri/renderD128` ist der Standard-Render-Node;
abweichende Pfade können direkt im Skript unter `VAAPI_DEV` angepasst werden.

Auf reinen CPU-Systemen (kein DRI-Node) wird automatisch libx264 verwendet —
ohne Konfigurationsänderung.

## Skript-Versionshistorie

| Version | Neuerung |
|---------|----------|
| v1 | Basis SER→MP4, little-endian-Fix, VAAPI/libx264-Fallback |
| v2 | Globaler Stretch (kein Flicker), 48 fps |
| v3 | Sky-Maske (echter schwarzer Hintergrund), CLAHE, feines USM (σ=1.5), Fade-in/out |
| v4 | Timestamp-basiertes Frame-Mapping (gleichmäßige Mondbewegung) |
| v5 | Phase-Correlation-Interpolation, Color-Grading-Presets, CLI-Optionen |

`moon-in-the-river.py` ist das ursprüngliche Referenzskript für FITS-Dateien (astropy).

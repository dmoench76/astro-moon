# astro-moon

Python-Pipeline zur Verarbeitung planetarer Aufnahmen (Mond, Jupiter, ...) in abspielbare MP4-Videos.
Unterstützt **SER RAW16** und **AVI RAW8** — Formaterkennung erfolgt automatisch anhand der Dateiendung.

## Was macht das Skript?

Rohdaten einer Planetenkamera (Bayer-Muster RGGB) werden in ein fertig prozessiertes,
farbnormiertes MP4-Video umgewandelt. Die Pipeline adressiert dabei typische Probleme
bei Lucky-Imaging-Aufnahmen:

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

Das Skript erkennt das Format automatisch an der Dateiendung und wählt den passenden Reader.

### SER RAW16 (`.ser`)

SharpCap speichert 16-Bit-Bayer-Rohdaten im SER-Format (LUCAM-RECORDER).
Jeder Frame wird direkt per Byte-Offset adressiert; Timestamps im Trailer
ermöglichen präzises Timestamp-basiertes Frame-Mapping.

**SharpCap-Besonderheit:** SER-Dateien sind immer little-endian, unabhängig vom
`LittleEndian`-Flag im Header. Das Skript ignoriert dieses Flag und liest
grundsätzlich als `<u2`.

Das Bayer-Muster wird aus dem ColorID-Feld im SER-Header bestimmt
(ColorID 8–11 → RGGB, GRBG, GBRG, BGGR).

### AVI RAW8 (`.avi`)

SharpCap speichert 8-Bit-Bayer-Rohdaten als `pal8` (Palette-indexed) in AVI.
Die Palette ist ein lineares Grau-Ramp; ein dekodierter Kanal ergibt den
originalen Bayer-Wert. Frames werden per `cv2.VideoCapture` gelesen.

Das Bayer-Muster und weitere Metadaten werden aus der SharpCap-Begleitdatei
(`<datei>.avi.txt`) ausgelesen falls vorhanden; Fallback: RGGB.

AVI-Dateien haben keine per-Frame-Timestamps; die Framerate wird
als konstant angenommen (lineare Reihenfolge, keine Interpolation).

### Getestet mit

- Kamera: **ZWO ASI662MC** (1920×1080, Farbsensor RGGB)
- Software: **SharpCap** (SER RAW16 und AVI RAW8)
- Objekte: Mond (SER), Jupiter (AVI)

## Verwendung

```bash
# SER RAW16 (Mond)
python3 ser2mp4.py aufnahme.ser ausgabe.mp4

# AVI RAW8 (Jupiter, Mars, ...)
python3 ser2mp4.py aufnahme.avi ausgabe.mp4 --color=pale

# Optionen
python3 ser2mp4.py aufnahme.ser ausgabe.mp4 --fps_out=60 --color=marsian

# Vorschau-Clip (3 Sekunden aus der Mitte, kein Rendering des gesamten Materials)
python3 ser2mp4.py aufnahme.ser ausgabe.mp4 --sample_time=3s --sample_from=middle
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


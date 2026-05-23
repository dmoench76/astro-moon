#!/usr/bin/env python3
"""
stack_planet.py – Planetarisches Lucky-Imaging-Stacking

1. Qualitäts-Ranking aller Frames via Laplacian-Varianz (Schärfemaß)
2. Top N% auswählen
3. Auf besten Frame alignen (Phase Correlation, sub-pixel)
4. Qualitätsgewichtetes Stacking
5. 16-bit PNG oder TIFF ausgeben

Usage:
  stack_planet.py input.avi output.png [--top=10] [--crop=300]
  stack_planet.py input.ser output.tif [--top=5] [--crop=400]
"""

import sys
import os
import struct
import argparse
import numpy as np
import cv2

WB_R = 1.05
WB_B = 0.88

BAYER_CODES = {
    'RGGB': cv2.COLOR_BayerRG2BGR,
    'GRBG': cv2.COLOR_BayerGR2BGR,
    'GBRG': cv2.COLOR_BayerGB2BGR,
    'BGGR': cv2.COLOR_BayerBG2BGR,
}


# ══ Format-Reader (analog ser2mp4.py) ═════════════════════════════════════════

def _read_sharpcap_meta(path):
    txt = path + '.txt'
    if not os.path.exists(txt):
        return {}
    meta = {}
    with open(txt, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if '=' in line:
                k, _, v = line.partition('=')
                meta[k.strip()] = v.strip()
    return meta


class SERReader:
    def __init__(self, path):
        self._f = open(path, 'rb')
        self.info = self._parse_header()

    def _parse_header(self):
        hdr = self._f.read(178)
        if len(hdr) < 178:
            raise ValueError("SER-Header zu kurz")
        color_id    = struct.unpack_from('<i', hdr, 18)[0]
        width       = struct.unpack_from('<i', hdr, 26)[0]
        height      = struct.unpack_from('<i', hdr, 30)[0]
        bit_depth   = struct.unpack_from('<i', hdr, 34)[0]
        frame_count = struct.unpack_from('<i', hdr, 38)[0]
        ser_bayer   = {8: cv2.COLOR_BayerRG2BGR, 9: cv2.COLOR_BayerGR2BGR,
                       10: cv2.COLOR_BayerGB2BGR, 11: cv2.COLOR_BayerBG2BGR}
        return dict(width=width, height=height, bit_depth=bit_depth,
                    frame_count=frame_count,
                    bytes_per_pixel=(bit_depth + 7) // 8,
                    bayer_code=ser_bayer.get(color_id),
                    color_id=color_id, fmt='SER')

    def read_frame(self, index):
        bpp         = self.info['bytes_per_pixel']
        frame_bytes = self.info['width'] * self.info['height'] * bpp
        self._f.seek(178 + index * frame_bytes)
        raw = self._f.read(frame_bytes)
        if len(raw) < frame_bytes:
            return None
        return np.frombuffer(raw, dtype='<u2').reshape(
            self.info['height'], self.info['width'])

    def close(self): self._f.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class AVIReader:
    def __init__(self, path):
        meta       = _read_sharpcap_meta(path)
        bayer_str  = meta.get('Debayer Type', 'RGGB')
        self._cap  = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Kann AVI nicht öffnen: {path}")
        w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n   = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.info = dict(width=w, height=h, bit_depth=8, frame_count=n,
                         bytes_per_pixel=1,
                         bayer_code=BAYER_CODES.get(bayer_str, cv2.COLOR_BayerRG2BGR),
                         color_id=bayer_str, fmt='AVI')
        self._pos = -1

    def read_frame(self, index):
        if index != self._pos + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        if not ret:
            return None
        self._pos = index
        return frame[:, :, 0]  # pal8 → ein Kanal = Bayer-Wert

    def close(self): self._cap.release()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def open_reader(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.ser': return SERReader(path)
    if ext == '.avi': return AVIReader(path)
    raise ValueError(f"Unbekanntes Format: '{ext}' (erwartet .ser oder .avi)")


# ══ Bildverarbeitung ══════════════════════════════════════════════════════════

def debayer(arr, bayer_code):
    """Bayer → BGR float32 mit WB-Korrektur."""
    bgr   = cv2.cvtColor(arr, bayer_code)
    bgr_f = bgr.astype(np.float32)
    bgr_f[:, :, 2] *= WB_R
    bgr_f[:, :, 0] *= WB_B
    return bgr_f


def find_planet_center(bgr_f, crop_size):
    """Planetenzentrum via Helligkeitsschwerpunkt der Top-1%-Pixel."""
    gray    = 0.299*bgr_f[:,:,2] + 0.587*bgr_f[:,:,1] + 0.114*bgr_f[:,:,0]
    blurred = cv2.GaussianBlur(gray, (0, 0), 20)
    thresh  = blurred >= np.percentile(blurred, 99.0)
    ys, xs  = np.where(thresh)
    h, w    = gray.shape
    half    = crop_size // 2
    if len(xs) == 0:
        return w // 2, h // 2
    cx = int(np.clip(np.mean(xs), half, w - half))
    cy = int(np.clip(np.mean(ys), half, h - half))
    return cx, cy


def crop_around(bgr_f, cx, cy, size):
    half = size // 2
    return bgr_f[cy-half:cy+half, cx-half:cx+half]


def sharpness(bgr_f):
    """Laplacian-Varianz als Schärfemaß (höher = schärfer)."""
    gray = 0.299*bgr_f[:,:,2] + 0.587*bgr_f[:,:,1] + 0.114*bgr_f[:,:,0]
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def wavelet_sharpen(img_f, levels=((1, 0.5), (2, 1.8), (4, 1.2), (8, 0.6))):
    """
    Laplacian-Pyramiden-Sharpening (RegiStax-Prinzip).
    Jede Ebene = Differenz zweier Gaußblurs → Detailschicht einer Raumfrequenz.
    Amplifikation: sigma=1 (Rauschen) niedrig, sigma=2-4 (Bänder) hoch.
    """
    result = img_f.copy().astype(np.float32)
    prev   = result.copy()
    for sigma, amount in levels:
        smooth  = cv2.GaussianBlur(prev, (0, 0), sigma)
        detail  = prev - smooth
        result += detail * amount
        prev    = smooth
    return result


def enhance_contrast(bgr_uint8):
    """CLAHE auf LAB-L-Kanal für lokalen Kontrast."""
    lab   = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ══ Stacking ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    with open_reader(args.input) as reader:
        info = reader.info
        n    = info['frame_count']
        bc   = info['bayer_code']

        print(f"{info['fmt']}: {info['width']}x{info['height']} {info['bit_depth']}bit "
              f"| {n} Frames")
        print(f"Optionen: top={args.top}%  crop={args.crop}px  "
              f"min_frames={args.min_frames}")

        # ── Pass 1: Per-Frame-Zentrumserkennung + Qualitäts-Ranking ─────────────
        # Bei nicht-nachgeführten Aufnahmen driftet der Planet über den Frame.
        # → Jeder Frame bekommt sein eigenes crop-Zentrum via Helligkeitsschwerpunkt.
        print(f"\nPass 1: Zentrum-Tracking + Qualitäts-Ranking ({n} Frames)...")
        scores  = np.zeros(n, dtype=np.float32)
        centers = np.zeros((n, 2), dtype=np.int32)

        for i in range(n):
            arr = reader.read_frame(i)
            if arr is None:
                continue
            bgr           = debayer(arr, bc)
            cx, cy        = find_planet_center(bgr, args.crop)
            centers[i]    = [cx, cy]
            cropped        = crop_around(bgr, cx, cy, args.crop)
            scores[i]     = sharpness(cropped)
            if (i + 1) % 500 == 0 or i == 0:
                print(f"  {i+1}/{n}  pos=({cx},{cy})  "
                      f"Score: {scores[:i+1].max():.1f}", flush=True)

        n_stack  = max(args.min_frames, int(n * args.top / 100.0))
        selected = np.argsort(scores)[::-1][:n_stack]
        sel_sort = np.sort(selected)
        best_idx = int(selected[0])
        print(f"Ausgewählt: {n_stack} Frames (Top {args.top}%)"
              f"  Score-Bereich: {scores[selected].min():.1f}–{scores[selected].max():.1f}")

        # ── Referenz-Frame (schärfster Frame) ─────────────────────────────────
        best_cx, best_cy = centers[best_idx]
        ref_bgr  = debayer(reader.read_frame(best_idx), bc)
        ref_crop = crop_around(ref_bgr, best_cx, best_cy, args.crop)
        ref_gray = (0.299*ref_crop[:,:,2] + 0.587*ref_crop[:,:,1]
                    + 0.114*ref_crop[:,:,0])

        # ── Pass 2: Alignen + Stacken ─────────────────────────────────────────
        # Per-Frame-Zentrierung korrigiert den Drift; Phase Correlation
        # korrigiert nur noch das verbleibende Seeing (wenige Pixel).
        print(f"\nPass 2: Stacking ({n_stack} Frames)...")
        sel_set  = set(sel_sort.tolist())
        acc      = np.zeros((args.crop, args.crop, 3), dtype=np.float64)
        w_total  = 0.0
        stacked  = 0
        MAX_SEEING_PX = 20   # max. erwarteter Seeing-Shift nach Driftkorrektur

        for i in range(n):
            if i not in sel_set:
                reader.read_frame(i)
                continue
            arr = reader.read_frame(i)
            if arr is None:
                continue
            bgr        = debayer(arr, bc)
            cx, cy     = centers[i]
            crp        = crop_around(bgr, cx, cy, args.crop)
            gray       = (0.299*crp[:,:,2] + 0.587*crp[:,:,1] + 0.114*crp[:,:,0])

            (dx, dy), _ = cv2.phaseCorrelate(ref_gray, gray)

            # Nur verbleibende Seeing-Shifts akzeptieren
            if abs(dx) > MAX_SEEING_PX or abs(dy) > MAX_SEEING_PX:
                continue

            M       = np.float32([[1, 0, dx], [0, 1, dy]])
            aligned = cv2.warpAffine(crp, M, (args.crop, args.crop),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)

            w        = float(scores[i])
            acc     += aligned.astype(np.float64) * w
            w_total += w
            stacked += 1

            if stacked % 100 == 0 or stacked == 1:
                print(f"  {stacked}/{n_stack}...", flush=True)

        if w_total == 0:
            print("Fehler: Keine Frames gestackt.", file=sys.stderr)
            sys.exit(1)

        discarded = n_stack - stacked
        print(f"\n{stacked} Frames gestackt"
              + (f" ({discarded} wegen Shift-Fehler verworfen)" if discarded else "") + ".")

        # ── Wavelet-Sharpening ────────────────────────────────────────────────
        result_f = acc / w_total
        result_f = wavelet_sharpen(result_f)

        # ── Stretch + Kontrast ────────────────────────────────────────────────
        lo = np.percentile(result_f, 0.05)
        hi = np.percentile(result_f, 99.95)
        result8 = np.clip(
            (result_f - lo) / (hi - lo + 1e-9) * 255, 0, 255
        ).astype(np.uint8)
        result8 = enhance_contrast(result8)

        # ── 16-bit speichern ──────────────────────────────────────────────────
        result16 = result8.astype(np.uint16) * 257  # 8-bit → 16-bit
        cv2.imwrite(args.output, result16)
        size_kb = os.path.getsize(args.output) / 1024
        print(f"Stack: {args.output} ({size_kb:.0f} KB, 16-bit)")

        # ── Kontext-Bild: besten Frame in größerem Ausschnitt ─────────────────
        # Zeigt Jupiter + Monde im Umfeld (kein Stacking, kein Sharpening)
        ctx_bgr = debayer(reader.read_frame(best_idx), bc)
        ctx_cx, ctx_cy = int(centers[best_idx][0]), int(centers[best_idx][1])
        ctx_size = args.crop * 3
        h, w = ctx_bgr.shape[:2]
        half = ctx_size // 2
        ctx_cx = int(np.clip(ctx_cx, half, w - half))
        ctx_cy = int(np.clip(ctx_cy, half, h - half))
        ctx_crop = crop_around(ctx_bgr, ctx_cx, ctx_cy, ctx_size)

        lo2 = np.percentile(ctx_crop, 99.0)
        hi2 = np.percentile(ctx_crop, 99.98)
        ctx8 = np.clip((ctx_crop - lo2) / (hi2 - lo2 + 1e-9) * 255, 0, 255).astype(np.uint8)

        ctx_path = os.path.splitext(args.output)[0] + '_context.png'
        cv2.imwrite(ctx_path, ctx8)
        print(f"Kontext: {ctx_path} ({ctx_size}×{ctx_size}px, bester Frame #{best_idx})")


# ══ CLI ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Planetarisches Lucky-Imaging-Stacking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s capture.avi jupiter.png
  %(prog)s capture.avi jupiter.tif --top=5 --crop=400
  %(prog)s capture.ser mond.png    --top=20 --crop=600
        """,
    )
    p.add_argument('input',  help='Eingabe: .avi oder .ser Datei')
    p.add_argument('output', help='Ausgabe: .png oder .tif (16-bit)')
    p.add_argument('--top', type=float, default=25.0,
                   metavar='PCT', help='Anteil bester Frames in %% (Standard: 25)')
    p.add_argument('--crop', type=int, default=300,
                   metavar='PX', help='Crop-Größe um Planet in Pixel (Standard: 300)')
    p.add_argument('--min_frames', type=int, default=50,
                   metavar='N', help='Mindestanzahl Frames unabhängig von --top (Standard: 50)')
    return p.parse_args()


if __name__ == '__main__':
    main()

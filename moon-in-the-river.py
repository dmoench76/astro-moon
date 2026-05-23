#!/usr/bin/env python3
"""
fit2mp4.py – FIT RAW16 RGGB Sequenz → MP4 (VAAPI h264, exakte Timestamps)
Usage: python3 fit2mp4.py <verzeichnis> <ausgabe.mp4>
"""

import sys
import os
import glob
import subprocess
import tempfile
import numpy as np
import cv2
from astropy.io import fits

# ── Konfiguration ─────────────────────────────────────────────────────────────
QP          = 20
FLIP_VERT   = False
WIDTH       = 1920
HEIGHT      = 1080
FRAME_DIR   = '/tmp/moon_frames'
# ──────────────────────────────────────────────────────────────────────────────


def read_fit_raw16(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
    return np.clip(data, 0, 65535).astype(np.uint16)


def process_frame(arr):
    bgr16 = cv2.cvtColor(arr, cv2.COLOR_BayerRG2BGR)
    bgr_f = bgr16.astype(np.float32)
    bgr_f[:, :, 2] *= 1.05
    bgr_f[:, :, 0] *= 0.88

    lum = 0.299*bgr_f[:,:,2] + 0.587*bgr_f[:,:,1] + 0.114*bgr_f[:,:,0]
    lo = np.percentile(lum, 15.0)
    hi = np.percentile(lum, 99.0)
    bgr_n = np.clip((bgr_f - lo) / (hi - lo), 0, 1)

    bgr8 = (bgr_n ** 0.75 * 255).astype(np.uint8)
    bgr8 = cv2.addWeighted(bgr8, 1.5, cv2.GaussianBlur(bgr8, (0,0), 3), -0.5, 0)

    # Abdunkeln via HSV
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,2] *= 0.6
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    bgr8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if FLIP_VERT:
        bgr8 = cv2.flip(bgr8, 0)

    return bgr8


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <verzeichnis> <ausgabe.mp4>")
        sys.exit(1)

    directory = sys.argv[1].rstrip('/')
    output    = sys.argv[2]

    fits_files = sorted(glob.glob(os.path.join(directory, '*.FIT')))
    if not fits_files:
        print("Keine .FIT-Dateien gefunden.")
        sys.exit(1)

    print(f"{len(fits_files)} FIT-Frames gefunden → {output}")

    # Timestamps aus mtime
    mtimes = [os.stat(f).st_mtime for f in fits_files]
    durations = [mtimes[i+1] - mtimes[i] for i in range(len(mtimes)-1)]
    durations.append(durations[-1])  # letzter Frame: gleiche Duration wie vorletzter

    # Frame-Verzeichnis anlegen
    os.makedirs(FRAME_DIR, exist_ok=True)

    # PNGs rendern
    concat_path = os.path.join(FRAME_DIR, 'concat.txt')
    with open(concat_path, 'w') as concat_f:
        for i, fit_path in enumerate(fits_files):
            try:
                arr  = read_fit_raw16(fit_path)
                bgr8 = process_frame(arr)
                png_path = os.path.join(FRAME_DIR, f'frame_{i:04d}.png')
                cv2.imwrite(png_path, bgr8)
                concat_f.write(f"file '{png_path}'\n")
                concat_f.write(f"duration {durations[i]:.6f}\n")
            except Exception as e:
                print(f"\nFehler bei Frame {i} ({fit_path}): {e}", file=sys.stderr)
                continue

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Frame {i+1}/{len(fits_files)}...", flush=True)

    print("Alle Frames gerendert, starte FFmpeg...")

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-r 24',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_path,
        '-vaapi_device', '/dev/dri/renderD128',
        '-vf', 'format=nv12,hwupload',
        '-c:v', 'h264_vaapi',
        '-qp', str(QP),
        '-movflags', '+faststart',
        output
    ]

    subprocess.run(ffmpeg_cmd, check=True)

    print(f"\nFertig: {output}")
    print(f"Temporäre Frames in {FRAME_DIR} — bei Bedarf löschen: rm -rf {FRAME_DIR}")


if __name__ == '__main__':
    main()

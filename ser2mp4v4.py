#!/usr/bin/env python3
"""
ser2mp4v4.py – SER RAW16 RGGB → MP4, v4
  • Timestamp-basiertes Frame-Mapping → gleichmäßige Mondbewegung
  • Globaler Stretch (kein Per-Frame-Flicker)
  • Echter schwarzer Hintergrund (Maske aus Rohdaten)
  • CLAHE für lokalen Kraterkontrast
  • Feineres USM (sigma=1.5)
  • Doppelte Geschwindigkeit (48 fps)
  • 2 s Fade-in / Fade-out
Usage: python3 ser2mp4v4.py <datei.ser> <ausgabe.mp4>
"""

import sys
import os
import struct
import subprocess
import numpy as np
import cv2

# ── Konfiguration ─────────────────────────────────────────────────────────────
QP            = 20
FLIP_VERT     = True
WB_R          = 1.05
WB_B          = 0.88
STRETCH_LO    = 15.0
STRETCH_HI    = 97.0
GAMMA         = 0.75
BRIGHT        = 0.6
CLAHE_LIMIT   = 2.0
CLAHE_GRID    = (8, 8)
SHARP_AMT     = 1.5
SHARP_SIGMA   = 1.5
SKY_T_LO      = 0.10
SKY_T_HI      = 0.22
SKY_BLUR_SIG  = 15
FPS_OUT       = 48
FADE_SECS     = 2.0
SAMPLE_STEP   = 20
VAAPI_DEV     = '/dev/dri/renderD128'
# ──────────────────────────────────────────────────────────────────────────────


def parse_ser_header(f):
    hdr = f.read(178)
    if len(hdr) < 178:
        raise ValueError("SER-Header zu kurz")
    color_id    = struct.unpack_from('<i', hdr, 18)[0]
    width       = struct.unpack_from('<i', hdr, 26)[0]
    height      = struct.unpack_from('<i', hdr, 30)[0]
    bit_depth   = struct.unpack_from('<i', hdr, 34)[0]
    frame_count = struct.unpack_from('<i', hdr, 38)[0]
    bayer_map = {8: cv2.COLOR_BayerRG2BGR, 9: cv2.COLOR_BayerGR2BGR,
                 10: cv2.COLOR_BayerGB2BGR, 11: cv2.COLOR_BayerBG2BGR}
    return dict(
        width=width, height=height, bit_depth=bit_depth,
        frame_count=frame_count,
        bytes_per_pixel=(bit_depth + 7) // 8,
        bayer_code=bayer_map.get(color_id),
        color_id=color_id,
    )


def read_timestamps(f, info):
    frame_bytes = info['width'] * info['height'] * info['bytes_per_pixel']
    f.seek(178 + info['frame_count'] * frame_bytes)
    ts_raw = f.read(info['frame_count'] * 8)
    if len(ts_raw) < info['frame_count'] * 8:
        return None
    return np.frombuffer(ts_raw, dtype='<u8')


def build_frame_map(ts, n_out):
    """Mappe n_out gleichmäßige Ausgabe-Zeitpunkte auf die nächste echte Aufnahme."""
    targets = ts[0] + np.arange(n_out) / max(n_out - 1, 1) * (ts[-1] - ts[0])
    idx = np.searchsorted(ts, targets).clip(1, len(ts) - 1)
    # Nächstgelegenen der beiden Nachbarn wählen
    left_dist  = np.abs(ts[idx - 1].astype(np.int64) - targets.astype(np.int64))
    right_dist = np.abs(ts[idx    ].astype(np.int64) - targets.astype(np.int64))
    return np.where(left_dist < right_dist, idx - 1, idx)


def read_frame(f, info, index):
    frame_bytes = info['width'] * info['height'] * info['bytes_per_pixel']
    f.seek(178 + index * frame_bytes)
    raw = f.read(frame_bytes)
    if len(raw) < frame_bytes:
        return None
    return np.frombuffer(raw, dtype='<u2').reshape(info['height'], info['width'])


def debayer_wb(arr, bayer_code):
    bgr16 = cv2.cvtColor(arr, bayer_code)
    bgr_f = bgr16.astype(np.float32)
    bgr_f[:, :, 2] *= WB_R
    bgr_f[:, :, 0] *= WB_B
    return bgr_f


def calibrate_stretch(f, info):
    indices = range(0, info['frame_count'], SAMPLE_STEP)
    los, his = [], []
    for i in indices:
        arr = read_frame(f, info, i)
        if arr is None:
            continue
        bgr_f = debayer_wb(arr, info['bayer_code'])
        lum = 0.299*bgr_f[:,:,2] + 0.587*bgr_f[:,:,1] + 0.114*bgr_f[:,:,0]
        los.append(np.percentile(lum, STRETCH_LO))
        his.append(np.percentile(lum, STRETCH_HI))
    lo = float(np.median(los))
    hi = float(np.median(his))
    print(f"Stretch: lo={lo:.0f}  hi={hi:.0f}  ({len(los)} Stichproben)")
    return lo, hi


def process_frame(arr, bayer_code, lo, hi):
    bgr_f = debayer_wb(arr, bayer_code)
    bgr_n = np.clip((bgr_f - lo) / (hi - lo + 1e-6), 0, 1)

    lum_raw  = 0.299*bgr_n[:,:,2] + 0.587*bgr_n[:,:,1] + 0.114*bgr_n[:,:,0]
    sky_mask = np.clip((lum_raw - SKY_T_LO) / (SKY_T_HI - SKY_T_LO), 0, 1).astype(np.float32)
    sky_mask = cv2.GaussianBlur(sky_mask, (0, 0), SKY_BLUR_SIG)

    bgr8 = (bgr_n ** GAMMA * 255).astype(np.uint8)

    lab = cv2.cvtColor(bgr8, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_LIMIT, tileGridSize=CLAHE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    bgr8 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    bgr8 = cv2.addWeighted(bgr8, SHARP_AMT,
                           cv2.GaussianBlur(bgr8, (0, 0), SHARP_SIGMA),
                           1.0 - SHARP_AMT, 0)

    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * BRIGHT, 0, 255)
    bgr8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    bgr8 = (bgr8.astype(np.float32) * sky_mask[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

    if FLIP_VERT:
        bgr8 = cv2.flip(bgr8, 0)
    return bgr8


def build_ffmpeg_cmd(info, output):
    use_vaapi = os.path.exists(VAAPI_DEV)
    base = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pixel_format', 'bgr24',
        '-video_size', f'{info["width"]}x{info["height"]}',
        '-framerate', str(FPS_OUT),
        '-i', 'pipe:0',
    ]
    if use_vaapi:
        enc = ['-vaapi_device', VAAPI_DEV,
               '-vf', 'format=nv12,hwupload',
               '-c:v', 'h264_vaapi', '-qp', str(QP)]
        print(f"Encoder: h264_vaapi ({VAAPI_DEV})")
    else:
        enc = ['-c:v', 'libx264', '-crf', str(QP), '-preset', 'fast']
        print("Encoder: libx264 (kein VAAPI)")
    return base + enc + ['-movflags', '+faststart', output]


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <datei.ser> <ausgabe.mp4>")
        sys.exit(1)

    ser_path = sys.argv[1]
    output   = sys.argv[2]

    with open(ser_path, 'rb') as f:
        info = parse_ser_header(f)

        if info['bayer_code'] is None:
            print(f"Fehler: Unbekannter ColorID {info['color_id']}", file=sys.stderr)
            sys.exit(1)

        print(f"SER: {info['width']}x{info['height']} {info['bit_depth']}bit "
              f"| {info['frame_count']} Frames | → {output}")

        ts = read_timestamps(f, info)
        if ts is not None:
            duration_s = (ts[-1] - ts[0]) / 1e7
            print(f"Timestamps: vorhanden — Echtzeit {duration_s:.1f} s, "
                  f"Ø {(duration_s/(len(ts)-1)*1000):.1f} ms/Frame")
            frame_map = build_frame_map(ts, info['frame_count'])
            unique = len(np.unique(frame_map))
            print(f"Frame-Map: {unique} eindeutige Quell-Frames → {info['frame_count']} Output-Frames")
        else:
            print("Keine Timestamps — lineare Reihenfolge.")
            frame_map = np.arange(info['frame_count'])

        lo, hi = calibrate_stretch(f, info)

        cmd  = build_ffmpeg_cmd(info, output)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        total       = info['frame_count']
        fade_frames = int(FPS_OUT * FADE_SECS)

        for out_i, src_i in enumerate(frame_map):
            arr = read_frame(f, info, int(src_i))
            if arr is None:
                print(f"\nWarnung: Frame {src_i} unvollständig.", file=sys.stderr)
                break

            bgr8 = process_frame(arr, info['bayer_code'], lo, hi)

            if out_i < fade_frames:
                bgr8 = (bgr8 * (out_i / fade_frames)).astype(np.uint8)
            elif out_i >= total - fade_frames:
                bgr8 = (bgr8 * ((total - 1 - out_i) / fade_frames)).astype(np.uint8)

            proc.stdin.write(bgr8.tobytes())

            if (out_i + 1) % 100 == 0 or out_i == 0:
                pct = (out_i + 1) / total * 100
                print(f"  Frame {out_i+1}/{total} ({pct:.0f}%)...", flush=True)

        proc.stdin.close()
        ret = proc.wait()

    if ret == 0:
        size_mb = os.path.getsize(output) / 1024**2
        print(f"\nFertig: {output} ({size_mb:.1f} MB)")
    else:
        print(f"\nFFmpeg-Fehler (Code {ret})", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
ser2mp4v2.py – SER RAW16 RGGB → MP4, v2
  • Globaler Stretch (kein Per-Frame-Flicker)
  • Mehr Kontrast (engere Stretch-Grenzen)
  • Doppelte Wiedergabegeschwindigkeit
Usage: python3 ser2mp4v2.py <datei.ser> <ausgabe.mp4>
"""

import sys
import os
import struct
import subprocess
import numpy as np
import cv2

# ── Konfiguration ─────────────────────────────────────────────────────────────
QP           = 20
FLIP_VERT    = True
WB_R         = 1.05
WB_B         = 0.88
STRETCH_LO   = 15.0   # Percentile Schwarz-Punkt
STRETCH_HI   = 97.0   # enger als v1 (99) → mehr Kontrast
GAMMA        = 0.75
BRIGHT       = 0.6
SHARP_AMT    = 1.5
SHARP_SIGMA  = 3.0
FPS_OUT      = 48     # doppelte Geschwindigkeit gegenüber v1
SAMPLE_STEP  = 20     # jeden N-ten Frame für globale Stretch-Kalibrierung
VAAPI_DEV    = '/dev/dri/renderD128'
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


def read_frame(f, info, index):
    frame_bytes = info['width'] * info['height'] * info['bytes_per_pixel']
    f.seek(178 + index * frame_bytes)
    raw = f.read(frame_bytes)
    if len(raw) < frame_bytes:
        return None
    # SharpCap schreibt little-endian trotz LittleEndian=0 im Header
    return np.frombuffer(raw, dtype='<u2').reshape(info['height'], info['width'])


def debayer_wb(arr, bayer_code):
    bgr16 = cv2.cvtColor(arr, bayer_code)
    bgr_f = bgr16.astype(np.float32)
    bgr_f[:, :, 2] *= WB_R
    bgr_f[:, :, 0] *= WB_B
    return bgr_f


def luminance(bgr_f):
    return 0.299*bgr_f[:,:,2] + 0.587*bgr_f[:,:,1] + 0.114*bgr_f[:,:,0]


def calibrate_stretch(f, info):
    """Stichprobe über die Serie → stabiles globales lo/hi (kein Flicker)."""
    indices = range(0, info['frame_count'], SAMPLE_STEP)
    los, his = [], []
    for i in indices:
        arr = read_frame(f, info, i)
        if arr is None:
            continue
        lum = luminance(debayer_wb(arr, info['bayer_code']))
        los.append(np.percentile(lum, STRETCH_LO))
        his.append(np.percentile(lum, STRETCH_HI))
    lo = float(np.median(los))
    hi = float(np.median(his))
    print(f"Stretch-Kalibrierung: lo={lo:.0f}  hi={hi:.0f}  "
          f"({len(los)} Stichproben-Frames)")
    return lo, hi


def process_frame(arr, bayer_code, lo, hi):
    bgr_f = debayer_wb(arr, bayer_code)
    bgr_n = np.clip((bgr_f - lo) / (hi - lo + 1e-6), 0, 1)
    bgr8  = (bgr_n ** GAMMA * 255).astype(np.uint8)
    bgr8  = cv2.addWeighted(bgr8, SHARP_AMT,
                            cv2.GaussianBlur(bgr8, (0, 0), SHARP_SIGMA),
                            1.0 - SHARP_AMT, 0)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * BRIGHT, 0, 255)
    bgr8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
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

        lo, hi = calibrate_stretch(f, info)

        cmd  = build_ffmpeg_cmd(info, output)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        for i in range(info['frame_count']):
            arr = read_frame(f, info, i)
            if arr is None:
                print(f"\nWarnung: Frame {i} unvollständig.", file=sys.stderr)
                break

            proc.stdin.write(process_frame(arr, info['bayer_code'], lo, hi).tobytes())

            if (i + 1) % 100 == 0 or i == 0:
                pct = (i + 1) / info['frame_count'] * 100
                print(f"  Frame {i+1}/{info['frame_count']} ({pct:.0f}%)...", flush=True)

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

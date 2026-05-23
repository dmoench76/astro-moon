#!/usr/bin/env python3
"""
ser2mp4v5.py – SER RAW16 RGGB → MP4, v5

Usage:
  ser2mp4v5.py input.ser output.mp4 [--fps_out=48] [--color=gold]
  ser2mp4v5.py input.ser output.mp4 --sample_time=3s --sample_from=middle
"""

import sys
import os
import struct
import subprocess
import argparse
import numpy as np
import cv2

# ── Farbpaletten (BGR-Multiplikatoren) ────────────────────────────────────────
COLOR_PRESETS = {
    'pale':    (0.92, 1.02, 1.02),   # fast neutral, minimalster Hauch warm
    'gold':    (0.65, 1.10, 1.08),   # gelb-golden, subtil
    'marsian': (0.42, 0.98, 1.22),   # orange-rot, klar sichtbar
}

# ── Konstante Konfiguration ───────────────────────────────────────────────────
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
FADE_SECS     = 2.0
SAMPLE_STEP   = 20
ALPHA_MIN     = 0.02
VAAPI_DEV     = '/dev/dri/renderD128'
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description='SER RAW16 RGGB → MP4 Mondvideo-Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s capture.ser output.mp4
  %(prog)s capture.ser output.mp4 --fps_out=60 --color=marsian
  %(prog)s capture.ser output.mp4 --sample_time=3s --sample_from=middle
        """,
    )
    p.add_argument('ser_file', help='Eingabe: .ser Datei')
    p.add_argument('output',   help='Ausgabe: .mp4 Datei')
    p.add_argument('--fps_out', type=int, default=48,
                   metavar='N', help='Ausgabe-Framerate (Standard: 48)')
    p.add_argument('--color', choices=list(COLOR_PRESETS), default='gold',
                   help='Farbpalette: pale, gold, marsian (Standard: gold)')
    p.add_argument('--sample_time', default=None,
                   metavar='DAUER', help='Vorschau-Dauer, z.B. 3s oder 500ms')
    p.add_argument('--sample_from', choices=['start', 'middle', 'end'],
                   default='middle', help='Vorschau-Position (Standard: middle)')
    return p.parse_args()


def parse_duration(s):
    """'3s' → 3.0,  '500ms' → 0.5"""
    s = s.strip().lower()
    if s.endswith('ms'):
        return float(s[:-2]) / 1000.0
    if s.endswith('s'):
        return float(s[:-1])
    return float(s)


def sample_output_path(output):
    base, ext = os.path.splitext(output)
    return f"{base}_sample{ext}"


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


def build_interpolation_map(ts, n_out):
    targets = (ts[0] + np.arange(n_out, dtype=np.float64)
               / max(n_out - 1, 1) * float(ts[-1] - ts[0]))
    i_lo = (np.searchsorted(ts.astype(np.float64), targets, side='right') - 1
            ).clip(0, len(ts) - 2)
    i_hi = (i_lo + 1).clip(0, len(ts) - 1)
    span = ts[i_hi].astype(np.float64) - ts[i_lo].astype(np.float64)
    alpha = np.where(span > 0,
                     (targets - ts[i_lo].astype(np.float64)) / (span + 1e-9),
                     0.0).clip(0, 1)
    return i_lo.astype(np.int32), i_hi.astype(np.int32), alpha


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
    los, his = [], []
    for i in range(0, info['frame_count'], SAMPLE_STEP):
        arr = read_frame(f, info, i)
        if arr is None:
            continue
        bgr_f = debayer_wb(arr, info['bayer_code'])
        lum = 0.299*bgr_f[:,:,2] + 0.587*bgr_f[:,:,1] + 0.114*bgr_f[:,:,0]
        los.append(np.percentile(lum, STRETCH_LO))
        his.append(np.percentile(lum, STRETCH_HI))
    lo, hi = float(np.median(los)), float(np.median(his))
    print(f"Stretch: lo={lo:.0f}  hi={hi:.0f}  ({len(los)} Stichproben)")
    return lo, hi


def process_frame(arr, bayer_code, lo, hi, grade):
    grade_b, grade_g, grade_r = grade

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

    g = bgr8.astype(np.float32)
    g[:, :, 0] = np.clip(g[:, :, 0] * grade_b, 0, 255)
    g[:, :, 1] = np.clip(g[:, :, 1] * grade_g, 0, 255)
    g[:, :, 2] = np.clip(g[:, :, 2] * grade_r, 0, 255)
    bgr8 = g.astype(np.uint8)

    if FLIP_VERT:
        bgr8 = cv2.flip(bgr8, 0)
    return bgr8


def interpolate_frames(frame_a, frame_b, alpha, warp_maps_cache, pair_key):
    if pair_key not in warp_maps_cache:
        ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(ga, gb)
        h, w = frame_a.shape[:2]
        cx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        cy = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))
        warp_maps_cache[pair_key] = (dx, dy, cx, cy)
        if len(warp_maps_cache) > 4:
            del warp_maps_cache[next(iter(warp_maps_cache))]

    dx, dy, cx, cy = warp_maps_cache[pair_key]

    map_ax = cx + dx * alpha
    map_ay = cy + dy * alpha
    warped_a = cv2.remap(frame_a, map_ax, map_ay,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    map_bx = cx - dx * (1.0 - alpha)
    map_by = cy - dy * (1.0 - alpha)
    warped_b = cv2.remap(frame_b, map_bx, map_by,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return cv2.addWeighted(warped_a, 1.0 - alpha, warped_b, alpha, 0)


def build_ffmpeg_cmd(info, output, fps_out):
    use_vaapi = os.path.exists(VAAPI_DEV)
    base = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pixel_format', 'bgr24',
        '-video_size', f'{info["width"]}x{info["height"]}',
        '-framerate', str(fps_out),
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


def render(f, info, i_lo_arr, i_hi_arr, alpha_arr, out_indices,
           output, fps_out, lo, hi, grade, fade):
    cmd  = build_ffmpeg_cmd(info, output, fps_out)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    total       = len(out_indices)
    fade_frames = int(fps_out * FADE_SECS) if fade else 0
    frame_cache = {}
    warp_cache  = {}

    for seq_i, out_i in enumerate(out_indices):
        i_lo  = int(i_lo_arr[out_i])
        i_hi  = int(i_hi_arr[out_i])
        alpha = float(alpha_arr[out_i])

        if i_lo not in frame_cache:
            arr = read_frame(f, info, i_lo)
            frame_cache[i_lo] = process_frame(arr, info['bayer_code'], lo, hi, grade)
        frame_lo = frame_cache[i_lo]

        if alpha < ALPHA_MIN or i_lo == i_hi:
            bgr8 = frame_lo
        else:
            if i_hi not in frame_cache:
                arr = read_frame(f, info, i_hi)
                frame_cache[i_hi] = process_frame(arr, info['bayer_code'], lo, hi, grade)
            frame_hi = frame_cache[i_hi]
            bgr8 = interpolate_frames(frame_lo, frame_hi, alpha,
                                      warp_cache, (i_lo, i_hi))

        if fade_frames:
            if seq_i < fade_frames:
                bgr8 = (bgr8 * (seq_i / fade_frames)).astype(np.uint8)
            elif seq_i >= total - fade_frames:
                bgr8 = (bgr8 * ((total - 1 - seq_i) / fade_frames)).astype(np.uint8)

        proc.stdin.write(bgr8.tobytes())

        for k in list(frame_cache):
            if k < i_lo - 1:
                del frame_cache[k]

        if (seq_i + 1) % 100 == 0 or seq_i == 0:
            print(f"  Frame {seq_i+1}/{total} ({(seq_i+1)/total*100:.0f}%)...", flush=True)

    proc.stdin.close()
    return proc.wait()


def main():
    args  = parse_args()
    grade = COLOR_PRESETS[args.color]

    with open(args.ser_file, 'rb') as f:
        info = parse_ser_header(f)

        if info['bayer_code'] is None:
            print(f"Fehler: Unbekannter ColorID {info['color_id']}", file=sys.stderr)
            sys.exit(1)

        print(f"SER: {info['width']}x{info['height']} {info['bit_depth']}bit "
              f"| {info['frame_count']} Frames | → {args.output}")
        print(f"Optionen: fps_out={args.fps_out}  color={args.color} "
              f"[B×{grade[0]}  G×{grade[1]}  R×{grade[2]}]")

        ts = read_timestamps(f, info)
        n  = info['frame_count']
        if ts is not None:
            duration_s = (ts[-1] - ts[0]) / 1e7
            print(f"Timestamps: {duration_s:.1f} s Echtzeit")
            i_lo_arr, i_hi_arr, alpha_arr = build_interpolation_map(ts, n)
        else:
            print("Keine Timestamps — lineare Reihenfolge, keine Interpolation.")
            i_lo_arr  = np.arange(n, dtype=np.int32)
            i_hi_arr  = np.arange(n, dtype=np.int32)
            alpha_arr = np.zeros(n)

        lo, hi = calibrate_stretch(f, info)

        if args.sample_time:
            sample_s      = parse_duration(args.sample_time)
            sample_frames = max(1, int(sample_s * args.fps_out))
            if args.sample_from == 'start':
                start = 0
            elif args.sample_from == 'end':
                start = max(0, n - sample_frames)
            else:  # middle
                start = max(0, (n - sample_frames) // 2)
            out_indices = list(range(start, min(n, start + sample_frames)))
            out_path    = sample_output_path(args.output)
            print(f"Vorschau: {len(out_indices)} Frames ab Index {start} "
                  f"(≈{len(out_indices)/args.fps_out:.1f} s) → {out_path}")
            ret = render(f, info, i_lo_arr, i_hi_arr, alpha_arr,
                         out_indices, out_path, args.fps_out, lo, hi, grade, fade=False)
        else:
            out_path    = args.output
            out_indices = list(range(n))
            ret = render(f, info, i_lo_arr, i_hi_arr, alpha_arr,
                         out_indices, out_path, args.fps_out, lo, hi, grade, fade=True)

    if ret == 0:
        size_mb = os.path.getsize(out_path) / 1024**2
        print(f"\nFertig: {out_path} ({size_mb:.1f} MB)")
    else:
        print(f"\nFFmpeg-Fehler (Code {ret})", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

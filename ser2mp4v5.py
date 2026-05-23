#!/usr/bin/env python3
"""
ser2mp4v5.py – SER RAW16 RGGB → MP4, v5
  • Phase-Correlation-Interpolation → butterweiche Mondbewegung
  • Timestamp-basiertes Frame-Mapping (keine Auto-Exposure-Beschleunigung)
  • Globaler Stretch (kein Per-Frame-Flicker)
  • Echter schwarzer Hintergrund (Maske aus Rohdaten)
  • CLAHE für lokalen Kraterkontrast
  • Feineres USM (sigma=1.5)
  • 48 fps, 2 s Fade-in / Fade-out
Usage: python3 ser2mp4v5.py <datei.ser> <ausgabe.mp4>
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
ALPHA_MIN     = 0.02   # unter diesem Wert kein Interpolieren nötig
# Color-Grading: gelb-golden (G leicht hoch, B runter, R kaum anfassen)
GRADE_B       = 0.65
GRADE_G       = 1.10
GRADE_R       = 1.08
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


def build_interpolation_map(ts, n_out):
    """
    Für jeden Output-Frame: (i_lo, i_hi, alpha).
    alpha=0 → Frame i_lo, alpha=1 → Frame i_hi.
    """
    targets = (ts[0] + np.arange(n_out, dtype=np.float64)
               / max(n_out - 1, 1) * float(ts[-1] - ts[0]))
    i_lo = (np.searchsorted(ts.astype(np.float64), targets, side='right') - 1
            ).clip(0, len(ts) - 2)
    i_hi = (i_lo + 1).clip(0, len(ts) - 1)
    span = (ts[i_hi].astype(np.float64) - ts[i_lo].astype(np.float64))
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

    # Color-Grading: gelb-golden
    f = bgr8.astype(np.float32)
    f[:, :, 0] = np.clip(f[:, :, 0] * GRADE_B, 0, 255)
    f[:, :, 1] = np.clip(f[:, :, 1] * GRADE_G, 0, 255)
    f[:, :, 2] = np.clip(f[:, :, 2] * GRADE_R, 0, 255)
    bgr8 = f.astype(np.uint8)

    if FLIP_VERT:
        bgr8 = cv2.flip(bgr8, 0)
    return bgr8


def interpolate_frames(frame_a, frame_b, alpha, warp_maps_cache, pair_key):
    """
    Phase-Correlation-Interpolation: berechnet Translationsvektor per FFT,
    warpt beide Frames auf die Zwischenposition, blendet.
    Warp-Maps werden pro Frame-Paar gecacht.
    """
    if pair_key not in warp_maps_cache:
        ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(ga, gb)
        h, w = frame_a.shape[:2]
        cx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        cy = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))
        warp_maps_cache[pair_key] = (dx, dy, cx, cy)
        # Cache-Größe begrenzen
        if len(warp_maps_cache) > 4:
            oldest = next(iter(warp_maps_cache))
            del warp_maps_cache[oldest]

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
            print(f"Timestamps: {duration_s:.1f} s Echtzeit")
            i_lo_arr, i_hi_arr, alpha_arr = build_interpolation_map(ts, info['frame_count'])
        else:
            print("Keine Timestamps — lineare Reihenfolge, keine Interpolation.")
            n = info['frame_count']
            i_lo_arr = np.arange(n, dtype=np.int32)
            i_hi_arr = np.arange(n, dtype=np.int32)
            alpha_arr = np.zeros(n)

        lo, hi = calibrate_stretch(f, info)

        cmd  = build_ffmpeg_cmd(info, output)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        total       = info['frame_count']
        fade_frames = int(FPS_OUT * FADE_SECS)
        frame_cache = {}   # src_idx → processed frame
        warp_cache  = {}   # (i_lo, i_hi) → (dx, dy, cx, cy)

        for out_i in range(total):
            i_lo  = int(i_lo_arr[out_i])
            i_hi  = int(i_hi_arr[out_i])
            alpha = float(alpha_arr[out_i])

            # Frames aus Cache oder frisch rendern
            if i_lo not in frame_cache:
                arr = read_frame(f, info, i_lo)
                frame_cache[i_lo] = process_frame(arr, info['bayer_code'], lo, hi)
            frame_lo = frame_cache[i_lo]

            if alpha < ALPHA_MIN or i_lo == i_hi:
                bgr8 = frame_lo
            else:
                if i_hi not in frame_cache:
                    arr = read_frame(f, info, i_hi)
                    frame_cache[i_hi] = process_frame(arr, info['bayer_code'], lo, hi)
                frame_hi = frame_cache[i_hi]
                bgr8 = interpolate_frames(frame_lo, frame_hi, alpha,
                                          warp_cache, (i_lo, i_hi))

            # Fade-in / Fade-out
            if out_i < fade_frames:
                bgr8 = (bgr8 * (out_i / fade_frames)).astype(np.uint8)
            elif out_i >= total - fade_frames:
                bgr8 = (bgr8 * ((total - 1 - out_i) / fade_frames)).astype(np.uint8)

            proc.stdin.write(bgr8.tobytes())

            # Cache-Größe begrenzen (letzte 3 Frames reichen)
            for k in list(frame_cache):
                if k < i_lo - 1:
                    del frame_cache[k]

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

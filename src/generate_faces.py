"""
generate_faces.py
Generates a synthetic dataset of face images.

Default: grayscale PGM (P2/ASCII).
With --color: color PPM (P3/ASCII).

Each call produces N images per expression (smiling, neutral, sad, angry, surprised)
at 16x16 pixels with a small amount of Gaussian noise.

Usage:
        python3 generate_faces.py [--count N] [--out DIR] [--seed SEED] [--color]

Defaults: N=10, out=pgm_faces (or ppm_faces with --color), seed=42
"""

import math, os, random, argparse

WIDTH, HEIGHT = 16, 16

# ── Utility ────────────────────────────────────────────────────────────────────

def clamp(v, lo=0, hi=255):
    if isinstance(v, tuple):
        return tuple(max(lo, min(hi, int(round(c)))) for c in v)
    return max(lo, min(hi, int(round(v))))

def make_canvas(bg=None, rng=None):
    """Blank canvas. bg can be an int (grayscale) or (R,G,B) tuple (color)."""
    if bg is None:
        bg = rng.randint(200, 230) if rng else 215
    return [[bg] * WIDTH for _ in range(HEIGHT)]

# ── Drawing primitives (color-agnostic; accept int or (R,G,B) color) ──────────

def fill_circle(canvas, cx, cy, r, color):
    cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
    for y in range(max(0, cy-r), min(HEIGHT, cy+r+1)):
        for x in range(max(0, cx-r), min(WIDTH, cx+r+1)):
            if (x-cx)**2 + (y-cy)**2 <= r*r:
                canvas[y][x] = clamp(color)

def draw_circle(canvas, cx, cy, r, color, thickness=1):
    cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
    for y in range(max(0, cy-r-2), min(HEIGHT, cy+r+3)):
        for x in range(max(0, cx-r-2), min(WIDTH, cx+r+3)):
            d = math.sqrt((x-cx)**2 + (y-cy)**2)
            if abs(d - r) < thickness:
                canvas[y][x] = clamp(color)

def draw_arc(canvas, cx, cy, r, a0, a1, color, thickness=1, steps=200):
    for i in range(steps+1):
        t = a0 + (a1-a0)*i/steps
        bx, by = cx + r*math.cos(t), cy + r*math.sin(t)
        for dy in range(-thickness+1, thickness):
            for dx in range(-thickness+1, thickness):
                nx, ny = int(round(bx+dx)), int(round(by+dy))
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    canvas[ny][nx] = clamp(color)

def draw_line(canvas, x0, y0, x1, y1, color, thickness=1):
    x0,y0,x1,y1 = int(round(x0)),int(round(y0)),int(round(x1)),int(round(y1))
    dx,dy = abs(x1-x0), abs(y1-y0)
    sx,sy = (1 if x0<x1 else -1), (1 if y0<y1 else -1)
    err,x,y = dx-dy, x0, y0
    while True:
        for dy2 in range(-thickness+1, thickness):
            for dx2 in range(-thickness+1, thickness):
                nx,ny = x+dx2, y+dy2
                if 0<=nx<WIDTH and 0<=ny<HEIGHT:
                    canvas[ny][nx] = clamp(color)
        if x==x1 and y==y1: break
        e2 = 2*err
        if e2 > -dy: err -= dy; x += sx
        if e2 <  dx: err += dx; y += sy

def draw_eyebrow(canvas, cx, cy, width, tilt, color, thickness):
    draw_line(canvas, cx-width//2, cy+tilt, cx+width//2, cy-tilt, color, thickness)

# ── Noise ──────────────────────────────────────────────────────────────────────

def noise_gaussian(canvas, rng, sigma=12):
    for y in range(HEIGHT):
        for x in range(WIDTH):
            p = canvas[y][x]
            if isinstance(p, tuple):
                canvas[y][x] = tuple(clamp(c + rng.gauss(0, sigma)) for c in p)
            else:
                canvas[y][x] = clamp(p + rng.gauss(0, sigma))

# ── Grayscale face generators ──────────────────────────────────────────────────

def random_params(cx, cy, rng):
    """Jittered grayscale face parameters around (cx, cy)."""
    scale = rng.uniform(0.9, 1.1)
    jx = rng.uniform(-0.7, 0.7)
    jy = rng.uniform(-0.7, 0.7)
    dark = rng.randint(25, 55)
    bg = rng.randint(200, 230)
    return dict(cx=cx+jx, cy=cy+jy, scale=scale, dark=dark, bg=bg, rng=rng)

def draw_base(canvas, p):
    s, cx, cy, d = p["scale"], p["cx"], p["cy"], p["dark"]
    rng = p["rng"]
    draw_circle(canvas, cx, cy, 6.2*s, d, max(1, int(round(1*s))))
    ex, ey = 2.8*s, 1.8*s
    er = rng.uniform(0.8, 1.2) * s
    fill_circle(canvas, cx-ex, cy-ey, er, d)
    fill_circle(canvas, cx+ex, cy-ey, er, d)
    fill_circle(canvas, cx-ex+0.2, cy-ey-0.2, max(1, int(round(er*0.3))), 255)
    fill_circle(canvas, cx+ex+0.2, cy-ey-0.2, max(1, int(round(er*0.3))), 255)
    return cx, cy, s, d, ex, ey

def face_smiling(rng, cx=32, cy=32):
    p = random_params(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"], rng=rng)
    cx2, cy2, s, d, ex, ey = draw_base(canvas, p)
    draw_eyebrow(canvas, cx2-ex, cy2-ey-1.1*s, int(2.4*s), 0, d, 1)
    draw_eyebrow(canvas, cx2+ex, cy2-ey-1.1*s, int(2.4*s), 0, d, 1)
    ma = rng.uniform(0.2, 0.3)*math.pi
    draw_arc(canvas, cx2, cy2+2.0*s, 2.3*s, ma, math.pi-ma, d, 1)
    for bx, by in [(cx2-3.6*s, cy2+1.6*s), (cx2+3.6*s, cy2+1.6*s)]:
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx*dx + dy*dy <= 1:
                    nx,ny = int(round(bx+dx)), int(round(by+dy))
                    if 0<=nx<WIDTH and 0<=ny<HEIGHT:
                        canvas[ny][nx] = clamp(canvas[ny][nx] - rng.randint(18,30))
    return canvas

def face_neutral(rng, cx=32, cy=32):
    p = random_params(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"], rng=rng)
    cx2, cy2, s, d, ex, ey = draw_base(canvas, p)
    draw_eyebrow(canvas, cx2-ex, cy2-ey-1.1*s, int(2.4*s), 0, d, 1)
    draw_eyebrow(canvas, cx2+ex, cy2-ey-1.1*s, int(2.4*s), 0, d, 1)
    mw = rng.uniform(1.8, 2.4)*s
    draw_line(canvas, cx2-mw, cy2+2.6*s, cx2+mw, cy2+2.6*s, d+10, 1)
    return canvas

def face_sad(rng, cx=32, cy=32):
    p = random_params(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"], rng=rng)
    cx2, cy2, s, d, ex, ey = draw_base(canvas, p)
    tilt = int(round(rng.uniform(1, 2)))
    draw_eyebrow(canvas, cx2-ex, cy2-ey-1.1*s, int(2.4*s),  tilt, d, 1)
    draw_eyebrow(canvas, cx2+ex, cy2-ey-1.1*s, int(2.4*s), -tilt, d, 1)
    ma = rng.uniform(0.2, 0.3)*math.pi
    draw_arc(canvas, cx2, cy2+3.0*s, 2.1*s, math.pi+ma, 2*math.pi-ma, d, 1)
    tx = cx2 - 3.2*s + rng.uniform(-0.2, 0.2)
    for ty in [cy2+0.8*s, cy2+1.8*s]:
        fill_circle(canvas, tx, ty, max(1, int(round(0.6*s))), 170)
    return canvas

def face_angry(rng, cx=32, cy=32):
    p = random_params(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"], rng=rng)
    cx2, cy2, s, d, ex, ey = draw_base(canvas, p)
    tilt = int(round(rng.uniform(1, 2)))
    draw_eyebrow(canvas, cx2-ex, cy2-ey-1.1*s, int(2.5*s), -tilt, d, 1)
    draw_eyebrow(canvas, cx2+ex, cy2-ey-1.1*s, int(2.5*s),  tilt, d, 1)
    draw_line(canvas, cx2-0.2*s, cy2-ey-0.2*s, cx2+0.1*s, cy2-ey+0.6*s, 80, 1)
    mw = rng.uniform(1.8, 2.4)*s
    draw_line(canvas, cx2-mw, cy2+2.7*s, cx2+mw, cy2+2.7*s, d, 1)
    draw_line(canvas, cx2-mw, cy2+2.7*s, cx2-mw-0.3*s, cy2+3.1*s, d, 1)
    draw_line(canvas, cx2+mw, cy2+2.7*s, cx2+mw+0.3*s, cy2+3.1*s, d, 1)
    return canvas

def face_surprised(rng, cx=32, cy=32):
    p = random_params(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"], rng=rng)
    cx2, cy2, s, d, ex, ey = draw_base(canvas, p)
    er_big = rng.uniform(1.1, 1.5)*s
    fill_circle(canvas, cx2-ex, cy2-ey, er_big, d-10)
    fill_circle(canvas, cx2+ex, cy2-ey, er_big, d-10)
    fill_circle(canvas, cx2-ex+0.2, cy2-ey-0.2, max(1,int(round(er_big*0.3))), 255)
    fill_circle(canvas, cx2+ex+0.2, cy2-ey-0.2, max(1,int(round(er_big*0.3))), 255)
    draw_arc(canvas, cx2-ex, cy2-ey-1.4*s, 1.6*s,
             math.radians(200), math.radians(340), d, 1)
    draw_arc(canvas, cx2+ex, cy2-ey-1.4*s, 1.6*s,
             math.radians(200), math.radians(340), d, 1)
    mr = rng.uniform(1.4, 1.8)*s
    draw_circle(canvas, cx2, cy2+2.8*s, mr, d, 1)
    fill_circle(canvas, cx2, cy2+2.8*s, mr-0.5, rng.randint(50, 80))
    return canvas

GENERATORS = {
    "smiling":   face_smiling,
    "neutral":   face_neutral,
    "sad":       face_sad,
    "angry":     face_angry,
    "surprised": face_surprised,
}

# ── Color face generators ──────────────────────────────────────────────────────

def random_params_color(cx, cy, rng):
    """Jittered color face parameters around (cx, cy)."""
    scale = rng.uniform(0.9, 1.1)
    jx = rng.uniform(-0.7, 0.7)
    jy = rng.uniform(-0.7, 0.7)
    base = rng.randint(40, 75)
    dark = (base, int(base * 0.62), int(base * 0.42))  # warm dark brown
    bg_r = rng.randint(210, 232)
    bg = (bg_r, int(bg_r * 0.92), int(bg_r * 0.84))   # warm beige
    return dict(cx=cx+jx, cy=cy+jy, scale=scale, dark=dark, bg=bg, rng=rng)

def draw_base_color(canvas, p):
    s, cx, cy, d = p["scale"], p["cx"], p["cy"], p["dark"]
    rng = p["rng"]
    draw_circle(canvas, cx, cy, 6.2*s, d, max(1, int(round(1*s))))
    ex, ey = 2.8*s, 1.8*s
    er = rng.uniform(0.8, 1.2) * s
    fill_circle(canvas, cx-ex, cy-ey, er, d)
    fill_circle(canvas, cx+ex, cy-ey, er, d)
    fill_circle(canvas, cx-ex+0.2, cy-ey-0.2, max(1, int(round(er*0.3))), (255, 255, 255))
    fill_circle(canvas, cx+ex+0.2, cy-ey-0.2, max(1, int(round(er*0.3))), (255, 255, 255))
    return cx, cy, s, d, ex, ey

def face_smiling_color(rng, cx=32, cy=32):
    p = random_params_color(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"])
    cx2, cy2, s, d, ex, ey = draw_base_color(canvas, p)
    draw_eyebrow(canvas, cx2-ex, cy2-ey-1.1*s, int(2.4*s), 0, d, 1)
    draw_eyebrow(canvas, cx2+ex, cy2-ey-1.1*s, int(2.4*s), 0, d, 1)
    ma = rng.uniform(0.2, 0.3)*math.pi
    draw_arc(canvas, cx2, cy2+2.0*s, 2.3*s, ma, math.pi-ma, d, 1)
    # Pink cheek flush
    for bx, by in [(cx2-3.6*s, cy2+1.6*s), (cx2+3.6*s, cy2+1.6*s)]:
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx*dx + dy*dy <= 1:
                    nx,ny = int(round(bx+dx)), int(round(by+dy))
                    if 0<=nx<WIDTH and 0<=ny<HEIGHT:
                        r, g, b = canvas[ny][nx]
                        delta = rng.randint(18, 30)
                        canvas[ny][nx] = (clamp(r), clamp(g - delta), clamp(b - delta))
    return canvas

def face_neutral_color(rng, cx=32, cy=32):
    p = random_params_color(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"])
    cx2, cy2, s, d, ex, ey = draw_base_color(canvas, p)
    draw_eyebrow(canvas, cx2-ex, cy2-ey-1.1*s, int(2.4*s), 0, d, 1)
    draw_eyebrow(canvas, cx2+ex, cy2-ey-1.1*s, int(2.4*s), 0, d, 1)
    r, g, b = d
    mouth = (clamp(r + 10), clamp(g + 10), clamp(b + 10))
    mw = rng.uniform(1.8, 2.4)*s
    draw_line(canvas, cx2-mw, cy2+2.6*s, cx2+mw, cy2+2.6*s, mouth, 1)
    return canvas

def face_sad_color(rng, cx=32, cy=32):
    p = random_params_color(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"])
    cx2, cy2, s, d, ex, ey = draw_base_color(canvas, p)
    tilt = int(round(rng.uniform(1, 2)))
    draw_eyebrow(canvas, cx2-ex, cy2-ey-1.1*s, int(2.4*s),  tilt, d, 1)
    draw_eyebrow(canvas, cx2+ex, cy2-ey-1.1*s, int(2.4*s), -tilt, d, 1)
    ma = rng.uniform(0.2, 0.3)*math.pi
    draw_arc(canvas, cx2, cy2+3.0*s, 2.1*s, math.pi+ma, 2*math.pi-ma, d, 1)
    # Blue tears
    tx = cx2 - 3.2*s + rng.uniform(-0.2, 0.2)
    for ty in [cy2+0.8*s, cy2+1.8*s]:
        fill_circle(canvas, tx, ty, max(1, int(round(0.6*s))), (140, 180, 230))
    return canvas

def face_angry_color(rng, cx=32, cy=32):
    p = random_params_color(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"])
    cx2, cy2, s, d, ex, ey = draw_base_color(canvas, p)
    tilt = int(round(rng.uniform(1, 2)))
    draw_eyebrow(canvas, cx2-ex, cy2-ey-1.1*s, int(2.5*s), -tilt, d, 1)
    draw_eyebrow(canvas, cx2+ex, cy2-ey-1.1*s, int(2.5*s),  tilt, d, 1)
    # Reddish brow crease
    draw_line(canvas, cx2-0.2*s, cy2-ey-0.2*s, cx2+0.1*s, cy2-ey+0.6*s, (100, 40, 30), 1)
    mw = rng.uniform(1.8, 2.4)*s
    draw_line(canvas, cx2-mw, cy2+2.7*s, cx2+mw, cy2+2.7*s, d, 1)
    draw_line(canvas, cx2-mw, cy2+2.7*s, cx2-mw-0.3*s, cy2+3.1*s, d, 1)
    draw_line(canvas, cx2+mw, cy2+2.7*s, cx2+mw+0.3*s, cy2+3.1*s, d, 1)
    return canvas

def face_surprised_color(rng, cx=32, cy=32):
    p = random_params_color(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"])
    cx2, cy2, s, d, ex, ey = draw_base_color(canvas, p)
    er_big = rng.uniform(1.1, 1.5)*s
    fill_circle(canvas, cx2-ex, cy2-ey, er_big, d)
    fill_circle(canvas, cx2+ex, cy2-ey, er_big, d)
    fill_circle(canvas, cx2-ex+0.2, cy2-ey-0.2, max(1,int(round(er_big*0.3))), (255, 255, 255))
    fill_circle(canvas, cx2+ex+0.2, cy2-ey-0.2, max(1,int(round(er_big*0.3))), (255, 255, 255))
    draw_arc(canvas, cx2-ex, cy2-ey-1.4*s, 1.6*s,
             math.radians(200), math.radians(340), d, 1)
    draw_arc(canvas, cx2+ex, cy2-ey-1.4*s, 1.6*s,
             math.radians(200), math.radians(340), d, 1)
    mr = rng.uniform(1.4, 1.8)*s
    draw_circle(canvas, cx2, cy2+2.8*s, mr, d, 1)
    # Dark red mouth interior
    fill_circle(canvas, cx2, cy2+2.8*s, mr-0.5, (rng.randint(40, 65), 20, 20))
    return canvas

COLOR_GENERATORS = {
    "smiling":   face_smiling_color,
    "neutral":   face_neutral_color,
    "sad":       face_sad_color,
    "angry":     face_angry_color,
    "surprised": face_surprised_color,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def split_counts(total):
    train_count = int(total * 0.6)
    validation_count = int(total * 0.2)
    test_count = total - train_count - validation_count
    return train_count, validation_count, test_count

# ── File writers ───────────────────────────────────────────────────────────────

def save_pgm(canvas, path):
    with open(path, "w") as f:
        f.write("P2\n")
        f.write(f"# {os.path.basename(path)}\n")
        f.write(f"{WIDTH} {HEIGHT}\n255\n")
        for row in canvas:
            f.write(" ".join(str(v) for v in row) + "\n")

def save_ppm(canvas, path):
    with open(path, "w") as f:
        f.write("P3\n")
        f.write(f"# {os.path.basename(path)}\n")
        f.write(f"{WIDTH} {HEIGHT}\n255\n")
        for row in canvas:
            values = []
            for r, g, b in row:
                values.extend([str(r), str(g), str(b)])
            f.write(" ".join(values) + "\n")

# ── Dataset generation ─────────────────────────────────────────────────────────

def generate_dataset(out_dir, count, seed, color=False):
    generators = COLOR_GENERATORS if color else GENERATORS
    save_fn = save_ppm if color else save_pgm
    ext = "ppm" if color else "pgm"
    fmt = "PPM color" if color else "PGM grayscale"

    total = len(generators) * count
    print(f"\nGenerating {total} {fmt} images  "
          f"({count} × {len(generators)} classes)  "
          f"size={WIDTH}x{HEIGHT}  noise=gaussian  seed={seed}\n")

    train_count, validation_count, _ = split_counts(count)

    for split_name in ("train", "validation", "test"):
        for label in generators:
            os.makedirs(os.path.join(out_dir, split_name, label), exist_ok=True)

    generated = 0
    for label_index, (label, gen_fn) in enumerate(generators.items()):
        for i in range(count):
            if i < train_count:
                split_name = "train"
                split_index = i
            elif i < train_count + validation_count:
                split_name = "validation"
                split_index = i - train_count
            else:
                split_name = "test"
                split_index = i - train_count - validation_count

            rng = random.Random(seed + label_index * 100000 + i * 997)
            cx = rng.uniform(7.5, 8.5)
            cy = rng.uniform(7.5, 8.5)

            canvas = gen_fn(rng, cx=cx, cy=cy)
            noise_gaussian(canvas, rng, sigma=rng.uniform(3, 6))

            fname = f"{label}_{split_index:04d}.{ext}"
            save_fn(canvas, os.path.join(out_dir, split_name, label, fname))
            generated += 1

        print(f"  [{label:10s}]  {count} images  →  {out_dir}/{{train,validation,test}}/{label}/")

    print(f"\n✓ Done. {generated} {ext.upper()} files written to '{out_dir}/'")

# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic face dataset.")
    parser.add_argument("--count", type=int, default=10,
                        help="Images per expression class (default: 10)")
    parser.add_argument("--out",   type=str, default=None,
                        help="Output directory (default: pgm_faces or ppm_faces)")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--color", action="store_true",
                        help="Generate color PPM images instead of grayscale PGM")
    args = parser.parse_args()
    out = args.out or ("data/color" if args.color else "data/black_white")
    generate_dataset(out, args.count, args.seed, color=args.color)

if __name__ == "__main__":
    main()

"""
generate_faces.py
Generates a synthetic dataset of tiny PGM (Portable Graymap P2/ASCII) face images.

Each call produces N images per expression (smiling, neutral, sad, angry, surprised)
at 16x16 pixels with a small amount of Gaussian noise.

Usage:
        python3 generate_faces.py [--count N] [--out DIR] [--seed SEED]

Defaults: N=10, out=pgm_faces, seed=42
"""

import math, os, random, argparse

WIDTH, HEIGHT = 16, 16

# ── Utility ────────────────────────────────────────────────────────────────────

def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, int(round(v))))

def make_canvas(bg=None, rng=None):
    """Blank canvas with a slightly randomised background brightness."""
    if bg is None:
        bg = rng.randint(200, 230) if rng else 215
    return [[bg] * WIDTH for _ in range(HEIGHT)]

# ── Drawing primitives (all accept float coords) ───────────────────────────────

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

# ── Noise function ─────────────────────────────────────────────────────────────

def noise_gaussian(canvas, rng, sigma=12):
    for y in range(HEIGHT):
        for x in range(WIDTH):
            canvas[y][x] = clamp(canvas[y][x] + rng.gauss(0, sigma))

# ── Face generators ────────────────────────────────────────────────────────────
# Each generator accepts a `rng` and a `p` dict of random parameters.

def random_params(cx, cy, rng):
    """Produce a dict of jittered face parameters around (cx, cy)."""
    scale = rng.uniform(0.9, 1.1)
    jx = rng.uniform(-0.7, 0.7)
    jy = rng.uniform(-0.7, 0.7)
    dark = rng.randint(25, 55)
    bg = rng.randint(200, 230)
    return dict(cx=cx+jx, cy=cy+jy, scale=scale, dark=dark, bg=bg, rng=rng)

def draw_base(canvas, p):
    s, cx, cy, d = p["scale"], p["cx"], p["cy"], p["dark"]
    rng = p["rng"]
    # Head
    draw_circle(canvas, cx, cy, 6.2*s, d, max(1, int(round(1*s))))
    # Eyes
    ex = 2.8*s
    ey = 1.8*s
    er = rng.uniform(0.8, 1.2) * s
    fill_circle(canvas, cx-ex, cy-ey, er, d)
    fill_circle(canvas, cx+ex, cy-ey, er, d)
    # Highlights
    fill_circle(canvas, cx-ex+0.2, cy-ey-0.2, max(1, int(round(er*0.3))), 255)
    fill_circle(canvas, cx+ex+0.2, cy-ey-0.2, max(1, int(round(er*0.3))), 255)
    return cx, cy, s, d, ex, ey

def face_smiling(rng, cx=32, cy=32):
    p = random_params(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"], rng=rng)
    cx2, cy2, s, d, ex, ey = draw_base(canvas, p)
    # Eyebrows – slight raise
    draw_eyebrow(canvas, cx2-ex, cy2-ey-1.1*s, int(2.4*s), 0, d, 1)
    draw_eyebrow(canvas, cx2+ex, cy2-ey-1.1*s, int(2.4*s), 0, d, 1)
    # Smile arc
    ma = rng.uniform(0.2, 0.3)*math.pi
    draw_arc(canvas, cx2, cy2+2.0*s, 2.3*s, ma, math.pi-ma, d, 1)
    # Cheeks
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
    # Flat mouth
    mw = rng.uniform(1.8, 2.4)*s
    draw_line(canvas, cx2-mw, cy2+2.6*s, cx2+mw, cy2+2.6*s, d+10, 1)
    return canvas

def face_sad(rng, cx=32, cy=32):
    p = random_params(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"], rng=rng)
    cx2, cy2, s, d, ex, ey = draw_base(canvas, p)
    # Sad brows: inner ends up
    tilt = int(round(rng.uniform(1, 2)))
    draw_eyebrow(canvas, cx2-ex, cy2-ey-1.1*s, int(2.4*s),  tilt, d, 1)
    draw_eyebrow(canvas, cx2+ex, cy2-ey-1.1*s, int(2.4*s), -tilt, d, 1)
    # Frown arc
    ma = rng.uniform(0.2, 0.3)*math.pi
    draw_arc(canvas, cx2, cy2+3.0*s, 2.1*s, math.pi+ma, 2*math.pi-ma, d, 1)
    # Teardrop
    tx = cx2 - 3.2*s + rng.uniform(-0.2, 0.2)
    for ty in [cy2+0.8*s, cy2+1.8*s]:
        fill_circle(canvas, tx, ty, max(1, int(round(0.6*s))), 170)
    return canvas

def face_angry(rng, cx=32, cy=32):
    p = random_params(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"], rng=rng)
    cx2, cy2, s, d, ex, ey = draw_base(canvas, p)
    # Angry brows: inner down, outer up
    tilt = int(round(rng.uniform(1, 2)))
    draw_eyebrow(canvas, cx2-ex, cy2-ey-1.1*s, int(2.5*s), -tilt, d, 1)
    draw_eyebrow(canvas, cx2+ex, cy2-ey-1.1*s, int(2.5*s),  tilt, d, 1)
    # Brow crease
    draw_line(canvas, cx2-0.2*s, cy2-ey-0.2*s, cx2+0.1*s, cy2-ey+0.6*s, 80, 1)
    # Clenched mouth
    mw = rng.uniform(1.8, 2.4)*s
    draw_line(canvas, cx2-mw, cy2+2.7*s, cx2+mw, cy2+2.7*s, d, 1)
    draw_line(canvas, cx2-mw, cy2+2.7*s, cx2-mw-0.3*s, cy2+3.1*s, d, 1)
    draw_line(canvas, cx2+mw, cy2+2.7*s, cx2+mw+0.3*s, cy2+3.1*s, d, 1)
    return canvas

def face_surprised(rng, cx=32, cy=32):
    p = random_params(cx, cy, rng)
    canvas = make_canvas(bg=p["bg"], rng=rng)
    cx2, cy2, s, d, ex, ey = draw_base(canvas, p)
    # Wide eyes (override base with larger pupils)
    er_big = rng.uniform(1.1, 1.5)*s
    fill_circle(canvas, cx2-ex, cy2-ey, er_big, d-10)
    fill_circle(canvas, cx2+ex, cy2-ey, er_big, d-10)
    fill_circle(canvas, cx2-ex+0.2, cy2-ey-0.2, max(1,int(round(er_big*0.3))), 255)
    fill_circle(canvas, cx2+ex+0.2, cy2-ey-0.2, max(1,int(round(er_big*0.3))), 255)
    # Raised arched brows
    draw_arc(canvas, cx2-ex, cy2-ey-1.4*s, 1.6*s,
             math.radians(200), math.radians(340), d, 1)
    draw_arc(canvas, cx2+ex, cy2-ey-1.4*s, 1.6*s,
             math.radians(200), math.radians(340), d, 1)
    # O-mouth
    mr = rng.uniform(1.4, 1.8)*s
    draw_circle(canvas, cx2, cy2+2.8*s, mr, d, 1)
    fill_circle(canvas, cx2, cy2+2.8*s, mr-0.5, rng.randint(50, 80))
    return canvas

# ── Label → generator map ──────────────────────────────────────────────────────

GENERATORS = {
    "smiling":   face_smiling,
    "neutral":   face_neutral,
    "sad":       face_sad,
    "angry":     face_angry,
    "surprised": face_surprised,
}

def split_counts(total):
    train_count = int(total * 0.6)
    validation_count = int(total * 0.2)
    test_count = total - train_count - validation_count
    return train_count, validation_count, test_count

# ── PGM writer ─────────────────────────────────────────────────────────────────

def save_pgm(canvas, path):
    with open(path, "w") as f:
        f.write("P2\n")
        f.write(f"# {os.path.basename(path)}\n")
        f.write(f"{WIDTH} {HEIGHT}\n255\n")
        for row in canvas:
            f.write(" ".join(str(v) for v in row) + "\n")

def generate_dataset(out_dir, count, seed):
    total = len(GENERATORS) * count
    print(f"\nGenerating {total} PGM images  "
          f"({count} × {len(GENERATORS)} classes)  "
          f"size={WIDTH}x{HEIGHT}  noise=gaussian  seed={seed}\n")

    train_count, validation_count, test_count = split_counts(count)
    generated = 0

    for split_name in ("train", "validation", "test"):
        for label in GENERATORS:
            os.makedirs(os.path.join(out_dir, split_name, label), exist_ok=True)

    for label_index, (label, gen_fn) in enumerate(GENERATORS.items()):
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

            # Each image gets its own seeded RNG → fully reproducible yet varied
            rng = random.Random(seed + label_index * 100000 + i * 997)

            # Random face centre (slight off-centre variation)
            cx = rng.uniform(7.5, 8.5)
            cy = rng.uniform(7.5, 8.5)

            canvas = gen_fn(rng, cx=cx, cy=cy)

            noise_gaussian(canvas, rng, sigma=rng.uniform(3, 6))

            fname = f"{label}_{split_index:04d}.pgm"
            save_pgm(canvas, os.path.join(out_dir, split_name, label, fname))
            generated += 1

        print(f"  [{label:10s}]  {count} images  →  {out_dir}/{{train,validation,test}}/{label}/")

    print(f"\n✓ Done. {generated} PGM files written to '{out_dir}/'")
    print("  Open with: display pgm_faces/train/smiling/smiling_0000.pgm  (ImageMagick)")
    print("  Or batch-convert: mogrify -format png pgm_faces/**/*.pgm\n")

# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PGM face dataset.")
    parser.add_argument("--count",  type=int,   default=10,
                        help="Images per expression class (default: 10)")
    parser.add_argument("--out",    type=str,   default="pgm_faces",
                        help="Output directory (default: pgm_faces)")
    parser.add_argument("--seed",   type=int,   default=42,
                        help="Base random seed (default: 42)")
    args = parser.parse_args()
    generate_dataset(args.out, args.count, args.seed)

if __name__ == "__main__":
    main()
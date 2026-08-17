#!/usr/bin/env python3
"""
Math Garden - a Gray-Scott reaction-diffusion system that grows a little
every day, renders itself as a lit relief sculpture, and hangs itself
on a gallery wall.

State persists in state.npz. Each run advances the chemistry, moves the
parameters one step along a curated journey through Pearson's parameter
space, renders a new plate, and rebuilds the gallery.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SIZE = 224             # simulation grid (wraps as a torus)
STEPS = 7000           # chemistry steps per run
RENDER_PX = 1100       # full-size plate
ARCHIVE_PX = 900       # archived plate
THUMB_PX = 320         # gallery strip thumbnail
MAX_ARCHIVE = 90       # keep this many past works on disk

DU, DV = 0.16, 0.08
DT = 1.0

RUNS_PER_REGIME = 4    # runs spent in each regime before moving on
FEED_SPREAD = 0.0035   # how far feed drifts across the canvas
KILL_SPREAD = 0.0018   # how far kill drifts across the canvas

DOCS = "docs"
ARCHIVE_DIR = os.path.join(DOCS, "archive")
STATE_FILE = "state.npz"
MANIFEST_FILE = os.path.join(DOCS, "archive.json")

# ----------------------------------------------------------------------
# The journey: named regions of Gray-Scott parameter space.
# Each produces a visually distinct behaviour.
# ----------------------------------------------------------------------
JOURNEY = [
    ("Labyrinth",   0.0290, 0.0570),
    ("Perforation", 0.0390, 0.0580),
    ("Turbulence",  0.0260, 0.0510),
    ("Fingerprint", 0.0300, 0.0560),
    ("Ripple",      0.0180, 0.0510),
    ("Migration",   0.0140, 0.0450),
    ("Vesicle",     0.0300, 0.0620),
    ("Pulse",       0.0250, 0.0600),
]

# ----------------------------------------------------------------------
# Palettes: 5-stop gradients, deep shadow -> lit highlight
# ----------------------------------------------------------------------
PALETTES = [
    ("Verdigris",   [(0.00, (10, 38, 42)),  (0.30, (16, 72, 74)),   (0.58, (36, 132, 118)), (0.82, (128, 204, 172)), (1.00, (238, 252, 236))]),
    ("Oxblood",     [(0.00, (44, 12, 16)),  (0.30, (92, 20, 26)),   (0.58, (162, 46, 42)),  (0.82, (232, 132, 78)),  (1.00, (254, 236, 202))]),
    ("Ultramarine", [(0.00, (14, 22, 58)),  (0.30, (26, 48, 116)),  (0.58, (54, 100, 196)), (0.82, (136, 188, 248)), (1.00, (240, 248, 255))]),
    ("Aurum",       [(0.00, (38, 26, 10)),  (0.30, (84, 56, 16)),   (0.58, (168, 120, 34)), (0.82, (238, 194, 96)),  (1.00, (255, 250, 224))]),
    ("Bone & Ink",  [(0.00, (28, 30, 36)),  (0.30, (62, 64, 72)),   (0.58, (128, 126, 132)),(0.82, (206, 202, 196)), (1.00, (250, 249, 245))]),
    ("Ultraviolet", [(0.00, (30, 14, 52)),  (0.30, (66, 22, 100)),  (0.58, (134, 50, 176)), (0.82, (218, 132, 230)), (1.00, (250, 232, 255))]),
    ("Glacier",     [(0.00, (14, 36, 54)),  (0.30, (20, 74, 100)),  (0.58, (54, 142, 168)), (0.82, (158, 220, 232)), (1.00, (243, 253, 255))]),
    ("Ember",       [(0.00, (40, 16, 14)),  (0.30, (88, 30, 14)),   (0.58, (176, 66, 20)),  (0.82, (244, 152, 54)),  (1.00, (255, 242, 200))]),
    ("Moss",        [(0.00, (26, 34, 18)),  (0.30, (48, 70, 26)),   (0.58, (100, 136, 50)), (0.82, (184, 208, 118)), (1.00, (245, 250, 220))]),
    ("Nocturne",    [(0.00, (20, 24, 44)),  (0.30, (38, 44, 84)),   (0.58, (84, 82, 146)),  (0.82, (168, 160, 222)), (1.00, (240, 240, 254))]),
]

# ----------------------------------------------------------------------
# Title vocabulary
# ----------------------------------------------------------------------
ADJECTIVES = [
    "Silent", "Slow", "Hollow", "Patient", "Drowned", "Waking", "Ancient",
    "Quiet", "Burning", "Folded", "Distant", "Tidal", "Iron", "Glass",
    "Sacred", "Restless", "Buried", "First", "Late", "Unfinished",
]
NOUNS = [
    "Meridian", "Aperture", "Vesper", "Lacuna", "Cascade", "Reliquary",
    "Threshold", "Ember", "Palimpsest", "Solstice", "Alluvium", "Cynosure",
    "Garden", "Choir", "Interval", "Bloom", "Aquifer", "Canticle",
    "Foundry", "Estuary",
]


# ----------------------------------------------------------------------
# Chemistry
# ----------------------------------------------------------------------
def laplacian(Z):
    """9-point kernel: center -1, orthogonal 0.2, diagonal 0.05.

    Sums to zero, so a flat field diffuses to nothing. This is the part
    that was broken before.
    """
    up, dn = np.roll(Z, 1, 0), np.roll(Z, -1, 0)
    return (
        -Z
        + 0.20 * (up + dn + np.roll(Z, 1, 1) + np.roll(Z, -1, 1))
        + 0.05 * (
            np.roll(up, 1, 1) + np.roll(up, -1, 1)
            + np.roll(dn, 1, 1) + np.roll(dn, -1, 1)
        )
    )


def step(A, B, feed, kill):
    reaction = A * B * B
    A += DT * (DU * laplacian(A) - reaction + feed * (1.0 - A))
    B += DT * (DV * laplacian(B) + reaction - (feed + kill) * B)
    np.clip(A, 0.0, 1.0, out=A)
    np.clip(B, 0.0, 1.0, out=B)
    return A, B


def parameter_field(base_feed, base_kill, rng):
    """Smooth, low-frequency variation of feed/kill across the canvas.

    This is what gives each plate composition instead of uniform wallpaper:
    different regions of the same canvas sit in different regimes and blend
    into one another.
    """
    def field(amplitude):
        n = blur(rng.random((SIZE, SIZE)).astype(np.float32), 70)
        n -= n.min()
        span = n.max() or 1.0
        return ((n / span) * 2.0 - 1.0) * amplitude

    feed = np.clip(base_feed + field(FEED_SPREAD), 0.008, 0.075).astype(np.float32)
    kill = np.clip(base_kill + field(KILL_SPREAD), 0.040, 0.070).astype(np.float32)
    return feed, kill


def sow(A, B, rng, count=7):
    """Drop circular seed blobs of B into the medium (wrapping at the edges)."""
    axis = np.arange(SIZE)
    for _ in range(count):
        cy, cx = rng.integers(0, SIZE, 2)
        r = int(rng.integers(3, 8))
        dy = np.minimum(np.abs(axis - cy), SIZE - np.abs(axis - cy))
        dx = np.minimum(np.abs(axis - cx), SIZE - np.abs(axis - cx))
        disc = (dy[:, None] ** 2 + dx[None, :] ** 2) <= r * r
        A[disc] = 0.50
        B[disc] = 0.25
    noise = rng.uniform(-0.02, 0.02, (SIZE, SIZE)).astype(np.float32)
    np.clip(A + noise, 0, 1, out=A)
    np.clip(B + noise, 0, 1, out=B)
    return A, B


def fresh_medium(rng):
    A = np.ones((SIZE, SIZE), dtype=np.float32)
    B = np.zeros((SIZE, SIZE), dtype=np.float32)
    return sow(A, B, rng, count=70)


def evolve(A, B, feed, kill, rng):
    """Run the chemistry, resuscitating the system if it dies out."""
    chunk = 400
    for _ in range(0, STEPS, chunk):
        for _ in range(chunk):
            A, B = step(A, B, feed, kill)
        if not np.isfinite(B).all():
            A, B = fresh_medium(rng)
        elif B.mean() < 0.004:          # medium has gone sterile
            A, B = sow(A, B, rng, count=40)
    return A, B


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def blur(Z, passes):
    """Separable [1,2,1] blur, repeated. Approaches a Gaussian."""
    out = Z
    for _ in range(passes):
        for axis in (0, 1):
            out = 0.25 * (np.roll(out, 1, axis) + 2.0 * out + np.roll(out, -1, axis))
    return out


def build_lut(stops):
    pos = np.array([s[0] for s in stops], dtype=np.float32)
    cols = np.array([s[1] for s in stops], dtype=np.float32) / 255.0
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    return np.stack([np.interp(x, pos, cols[:, c]) for c in range(3)], axis=1).astype(np.float32)


def render(B, lut, seed, px=RENDER_PX):
    """Turn the chemical field into a lit relief surface."""
    # Robust contrast stretch
    lo, hi = np.percentile(B, 4.0), np.percentile(B, 98.0)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    h = np.clip((B - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    # Smooth upscale to plate resolution
    h = np.asarray(
        Image.fromarray(h, mode="F").resize((px, px), Image.BICUBIC),
        dtype=np.float32,
    )
    h = np.clip(h, 0.0, 1.0)
    surface = blur(h, 2)
    tone = h ** 0.52          # lift midtones out of the shadows

    # Surface normals from the height field
    gy, gx = np.gradient(surface)
    relief = px * 0.30
    nx, ny = -gx * relief, -gy * relief
    nz = np.ones_like(surface)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx * inv, ny * inv, nz * inv

    # Key light from upper left, viewer straight on
    lx, ly, lz = -0.42, -0.58, 0.70
    m = 1.0 / np.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx * m, ly * m, lz * m
    diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)

    hx, hy, hz = lx, ly, lz + 1.0
    m = 1.0 / np.sqrt(hx * hx + hy * hy + hz * hz)
    spec = np.clip(nx * hx * m + ny * hy * m + nz * hz * m, 0.0, 1.0) ** 42.0

    # Rim light from the lower right keeps the shadows from going flat
    rim = np.clip(nx * 0.55 + ny * 0.55 + nz * 0.30, 0.0, 1.0) ** 3.0

    base = lut[(tone * 255).astype(np.uint8)]
    shade = (0.58 + 0.62 * diffuse)[..., None]
    img = base * shade
    img += spec[..., None] * np.array([0.42, 0.43, 0.47], dtype=np.float32)
    img += rim[..., None] * base * 0.35

    # Bloom
    luma = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    img += blur(np.clip(luma - 0.62, 0.0, 1.0), 12)[..., None] * 1.15 * np.array(
        [1.0, 0.97, 0.90], dtype=np.float32
    )

    # Vignette
    ax = np.linspace(-1.0, 1.0, px, dtype=np.float32)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    img *= (1.0 - 0.34 * np.clip(np.sqrt(xx * xx + yy * yy) / 1.414, 0, 1) ** 2.2)[..., None]

    # Filmic shoulder, then a light S-curve
    img = np.clip(img, 0.0, None)
    img = img * (1.0 + img / 4.0) / (1.0 + img)
    img = np.clip(img, 0.0, 1.0)
    img = img * 0.72 + (img * img * (3.0 - 2.0 * img)) * 0.28

    # Fine grain, so it reads as a printed plate rather than a screenshot
    grain = np.random.default_rng(seed).normal(0.0, 0.010, (px, px, 1)).astype(np.float32)
    img = np.clip(img + grain, 0.0, 1.0)

    return Image.fromarray((img * 255).astype(np.uint8), mode="RGB")


# ----------------------------------------------------------------------
# Naming
# ----------------------------------------------------------------------
def name_work(run_index, regime, palette_name):
    digest = hashlib.sha256(f"{run_index}|{regime}|{palette_name}".encode()).digest()
    return f"{ADJECTIVES[digest[0] % len(ADJECTIVES)]} {NOUNS[digest[1] % len(NOUNS)]}"


# ----------------------------------------------------------------------
# Gallery page
# ----------------------------------------------------------------------
def build_page(current, works):
    strip = "".join(
        f'<a class="past" href="archive/{w["file"]}" '
        f'title="{w["title"]} - generation {w["generation"]:,}">'
        f'<img loading="lazy" src="archive/{w["thumb"]}" alt="{w["title"]}">'
        f'<span>{w["title"]}</span></a>'
        for w in works
    )
    strip_block = (
        f'<section class="series" aria-label="Earlier works">'
        f'<h2>Earlier in this series</h2><div class="strip">{strip}</div></section>'
        if works else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0d1014">
<meta http-equiv="refresh" content="21600">
<title>{current['title']} - Math Garden</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300&display=swap" rel="stylesheet">
<style>
  :root {{
    --wall:#0d1014; --wall-lit:#1c222a;
    --brass:#b99a5e; --brass-dim:#7d6740;
    --mat:#ded8cc; --mat-shadow:#a49b8b;
    --frame:#191512;
    --serif:"Cormorant Garamond",Georgia,serif;
    --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  }}
  *{{box-sizing:border-box}}
  body{{
    margin:0; min-height:100vh; background:var(--wall); color:#8f8574;
    font-family:var(--sans); -webkit-font-smoothing:antialiased;
    padding:clamp(2rem,7vw,5rem) clamp(1rem,5vw,3rem) 4rem;
    display:flex; flex-direction:column; align-items:center;
  }}
  /* The spotlight */
  body::before{{
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
    background:
      radial-gradient(ellipse 78% 52% at 50% -6%, #232a34 0%, rgba(35,42,52,0) 68%),
      radial-gradient(ellipse 42% 34% at 50% 26%, rgba(196,178,140,.13) 0%, rgba(196,178,140,0) 70%);
    animation:breathe 19s ease-in-out infinite;
  }}
  @keyframes breathe{{0%,100%{{opacity:.86}}50%{{opacity:1}}}}
  main{{position:relative; z-index:1; width:100%; max-width:660px; text-align:center}}

  .eyebrow{{
    font-size:.62rem; letter-spacing:.34em; text-transform:uppercase;
    color:var(--brass-dim); margin:0 0 2.4rem;
  }}

  /* Frame stack: walnut frame -> mat -> bevel -> plate */
  .frame{{
    background:linear-gradient(150deg,#241d17,#0f0c0a 55%,#1d1712);
    padding:clamp(.5rem,1.6vw,.85rem); border-radius:2px;
    box-shadow:
      0 2px 3px rgba(0,0,0,.7),
      0 26px 46px -12px rgba(0,0,0,.85),
      0 60px 90px -30px rgba(0,0,0,.7),
      inset 0 1px 0 rgba(190,160,110,.22);
    animation:hang 1.1s cubic-bezier(.16,.84,.34,1) both;
  }}
  @keyframes hang{{from{{opacity:0;transform:translateY(-14px)}}to{{opacity:1;transform:none}}}}
  .mat{{
    background:linear-gradient(168deg,var(--mat),#c6bfb0);
    padding:clamp(1.4rem,5.5vw,3rem);
    box-shadow:inset 0 1px 2px rgba(255,255,255,.5);
  }}
  .bevel{{
    padding:2px;
    background:linear-gradient(150deg,#fffdf7 0%,#efe9dc 42%,var(--mat-shadow) 100%);
    box-shadow:0 0 0 1px rgba(120,110,95,.45), 0 6px 14px -4px rgba(0,0,0,.55);
  }}
  .plate{{display:block; width:100%; height:auto; background:#0b0d10}}

  /* The plaque */
  .plaque{{
    margin:2.5rem auto 0; max-width:29rem; padding:1.5rem 1.75rem 1.35rem;
    background:linear-gradient(163deg,#2c2213,#191309 48%,#241b0f);
    border-top:1px solid rgba(214,186,132,.30);
    border-bottom:1px solid rgba(0,0,0,.6);
    box-shadow:0 12px 26px -14px rgba(0,0,0,.9), inset 0 0 34px rgba(0,0,0,.45);
  }}
  .plaque h1{{
    font-family:var(--serif); font-style:italic; font-weight:300;
    font-size:clamp(1.9rem,6.2vw,2.7rem); line-height:1.08;
    margin:0 0 .15rem; color:#e2c98f;
    text-shadow:0 1px 0 rgba(0,0,0,.85), 0 -1px 0 rgba(226,201,143,.16);
  }}
  .plaque .attrib{{
    font-family:var(--serif); font-size:1.02rem; color:var(--brass-dim);
    margin:0 0 1.15rem;
  }}
  .rule{{height:1px; background:linear-gradient(90deg,transparent,rgba(201,171,110,.4),transparent); margin:0 0 1.15rem}}
  dl{{
    margin:0; display:grid; grid-template-columns:auto 1fr; gap:.42rem 1.1rem;
    font-size:.72rem; text-align:left;
  }}
  dt{{letter-spacing:.16em; text-transform:uppercase; color:#6f5c39; white-space:nowrap}}
  dd{{margin:0; color:#a9946c; font-variant-numeric:tabular-nums}}

  /* Earlier works */
  .series{{margin-top:5.5rem; width:100%}}
  .series h2{{
    font-size:.62rem; letter-spacing:.32em; text-transform:uppercase;
    color:var(--brass-dim); font-weight:400; margin:0 0 1.5rem;
  }}
  .strip{{
    display:flex; gap:1rem; overflow-x:auto; padding-bottom:1rem;
    scroll-snap-type:x mandatory; -webkit-overflow-scrolling:touch;
  }}
  .past{{
    flex:0 0 auto; width:118px; text-decoration:none; scroll-snap-align:start;
    opacity:.55; transition:opacity .35s, transform .35s;
  }}
  .past:hover,.past:focus-visible{{opacity:1; transform:translateY(-3px)}}
  .past:focus-visible{{outline:1px solid var(--brass); outline-offset:6px}}
  .past img{{
    width:100%; height:auto; display:block; border:3px solid #17120e;
    box-shadow:0 8px 18px -8px rgba(0,0,0,.9);
  }}
  .past span{{
    display:block; margin-top:.5rem; font-family:var(--serif); font-style:italic;
    font-size:.82rem; color:#8e7a52;
  }}
  footer{{
    margin-top:4rem; font-size:.6rem; letter-spacing:.2em; text-transform:uppercase;
    color:#4a4136; text-align:center; line-height:2;
  }}
  @media (prefers-reduced-motion:reduce){{
    *{{animation:none!important; transition:none!important}}
  }}
</style>
</head>
<body>
<main>
  <p class="eyebrow">Math Garden &middot; Room I</p>

  <figure class="frame" style="margin:0">
    <div class="mat"><div class="bevel">
      <img class="plate" src="current.jpg?v={current['generation']}"
           alt="{current['title']}: a reaction-diffusion pattern in {current['palette'].lower()} tones"
           width="{RENDER_PX}" height="{RENDER_PX}">
    </div></div>
  </figure>

  <div class="plaque">
    <h1>{current['title']}</h1>
    <p class="attrib">Gray&ndash;Scott process, {current['date_long']}</p>
    <div class="rule"></div>
    <dl>
      <dt>Medium</dt><dd>Reaction&ndash;diffusion on a {SIZE}&times;{SIZE} torus,<br>parameters graded across the field</dd>
      <dt>Regime</dt><dd>{current['regime']}</dd>
      <dt>Feed / kill</dt><dd>{current['feed']:.4f} &plusmn;{FEED_SPREAD:.4f} &middot; {current['kill']:.4f} &plusmn;{KILL_SPREAD:.4f}</dd>
      <dt>Palette</dt><dd>{current['palette']}</dd>
      <dt>Generation</dt><dd>{current['generation']:,} steps</dd>
      <dt>Plate</dt><dd>No. {current['run']}</dd>
    </dl>
  </div>

  {strip_block}

  <footer>
    The pattern is never repainted &mdash; only continued<br>
    Last visited {current['date_stamp']}
  </footer>
</main>
</body>
</html>"""


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # --- load or create state ---
    A = B = None
    generation = run = 0
    if os.path.exists(STATE_FILE):
        try:
            d = np.load(STATE_FILE)
            keys = set(d.files)
            if {"A", "B"} <= keys:
                a = d["A"].astype(np.float32)
                b = d["B"].astype(np.float32)
                # Reject legacy grids and any state that diverged
                if (a.shape == (SIZE, SIZE)
                        and np.isfinite(a).all() and np.isfinite(b).all()):
                    A, B = a, b
                    generation = int(d["generation"]) if "generation" in keys else 0
                    run = int(d["run"]) if "run" in keys else 0
        except (OSError, ValueError, KeyError):
            pass    # unreadable state; start fresh

    run += 1
    rng = np.random.default_rng(run * 7919 + 13)

    if A is None:
        A, B = fresh_medium(rng)

    # --- where are we on the journey ---
    regime, base_feed, base_kill = JOURNEY[(run // RUNS_PER_REGIME) % len(JOURNEY)]
    feed, kill = parameter_field(base_feed, base_kill, rng)
    palette_name, stops = PALETTES[run % len(PALETTES)]

    # A few fresh seeds each run keeps new structure emerging
    A, B = sow(A, B, rng, count=3)
    A, B = evolve(A, B, feed, kill, rng)
    generation += STEPS

    # --- save state (float16 keeps the repo light) ---
    np.savez_compressed(
        STATE_FILE,
        A=A.astype(np.float16), B=B.astype(np.float16),
        generation=generation, run=run,
    )

    # --- render ---
    plate = render(B, build_lut(stops), seed=run)
    plate.save(os.path.join(DOCS, "current.jpg"),
               quality=94, optimize=True, progressive=True, subsampling=0)

    archive_name = f"plate-{run:05d}.jpg"
    thumb_name = f"plate-{run:05d}-t.jpg"
    plate.resize((ARCHIVE_PX, ARCHIVE_PX), Image.LANCZOS).save(
        os.path.join(ARCHIVE_DIR, archive_name), quality=90, optimize=True, progressive=True
    )
    plate.resize((THUMB_PX, THUMB_PX), Image.LANCZOS).save(
        os.path.join(ARCHIVE_DIR, thumb_name), quality=82, optimize=True
    )

    now = datetime.now(timezone.utc)
    current = {
        "run": run,
        "generation": generation,
        "regime": regime,
        "feed": base_feed,
        "kill": base_kill,
        "palette": palette_name,
        "title": name_work(run, regime, palette_name),
        "file": archive_name,
        "thumb": thumb_name,
        "date_long": now.strftime("%B %-d, %Y"),
        "date_stamp": now.strftime("%Y-%m-%d %H:%M UTC"),
    }

    # --- manifest ---
    works = []
    if os.path.exists(MANIFEST_FILE):
        try:
            works = json.load(open(MANIFEST_FILE))
        except (json.JSONDecodeError, OSError):
            works = []
    works.insert(0, current)

    for stale in works[MAX_ARCHIVE:]:
        for key in ("file", "thumb"):
            path = os.path.join(ARCHIVE_DIR, stale.get(key, ""))
            if os.path.isfile(path):
                os.remove(path)
    works = works[:MAX_ARCHIVE]

    with open(MANIFEST_FILE, "w") as f:
        json.dump(works, f, indent=1)
    with open(os.path.join(DOCS, "index.html"), "w") as f:
        f.write(build_page(current, works[1:]))

    print(f'Plate {run}: "{current["title"]}" - {regime} / {palette_name} '
          f'- generation {generation:,} - B mean {B.mean():.4f}')


if __name__ == "__main__":
    main()

import numpy as np
from PIL import Image
import os
from datetime import datetime, timezone

# --- Settings (tweak later if you want) ---
SIZE = 256
STEPS = 120          # how many simulation steps per run
Du, Dv = 0.16, 0.08
F, K = 0.0545, 0.062

STATE_FILE = "state.npz"
IMAGE_FILE = "docs/current.png"
GEN_FILE = "generation.txt"

def laplacian(Z):
    return (
        -Z
        + np.roll(Z, 1, 0) + np.roll(Z, -1, 0)
        + np.roll(Z, 1, 1) + np.roll(Z, -1, 1)
    ) * 0.25

def step(A, B):
    La = laplacian(A)
    Lb = laplacian(B)
    reaction = A * B * B
    A += Du * La - reaction + F * (1 - A)
    B += Dv * Lb + reaction - (F + K) * B
    return A, B

def main():
    os.makedirs("docs", exist_ok=True)

    if os.path.exists(STATE_FILE):
        data = np.load(STATE_FILE)
        A, B = data["A"], data["B"]
        gen = int(open(GEN_FILE).read().strip()) if os.path.exists(GEN_FILE) else 0
    else:
        # First-time initialization
        A = np.ones((SIZE, SIZE), dtype=np.float32)
        B = np.zeros((SIZE, SIZE), dtype=np.float32)
        # Seed a small square in the center + a bit of noise
        r = SIZE // 8
        c = SIZE // 2
        A[c-r:c+r, c-r:c+r] = 0.50
        B[c-r:c+r, c-r:c+r] = 0.25
        A += np.random.uniform(-0.05, 0.05, (SIZE, SIZE)).astype(np.float32)
        B += np.random.uniform(-0.05, 0.05, (SIZE, SIZE)).astype(np.float32)
        A = np.clip(A, 0, 1)
        B = np.clip(B, 0, 1)
        gen = 0

    for _ in range(STEPS):
        A, B = step(A, B)

    gen += STEPS

    # Save state
    np.savez_compressed(STATE_FILE, A=A, B=B)
    with open(GEN_FILE, "w") as f:
        f.write(str(gen))

    # Render nice image (B channel gives the interesting patterns)
    img_data = (np.clip(B, 0, 1) * 255).astype(np.uint8)
    # Simple but good-looking colormap (dark purple → bright cyan/green)
    r = (img_data * 0.2).astype(np.uint8)
    g = img_data
    b = (255 - img_data * 0.6).astype(np.uint8)
    img = Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")
    img = img.resize((512, 512), Image.NEAREST)  # bigger for phone viewing
    img.save(IMAGE_FILE)

    # Simple viewer page
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="3600">
  <title>Math Garden</title>
  <style>
    body {{ margin:0; background:#111; color:#eee; font-family:system-ui; text-align:center; }}
    img {{ max-width:100%; height:auto; margin-top:1rem; border-radius:8px; }}
  </style>
</head>
<body>
  <h1>Pattern System</h1>
  <p>Generation {gen} · Updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
  <img src="current.png" alt="Current pattern">
  <p style="opacity:0.6;font-size:0.9rem">Gray-Scott reaction-diffusion · runs automatically</p>
</body>
</html>"""
    with open("docs/index.html", "w") as f:
        f.write(html)

    print(f"Done. Generation {gen}")

if __name__ == "__main__":
    main()

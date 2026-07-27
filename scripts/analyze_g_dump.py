"""Analyze dumped CA signal g=v_with-v_without to decide: is the subject-region style
signal REAL (coherent across inner iters) but crushed by the global rms-norm, or NOISE?
Also quantifies concentration and the global-norm crush. Reads agent_runs/{g_dump,xhat_dump}.
Usage: python scripts/analyze_g_dump.py gprobe_style16_kp4
"""
import sys, glob, os, math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PFX = sys.argv[1] if len(sys.argv) > 1 else "gprobe_style16_kp4"
GD, XD = "agent_runs/g_dump", "agent_runs/xhat_dump"
outers = sorted({int(f.split("_o")[1][:2]) for f in glob.glob(f"{GD}/{PFX}_o*_i*.pt")})
print("PFX", PFX, "| outers", outers)

def grid(n):
    h = int(round(math.sqrt(n)))
    while n % h: h -= 1
    return h, n // h

panels = []
for o in outers:
    files = sorted(glob.glob(f"{GD}/{PFX}_o{o:02d}_i*.pt"))
    G = torch.stack([torch.load(f)["g"][0].float() for f in files], 0)  # (I,N,C)
    I, N, C = G.shape
    H, W = grid(N)
    gmag = G.norm(dim=-1).mean(0)                                       # (N,) mean |g|/iters
    coh = torch.zeros(N)
    for a in range(I - 1):
        coh += torch.cosine_similarity(G[a], G[a + 1], dim=-1)
    coh = coh / max(I - 1, 1)                                           # (N,) iter-to-iter direction coherence
    # subject region from x0 dump: subject = where adding style barely moved x0
    xf = f"{XD}/{PFX}_o{o:02d}.pt"; subj = None
    if os.path.exists(xf):
        d = torch.load(xf); subj = (d["x0s"][0].float() - d["x00"][0].float()).norm(dim=-1)  # low=subject
    conc = (gmag.max() / gmag.median()).item()
    # discriminator via |g| quartiles
    q = torch.quantile(gmag, torch.tensor([0.25, 0.75]))
    lo, hi = gmag <= q[0], gmag >= q[1]
    coh_lo, coh_hi = coh[lo].mean().item(), coh[hi].mean().item()
    # crush: after global rms-norm the per-token step ∝ |g|/global_rms; subject share of total step
    subj_share = (gmag[lo].sum() / gmag.sum()).item()
    print(f"\n=== outer {o}: I={I} N={N} grid={H}x{W} ===")
    print(f"  |g| concentration  max/median = {conc:.1f}x   (median {gmag.median():.4f}, max {gmag.max():.4f})")
    print(f"  coherence: low-|g|(subject) {coh_lo:.3f}  vs  high-|g|(bg) {coh_hi:.3f}")
    print(f"    -> subject signal is {'REAL/coherent' if coh_lo > 0.3 else 'NOISE/incoherent'}")
    if subj is not None:
        qs = torch.quantile(subj, torch.tensor([0.25, 0.75]))
        sm, bm = subj <= qs[0], subj >= qs[1]
        print(f"  [x0-defined subject] coherence subject {coh[sm].mean():.3f} vs bg {coh[bm].mean():.3f}"
              f" | |g| subject {gmag[sm].mean():.4f} vs bg {gmag[bm].mean():.4f}")
    print(f"  crush: low-|g| quartile gets {100*subj_share:.1f}% of total step magnitude after global rms-norm")
    panels.append((o, H, W, gmag.reshape(H, W).numpy(), coh.reshape(H, W).numpy(),
                   (subj.reshape(H, W).numpy() if subj is not None else None)))

# visualize
rows = len(panels)
fig, ax = plt.subplots(rows, 3, figsize=(11, 3.4 * rows))
if rows == 1: ax = ax.reshape(1, 3)
for r, (o, H, W, gm, ch, sj) in enumerate(panels):
    im0 = ax[r, 0].imshow(gm, cmap="magma"); ax[r, 0].set_title(f"o{o} |g| (concentration)"); plt.colorbar(im0, ax=ax[r, 0], fraction=.046)
    im1 = ax[r, 1].imshow(ch, cmap="RdBu_r", vmin=-1, vmax=1); ax[r, 1].set_title(f"o{o} g coherence (iter-to-iter)"); plt.colorbar(im1, ax=ax[r, 1], fraction=.046)
    if sj is not None:
        im2 = ax[r, 2].imshow(sj, cmap="viridis"); ax[r, 2].set_title(f"o{o} |x0s-x00| (low=subject)"); plt.colorbar(im2, ax=ax[r, 2], fraction=.046)
    for c in range(3): ax[r, c].axis("off")
plt.tight_layout(); out = f"agent_runs/g_analysis_{PFX}.png"; plt.savefig(out, dpi=110); print("\nsaved", out)

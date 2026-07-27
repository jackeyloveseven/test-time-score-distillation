"""Reusable comparison montage builder for the style-loss iteration.
Usage:
  python scripts/montage.py OUT.png "row1:style16_kp4,row2:style04_kp2" \
      "baseline:skprobe" "puresw50:statsprobe_puresw50" ...
Each column arg is "colLabel:out_prefix"; cell = results/{prefix}_ca_only_{rowkey}_seed65.png
Missing cells are drawn as gray placeholders (so partial rounds still render).
"""
import sys, os
from PIL import Image, ImageDraw

out = sys.argv[1]
rows = [x.split(":", 1) for x in sys.argv[2].split(",")]      # [(label,rowkey)]
cols = [x.split(":", 1) for x in sys.argv[3:]]                # [(label,prefix)]
cell, pad, hdr, lab = 320, 6, 30, 90
W = lab + len(cols) * (cell + pad) + pad
H = hdr + len(rows) * (cell + pad) + pad
c = Image.new("RGB", (W, H), (245, 245, 247)); d = ImageDraw.Draw(c)
for j, (cl, _) in enumerate(cols):
    d.text((lab + pad + j * (cell + pad) + 6, 9), cl[:26], fill=(20, 20, 20))
for i, (rl, rk) in enumerate(rows):
    y = hdr + pad + i * (cell + pad)
    d.text((6, y + cell // 2), rl, fill=(20, 20, 20))
    for j, (cl, pfx) in enumerate(cols):
        p = f"results/{pfx}_ca_only_{rk}_seed65.png"
        x = lab + pad + j * (cell + pad)
        if os.path.exists(p):
            c.paste(Image.open(p).convert("RGB").resize((cell, cell)), (x, y))
        else:
            d.rectangle([x, y, x + cell, y + cell], fill=(60, 60, 66))
            d.text((x + cell // 2 - 20, y + cell // 2), "MISS", fill=(180, 180, 180))
c.save(out); print("saved", out, c.size)

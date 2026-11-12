"""Label audit: one row of four evenly spaced first frames per individual folder.

Used to check that every folder shows the species and the single animal its
name claims (the audit behind `visible_species` and `exclude` in
manifests/clips.csv). Writes results/audit_contact_sheet.jpg.

    python tools/contact_sheet.py
"""
import os

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "audit_contact_sheet.jpg")
W, H, PER_ROW = 240, 135, 4


def main():
    df = pd.read_csv(os.path.join(ROOT, "manifests", "clips.csv"), keep_default_na=False)
    rows = []
    for (folder, ind), g in df.groupby(["species", "individual"]):
        tiles = []
        for p in g.path.iloc[np.linspace(0, len(g) - 1, PER_ROW).astype(int)]:
            cap = cv2.VideoCapture(os.path.join(ROOT, p))
            ok, f = cap.read()
            cap.release()
            tiles.append(cv2.resize(f, (W, H)) if ok else np.zeros((H, W, 3), np.uint8))
        label = np.full((H, 150, 3), 255, np.uint8)
        lines = [f"{folder}/{ind}", f"{len(g)} clips"]
        if g.visible_species.iloc[0]:
            lines.append(f"-> {g.visible_species.iloc[0]}")
        if g.exclude.iloc[0]:
            lines.append("excluded")
        for k, text in enumerate(lines):
            cv2.putText(label, text, (6, 40 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows.append(np.hstack([label] + tiles))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"Wrote {OUT} ({len(rows)} individuals)")


if __name__ == "__main__":
    main()

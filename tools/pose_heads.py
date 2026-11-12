"""Run every ViTPose+ expert head on one frame, to identify the animal head."""
import os, sys, cv2, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from extract_features import load_manifests, read_frames, downscale, SEG_MAX_SIDE
from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
import extract_pose as E
cid = sys.argv[1]
df = load_manifests().set_index("clip_id")
z = np.load(os.path.join(ROOT, "output", "pose", *cid.split("/")) + ".npz")
fr = downscale(read_frames(os.path.join(ROOT, df.loc[cid].path)), SEG_MAX_SIDE)
j = len(z["frame"]) // 2; frame = fr[z["frame"][j]]; box = [float(v) for v in z["box"][j]]
P = VitPoseImageProcessor.from_pretrained(E.VITPOSE_ID)
M = VitPoseForPoseEstimation.from_pretrained(E.VITPOSE_ID).cuda().eval()
print(M.config.backbone_config.num_experts if hasattr(M.config.backbone_config, "num_experts") else "")
x, y, w, h = [int(v) for v in box]; s = 500 / w; tiles = []
for head in range(6):
    inp = P([frame], boxes=[[box]], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = M(**inp, dataset_index=torch.tensor([head]).cuda())
    r = P.post_process_pose_estimation(out, boxes=[[box]])[0][0]
    xy, sc = r["keypoints"].cpu().numpy(), r["scores"].cpu().numpy()
    print(head, xy.shape, np.round(sc, 2))
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)[y:y+h, x:x+w], None, fx=s, fy=s)
    for k, p in enumerate(xy):
        q = (int((p[0]-x)*s), int((p[1]-y)*s)); cv2.circle(img, q, 3, (0, 0, 255), -1)
        cv2.putText(img, str(k), (q[0]+4, q[1]-4), 0, 0.45, (0, 0, 0), 3); cv2.putText(img, str(k), (q[0]+4, q[1]-4), 0, 0.45, (0, 255, 255), 1)
    cv2.putText(img, f"head {head}", (5, 18), 0, 0.6, (255, 255, 255), 2); tiles.append(img)
cv2.imwrite(os.path.join(ROOT, "results", "pose_check", cid.replace("/", "_") + "_heads.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))

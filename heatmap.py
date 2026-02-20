import json, numpy as np
from pathlib import Path
from matplotlib import cm

video_id = "169115044301335945-480-000-500-000-front"
frames_dir = Path("data/waymo/train/frames") / video_id
labels_json = Path("data/waymo/train/labels.json")  # VID 标注
out_dir = Path(f"vis_heatmap_token_{video_id}_colored")
patch_size = 16
save_per_class = True  # 如不需按类，设 False

out_dir.mkdir(exist_ok=True)

# 读标注，建立 basename 对齐
data = json.loads(labels_json.read_text())
meta_by_bn = {}
id2bn = {}
for im in data["images"]:
    bn = Path(im["file_name"]).name  # 可能是 videoid_000000.jpg
    meta_by_bn[bn] = (im["width"], im["height"], im["id"])
    id2bn[im["id"]] = bn

cat_ids = sorted({ann["category_id"] for ann in data["annotations"]})
cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

grid_sum = None
grid_per_class = None

frames = sorted(frames_dir.glob("*.jpg"))
print(f"frames in folder: {len(frames)}")
for img_path in frames:
    bn = img_path.name  # 实际文件名 000000.jpg
    # 标注可能带前缀，做两种键
    candidates = [bn, f"{video_id}_{bn}"]
    meta = None
    for k in candidates:
        if k in meta_by_bn:
            meta = meta_by_bn[k]
            bn_key = k
            break
    if meta is None:
        continue
    w, h, img_id = meta
    if grid_sum is None:
        gw, gh = (w + patch_size - 1)//patch_size, (h + patch_size - 1)//patch_size
        grid_sum = np.zeros((gh, gw), dtype=np.float32)
        grid_per_class = np.zeros((len(cat_ids), gh, gw), dtype=np.float32)

    # 累加框
    for ann in data["annotations"]:
        if ann["image_id"] != img_id:
            continue
        x, y, bw, bh = ann["bbox"]
        x2, y2 = x + bw, y + bh
        gx1, gx2 = int(x // patch_size), int(np.ceil(x2 / patch_size))
        gy1, gy2 = int(y // patch_size), int(np.ceil(y2 / patch_size))
        gx1, gy1 = max(gx1, 0), max(gy1, 0)
        gx2, gy2 = min(gx2, gw), min(gy2, gh)
        if gx1 < gx2 and gy1 < gy2:
            grid_sum[gy1:gy2, gx1:gx2] += 1.0
            cls_idx = cat_to_idx.get(ann["category_id"])
            if cls_idx is not None:
                grid_per_class[cls_idx, gy1:gy2, gx1:gx2] += 1.0

def save_color(grid, path):
    norm = grid / grid.max() if grid.max() > 0 else grid
    rgba = (cm.get_cmap("viridis")(norm) * 255).astype(np.uint8)  # H×W×4
    from PIL import Image
    Image.fromarray(rgba).save(path)

if grid_sum is not None:
    save_color(grid_sum, out_dir / "heat_all.png")
    if save_per_class:
        for cid, idx in cat_to_idx.items():
            save_color(grid_per_class[idx], out_dir / f"heat_cls_{cid}.png")
    print("saved to", out_dir)
else:
    print("No matched frames/annotations; nothing saved.")
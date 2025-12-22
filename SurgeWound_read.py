import os, base64
from datasets import load_dataset

out_dir = "SurgWound_images"
os.makedirs(out_dir, exist_ok=True)

# streaming avoids loading huge JSON fully into RAM
ds = load_dataset("xuxuxuxuxu/SurgWound", split="train", streaming=True)

seen = set()
for row in ds:
    name = row["image_name"]           # e.g., "76.jpg"
    if name in seen:
        continue

    b64 = row["image"]
    # if it ever comes as a data-URI, strip the header
    if isinstance(b64, str) and b64.startswith("data:image"):
        b64 = b64.split(",", 1)[1]

    img_bytes = base64.b64decode(b64)
    with open(os.path.join(out_dir, name), "wb") as f:
        f.write(img_bytes)

    seen.add(name)

print("Saved unique images:", len(seen))

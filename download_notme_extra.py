
import os
import time
import requests
from PIL import Image
from io import BytesIO
from ddgs import DDGS

SAVE_FOLDER = "dataset/NOT_ME"
os.makedirs(SAVE_FOLDER, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Diverse queries so the model learns all kinds of "NOT ME" faces
QUERIES = [
    "Brad Pitt face photo",
    "random man portrait photo",
    "woman face portrait photo",
    "old man face photo",
    "young woman face photo",
    "Leonardo DiCaprio face",
    "random person face photo",
    "asian man face photo",
    "african man portrait photo",
    "middle aged woman face photo",
]

PER_QUERY = 20  # 20 images per query = 200 extra images total

print("=" * 50)
print("  Extra NOT_ME Image Downloader")
print(f"  {len(QUERIES)} queries x {PER_QUERY} images = {len(QUERIES)*PER_QUERY} target")
print("=" * 50)

# Count existing files to avoid overwriting
existing = len([f for f in os.listdir(SAVE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
saved = 0

for query in QUERIES:
    print(f"\n🔍 Searching: '{query}'")
    urls = []

    with DDGS() as ddgs:
        results = ddgs.images(query, region="wt-wt", safesearch="off", max_results=30)
        for r in results:
            if r.get("image"):
                urls.append(r["image"])

    query_saved = 0
    for url in urls:
        if query_saved >= PER_QUERY:
            break
        try:
            res = requests.get(url, headers=HEADERS, timeout=8)
            img = Image.open(BytesIO(res.content)).convert("RGB")
            filename = os.path.join(SAVE_FOLDER, f"notme_{existing + saved + 1:04d}.jpg")
            img.save(filename, "JPEG")
            saved += 1
            query_saved += 1
            time.sleep(0.2)
        except Exception:
            continue

    print(f"  ✅ Saved {query_saved} images")

print(f"\n{'='*50}")
print(f"  Total new images added : {saved}")
print(f"  Total in NOT_ME folder : {existing + saved}")
print(f"\n  Now run: python train_model.py")
print("=" * 50)

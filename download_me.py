"""
download_me.py  (v2)
────────────────────
Downloads 100 images of Robert Pattinson into dataset/ME/

Usage:
    pip install requests duckduckgo-search Pillow
    python download_me.py
"""

import os
import time
import requests
from PIL import Image
from io import BytesIO
from duckduckgo_search import DDGS

# ─────────────────────────────────────────────
SAVE_FOLDER = "dataset/ME"
TARGET      = 100
QUERY       = "Robert Pattinson face photo"
# ─────────────────────────────────────────────

os.makedirs(SAVE_FOLDER, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

print("=" * 50)
print("  Robert Pattinson Image Downloader v2")
print(f"  Target : {TARGET} images → {SAVE_FOLDER}/")
print("=" * 50)

# ── Step 1: collect image URLs via DuckDuckGo ─
print(f"\n🔍 Searching for images...\n")

urls = []
with DDGS() as ddgs:
    results = ddgs.images(
        keywords=QUERY,
        region="wt-wt",
        safesearch="off",
        max_results=150
    )
    for r in results:
        if r.get("image"):
            urls.append(r["image"])

print(f"  Found {len(urls)} URLs\n  Downloading...\n")

# ── Step 2: download & verify each image ──────
saved   = 0
skipped = 0

for url in urls:
    if saved >= TARGET:
        break

    try:
        res = requests.get(url, headers=HEADERS, timeout=8)

        # Verify it's a valid image using Pillow
        img = Image.open(BytesIO(res.content))
        img = img.convert("RGB")  # normalize format

        filename = os.path.join(SAVE_FOLDER, f"me_{saved + 1:03d}.jpg")
        img.save(filename, "JPEG")

        saved += 1
        bar = "█" * (saved * 30 // TARGET) + "░" * (30 - saved * 30 // TARGET)
        print(f"\r  [{bar}] {saved}/{TARGET}", end="", flush=True)
        time.sleep(0.2)

    except Exception:
        skipped += 1
        continue

# ── Done ──────────────────────────────────────
print(f"\n\n{'=' * 50}")
print(f"  ✅ Saved   : {saved} images")
print(f"  ⏭  Skipped : {skipped} (bad/unreadable links)")
print(f"  📁 Folder  : {SAVE_FOLDER}/")

if saved < TARGET:
    print(f"\n  ⚠️  Only got {saved}/{TARGET}. Run the script again to get more.")
else:
    print(f"\n  🎉 Done! Your ME folder is ready.")
print("=" * 50)

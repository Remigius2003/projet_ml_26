import torch
import numpy as np
from PIL import Image
import pickle

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def preprocess(image_path):
    img_pil = Image.open(image_path).convert('RGB')
    img = img_pil.resize((224, 224), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.transpose((2, 0, 1))

    mu = torch.Tensor(IMAGENET_MEAN).view(3, 1, 1)
    sigma = torch.Tensor(IMAGENET_STD).view(3, 1, 1)
    x = (torch.Tensor(img) - mu) / sigma

    return img_pil, x.unsqueeze(0)

def preprocess_tensor_only(image_path):
    _, x = preprocess(image_path)
    return x.squeeze(0)


def stable_softmax(logits):
    shifted = logits - np.max(logits)
    e = np.exp(shifted)
    return e / np.sum(e)

def load_imagenet_classes():
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, "imagenet_classes.txt")
        with open("imagenet_classes.txt") as f:
            classes = [line.strip() for line in f.readlines()]
        pickle.dump(classes, open('imagenet_classes.pkl', 'wb'))
        return classes
    except Exception:
        return pickle.load(open('imagenet_classes.pkl', 'rb'))

def load_vgg16():
    import torchvision
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def download_15scene(dest_path="data/15-Scene"):
    """
    Download the 15-Scene dataset and extract it.
    COMMENT: ~120MB zip, 4,485 grayscale images, 15 scene categories (CC BY 4.0).
    Skips download if dest_path already has >=15 class folders.
    """
    import os, zipfile, shutil, urllib.request
 
    # COMMENT: Skip if already downloaded
    if os.path.isdir(dest_path):
        subdirs = [d for d in os.listdir(dest_path) if os.path.isdir(os.path.join(dest_path, d))]
        if len(subdirs) >= 15:
            print(f"  Dataset found at '{dest_path}' ({len(subdirs)} classes).")
            return dest_path
 
    os.makedirs(dest_path, exist_ok=True)
    parent = os.path.dirname(dest_path) or "."
    zip_path = os.path.join(parent, "15scene.zip")
 
    # COMMENT: Figshare blocks urllib's default User-Agent ("Python-urllib/3.x").
    # We must send a browser-like User-Agent or the server returns 403/empty response.
    # Multiple URLs are tried: Figshare article downloads + Figshare file API.
    urls = [
        "https://figshare.com/ndownloader/articles/7007177/versions/1",
        "https://ndownloader.figshare.com/articles/7007177/versions/1",
        "https://figshare.com/ndownloader/articles/12103434/versions/1",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
 
    downloaded = False
    for url in urls:
        try:
            print(f"  Downloading 15-Scene dataset...")
            print(f"  URL: {url}")
            # COMMENT: Build a Request with browser-like headers so Figshare doesn't block us
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                total = int(response.headers.get('Content-Length', 0))
                chunk_size = 1024 * 256  # 256 KB chunks
                downloaded_bytes = 0
                with open(zip_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        if total > 0:
                            pct = downloaded_bytes * 100 // total
                            mb = downloaded_bytes / 1e6
                            print(f"\r  {mb:.1f} MB / {total/1e6:.1f} MB ({pct}%)", end="", flush=True)
                print()  # newline after progress
 
            if os.path.getsize(zip_path) > 1_000_000:  # COMMENT: sanity check > 1MB
                downloaded = True
                break
            else:
                print(f"  File too small ({os.path.getsize(zip_path)} bytes), trying next URL...")
        except Exception as e:
            print(f"  Failed: {e}")
 
    if not downloaded:
        print(f"\n  Auto-download failed. Please download manually:")
        print(f"  1. Go to: https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177")
        print(f"  2. Click 'Download all' (or download the zip file)")
        print(f"  3. Extract so class folders are directly under: {dest_path}/")
        print(f"     e.g. {dest_path}/bedroom/, {dest_path}/coast/, ...")
        raise FileNotFoundError(f"Could not download 15-Scene dataset to '{dest_path}'")
 
    # COMMENT: Extract — handle nested zips (Figshare wraps files in an outer zip)
    print(f"  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_path)
 
    # COMMENT: Figshare often nests the real zip inside the article zip
    for f in os.listdir(dest_path):
        fpath = os.path.join(dest_path, f)
        if f.endswith('.zip') and os.path.isfile(fpath):
            print(f"  Extracting inner zip: {f}")
            with zipfile.ZipFile(fpath, 'r') as z2:
                z2.extractall(dest_path)
            os.remove(fpath)
 
    # COMMENT: Flatten if there's a single wrapper folder (e.g. dest/15-Scene/bedroom → dest/bedroom)
    entries = [e for e in os.listdir(dest_path) if os.path.isdir(os.path.join(dest_path, e))]
    if len(entries) == 1:
        inner = os.path.join(dest_path, entries[0])
        for item in os.listdir(inner):
            shutil.move(os.path.join(inner, item), os.path.join(dest_path, item))
        os.rmdir(inner)
 
    if os.path.exists(zip_path):
        os.remove(zip_path)
 
    subdirs = sorted([d for d in os.listdir(dest_path) if os.path.isdir(os.path.join(dest_path, d))])
    print(f"  Done! {len(subdirs)} classes: {', '.join(subdirs[:5])}...")
    return dest_path
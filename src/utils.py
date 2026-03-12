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
    import shutil, urllib.request, zipfile, subprocess, sys, os

    if os.path.isdir(dest_path) and sum(
        os.path.isdir(os.path.join(dest_path, d)) for d in os.listdir(dest_path)
    ) >= 15:
        print(f"  Dataset found ({len(os.listdir(dest_path))} classes).")
        return dest_path

    os.makedirs(dest_path, exist_ok=True)
    parent = os.path.dirname(dest_path) or "."
    zip_path = os.path.join(parent, "15scene.zip")
    tmp = os.path.join(parent, "_tmp15")

    for url in [
        "https://figshare.com/ndownloader/articles/7007177/versions/1",
        "https://ndownloader.figshare.com/articles/7007177/versions/1",
    ]:
        try:
            print("  Downloading...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(zip_path, "wb") as f:
                shutil.copyfileobj(r, f)
            if os.path.getsize(zip_path) > 1_000_000:
                break
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        raise FileNotFoundError("Download failed from all URLs")

    os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z: z.extractall(tmp)
    os.remove(zip_path)

    rar = next(
        (os.path.join(r, f) for r, _, fs in os.walk(tmp) for f in fs if f.endswith(".rar")),
        None,
    )
    if not rar:
        raise FileNotFoundError("No .rar found inside the .zip")

    print("  Extracting .rar...")
    commands = [
        ["unrar", "x", "-y", "-o+", rar, tmp + os.sep],
        ["7z", "x", f"-o{tmp}", "-y", rar],
        [r"C:\Program Files\7-Zip\7z.exe", "x", f"-o{tmp}", "-y", rar],
        [r"C:\Program Files (x86)\7-Zip\7z.exe", "x", f"-o{tmp}", "-y", rar],
    ]

    for cmd in commands:
        try: subprocess.run(cmd, check=True, capture_output=True); break
        except (FileNotFoundError, subprocess.CalledProcessError): continue
    else:
        shutil.rmtree(tmp, ignore_errors=True)
        print("\n No .rar extractor found. Install one or manually download 15-Scene at :", dest_path)
        raise RuntimeError("No .rar extractor found, see instructions above")

    for root, dirs, _ in os.walk(tmp):
        if len(dirs) >= 15:
            for d in dirs: shutil.move(os.path.join(root, d), os.path.join(dest_path, d))
            break

    shutil.rmtree(tmp, ignore_errors=True)
    n = sum(os.path.isdir(os.path.join(dest_path, d)) for d in os.listdir(dest_path))
    if n < 15: raise RuntimeError(f"Expected 15 classes, got {n}")
    print(f"  Done! {n} classes.")
    
    return dest_path
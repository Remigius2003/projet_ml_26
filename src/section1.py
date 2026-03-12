import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import preprocess, stable_softmax, load_imagenet_classes, load_vgg16

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    conv_p = sum(p.numel() for p in model.features.parameters())
    fc_p = sum(p.numel() for p in model.classifier.parameters())
    print(f"  Total: {total:,} | Conv: {conv_p:,} ({100*conv_p/total:.1f}%) | FC: {fc_p:,} ({100*fc_p/total:.1f}%)")
    return total

def classify(image_path, model, classes, show=True):
    img_pil, x = preprocess(image_path)
    if show:
        plt.figure(figsize=(4, 4)); plt.imshow(img_pil); plt.axis('off')
        plt.title(Path(image_path).name); plt.show()

    with torch.no_grad():
        logits = model(x).numpy()[0]

    probs = stable_softmax(logits)
    top5 = np.argsort(probs)[-5:][::-1]
    print(f"  Top-5 for '{Path(image_path).name}':")
    for rank, idx in enumerate(top5, 1):
        print(f"    {rank}. {classes[idx]:<30s} {probs[idx]:.2%}")
    return probs

def get_activation_maps(model, image_path, layer_index=0):
    _, x = preprocess(image_path)
    captured = {}

    hook = model.features[layer_index].register_forward_hook(
        lambda _1, _2, out: captured.update({'act': out.detach()})
    )
    with torch.no_grad(): model(x)
    hook.remove()

    return captured['act'].squeeze(0).numpy()

def show_activation_maps(activations, top_k=9, title="Activation Maps"):
    strength = activations.reshape(activations.shape[0], -1).mean(axis=1)
    top_idx = np.argsort(strength)[-top_k:][::-1]

    cols = min(top_k, 3)
    rows = (top_k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for i, fidx in enumerate(top_idx):
        ax = axes.flat[i]
        a = activations[fidx]
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        ax.imshow(a, cmap='hot'); ax.set_title(f'Filter {fidx}', fontsize=9); ax.axis('off')
    for i in range(top_k, rows*cols):
        axes.flat[i].axis('off')
    plt.tight_layout(); plt.show()

def show_all_maps(activations, title="All Activation Maps"):
    n = activations.shape[0]
    fig, axes = plt.subplots(8, 8, figsize=(14, 14))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i, ax in enumerate(axes.flat):
        if i < n:
            a = activations[i]
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            ax.imshow(a, cmap='hot'); ax.set_title(f'{i}', fontsize=6)
        ax.axis('off')
    plt.tight_layout(); plt.show()
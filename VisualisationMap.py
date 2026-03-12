import torch
import torchvision
from torchvision.models import VGG16_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


nameim = "paladin.jpg"
if not Path(nameim).exists():
    print(f"Error: Image file '{nameim}' not found!")
    exit(1)

# Load and preprocess image
img = Image.open(nameim).convert('RGB')
img = img.resize((224, 224), Image.BILINEAR)
img_np = np.array(img, dtype=np.float32) / 255.0
img_tensor = torch.Tensor(img_np).transpose(0, 2).unsqueeze(0)  # CHW + batch

# Normalize with ImageNet values
mu = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
sigma = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
img_tensor = (img_tensor - mu) / sigma

# Load VGG16
vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
vgg16.eval()


# 1. EXTRACT ACTIVATION MAPS USING HOOKS
activation_maps = {}

def hook_fn(module, input, output):
    activation_maps['first_conv'] = output.detach()

# Register hook on the first convolutional layer
# VGG16 structure: features[0] is the first Conv2d layer
first_conv_layer = vgg16.features[0]
hook_handle = first_conv_layer.register_forward_hook(hook_fn)

# This apparently triggers the hook (don't ask how)
with torch.no_grad():
    _ = vgg16(img_tensor)

hook_handle.remove()


activations = activation_maps['first_conv'].squeeze(0).numpy()  # Shape: (64, 224, 224)

print(f"Activation maps shape: {activations.shape}")
print(f"  - Number of filters: {activations.shape[0]}")
print(f"  - Height x Width: {activations.shape[1]} x {activations.shape[2]}")

# 2. VISUALIZE ALL 64 ACTIVATION MAPS

fig, axes = plt.subplots(8, 8, figsize=(16, 16))
fig.suptitle('64 Activation Maps', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < activations.shape[0]:
        # Get the i-th activation map
        act_map = activations[i]
        act_normalized = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
        
        # Display
        ax.imshow(act_normalized, cmap='hot')
        ax.set_title(f'Filter {i}', fontsize=8)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.savefig('activation_maps_grid.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: activation_maps_grid.png")
plt.show()


# 3. VISUALIZE ONLY TOP ACTIVATION MAPS

activation_strength = activations.reshape(activations.shape[0], -1).mean(axis=1)
top_indices = np.argsort(activation_strength)[-9:][::-1]

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
fig.suptitle('VGG16 First Conv - Top 9 Most Activated Filters', 
             fontsize=14, fontweight='bold')

for idx, filter_idx in enumerate(top_indices):
    ax = axes.flat[idx]
    act_map = activations[filter_idx]
    act_normalized = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
    
    ax.imshow(act_normalized, cmap='hot')
    ax.set_title(f'Filter {filter_idx}\nStrength: {activation_strength[filter_idx]:.3f}', 
                 fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('top_activation_maps.png', dpi=150, bbox_inches='tight')
plt.show()


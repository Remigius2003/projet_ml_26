import torch
import torch.nn as nn
import torchvision
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Model ─────────────────────────────────────────────────────────────────────

class VGG16relu7(nn.Module):
    def __init__(self, vgg16):
        super(VGG16relu7, self).__init__()
        # Copy the entire convolutional part
        self.features = nn.Sequential(*list(vgg16.features.children()))
        # Keep a piece of the classifier: -2 to stop at relu7
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ── Helpers ──────────────────────────────────────────────────────────────────

def stable_softmax(logits):
    logits_shifted = logits - np.max(logits)
    exp_logits = np.exp(logits_shifted)
    return exp_logits / np.sum(exp_logits)


def load_imagenet_classes():
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, "imagenet_classes.txt")
        with open("imagenet_classes.txt") as f:
            classes = [line.strip() for line in f.readlines()]
        pickle.dump(classes, open('imagenet_classes.pkl', 'wb'))
        return classes
    except Exception as e:
        print(f"Warning: Could not download classes ({e}). Using fallback.")
        try:
            return pickle.load(open('imagenet_classes.pkl', 'rb'))
        except:
            print("Error: Could not load ImageNet classes. Exiting.")
            exit(1)


def normalize(img, mu, sigma):
    mu = mu.view(3, 1, 1)
    sigma = sigma.view(3, 1, 1)
    return (torch.Tensor(img) - mu) / sigma


def get_prediction(logits, imagenet_classes):
    predicted_idx = np.argmax(logits[0])
    softmax_scores = stable_softmax(logits[0])
    confidence = softmax_scores[predicted_idx]
    predicted_class = imagenet_classes[predicted_idx]
    return predicted_class, confidence, softmax_scores


# ── Main ──────────────────────────────────────────────────────────────────────

nameim = "images/paladin.jpg"
if not Path(nameim).exists():
    print(f"Error: Image file '{nameim}' not found!")
    exit(1)

# Load and display image
img = Image.open(nameim)
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis('off')
plt.title("Input Image")
plt.show(block=False)
plt.pause(0.5)

# Loading ImageNet classes
imagenet_classes = load_imagenet_classes()

# Normalization
img = img.convert('RGB')
img = img.resize((224, 224), Image.BILINEAR)
img = np.array(img, dtype=np.float32) / 255.0
img = img.transpose((2, 0, 1))

# ImageNet mean/std
mu = torch.Tensor([0.485, 0.456, 0.406])
sigma = torch.Tensor([0.229, 0.224, 0.225])

# Expand mu & sigma to match image size + compute normalized image
x = normalize(img, mu, sigma)

# Loading pre-trained VGG
vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
vgg16.eval()

# Forward pass on VGG
x = x.unsqueeze(0)
with torch.no_grad():
    y = vgg16(x)
y = y.numpy()

# Get prediction (i.e., ImageNet class label)
predicted_class, confidence, softmax_scores = get_prediction(y, imagenet_classes)

print(f" ")
print(f"Predicted class : {predicted_class}")
print(f"Confidence      : {confidence:.2%}")
print(f" ")

top_5_indices = np.argsort(softmax_scores)[-5:][::-1]
print("Top 5 predictions:")
for rank, idx in enumerate(top_5_indices, 1):
    print(f"  {rank}. {imagenet_classes[idx]:<30} {softmax_scores[idx]:.2%}")

# Feature extraction with VGG16relu7
vgg16relu7 = VGG16relu7(vgg16)
vgg16relu7.eval()

with torch.no_grad():
    y = vgg16relu7(x)
y = y.numpy()

print(f" ")
print(f"relu7 feature vector shape : {y.shape}")  # Expected: (1, 4096)
print(f"relu7 features (first 10)  : {y[0, :10]}")

# ── SVM Classifier ────────────────────────────────────────────────────────────

from sklearn.svm import LinearSVC

# Train multi-class SVM (one-versus-all) on deep features
svm = LinearSVC(C=1)
svm.fit(Xtrain, ytrain)

# Evaluate on test set
accuracy = svm.score(Xtest, ytest)
print(f" ")
print(f"SVM Classification Accuracy : {accuracy:.2%}")
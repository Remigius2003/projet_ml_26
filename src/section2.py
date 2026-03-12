import torch
import torch.nn as nn
import numpy as np
import os, glob
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize
from utils import download_15scene, preprocess_tensor_only

class VGG16relu7(nn.Module):
    def __init__(self, vgg16):
        super().__init__()
        self.features = nn.Sequential(*list(vgg16.features.children()))
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_15scene(path, n_train=100, seed=42):
    download_15scene(path)
    
    np.random.seed(seed)
    class_names = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    tr_paths, tr_labels, te_paths, te_labels = [], [], [], []

    for idx, c in enumerate(class_names):
        imgs = sorted(glob.glob(os.path.join(path, c, '*.jpg')))
        np.random.shuffle(imgs)
        tr_paths.extend(imgs[:n_train]);  tr_labels.extend([idx]*min(n_train, len(imgs)))
        te_paths.extend(imgs[n_train:]);  te_labels.extend([idx]*len(imgs[n_train:]))
        print(f"  {c:20s} train:{min(n_train,len(imgs)):3d}  test:{len(imgs[n_train:]):3d}")

    print(f"  Total: train:{len(tr_paths)} test:{len(te_paths)}")
    return tr_paths, np.array(tr_labels), te_paths, np.array(te_labels), class_names

def extract_features(paths, model, device='cpu', batch_size=32):
    model.eval(); model = model.to(device)
    all_feats = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([preprocess_tensor_only(p) for p in paths[i:i+batch_size]]).to(device)
        with torch.no_grad():
            all_feats.append(model(batch).cpu().numpy())
        if (i // batch_size + 1) % 10 == 0:
            print(f"    {min(i+batch_size, len(paths))}/{len(paths)}")
    return np.concatenate(all_feats, axis=0)

def extract_and_normalize(paths, model, device='cpu'):
    X = extract_features(paths, model, device)
    return normalize(X, norm='l2', axis=1)

def train_and_eval(X_train, y_train, X_test, y_test, C=1.0, class_names=None):
    svm = LinearSVC(C=C, max_iter=10000)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  C={C} -> Train:{svm.score(X_train,y_train):.2%}  Test:{acc:.2%}")
    if class_names is not None:
        print(classification_report(y_test, y_pred, target_names=class_names))
    return svm, acc, y_pred

def plot_confusion(y_true, y_pred, names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    _, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(cm, cmap='Blues')
    ax.set(xticks=range(len(names)), yticks=range(len(names)),
           xticklabels=names, yticklabels=names,
           xlabel='Predicted', ylabel='True', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha='center', va='center', fontsize=6,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.tight_layout(); plt.show()
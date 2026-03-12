import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from utils import preprocess_tensor_only
from section2 import extract_features

class VGG16Extractor(nn.Module):
    def __init__(self, layer='relu7'):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        cuts = {'pool5': None, 'relu6': 2, 'relu7': 5}
        n = cuts[layer]
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:n]) if n else None

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x) if self.classifier else x


def experiment_layers(tr_p, y_tr, te_p, y_te, device='cpu', C=1.0):
    results = {}
    for layer in ['pool5', 'relu6', 'relu7']:
        ext = VGG16Extractor(layer).eval().to(device)
        X_tr = normalize(extract_features(tr_p, ext, device), norm='l2')
        X_te = normalize(extract_features(te_p, ext, device), norm='l2')
        svm = LinearSVC(C=C, max_iter=10000); svm.fit(X_tr, y_tr)
        acc = svm.score(X_te, y_te)
        results[layer] = {'acc': acc, 'dim': X_tr.shape[1]}
        print(f"  {layer:6s}  dim={X_tr.shape[1]:>6d}  acc={acc:.2%}")
    return results

def experiment_C(X_tr, y_tr, X_te, y_te, C_vals=None):
    if C_vals is None:
        C_vals = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
    
    tr_accs, te_accs = [], []
    for C in C_vals:
        svm = LinearSVC(C=C, max_iter=10000); svm.fit(X_tr, y_tr)
        tr_a, te_a = svm.score(X_tr, y_tr), svm.score(X_te, y_te)
        tr_accs.append(tr_a); te_accs.append(te_a)
        print(f"  C={C:>8.3f}  Train:{tr_a:.2%}  Test:{te_a:.2%}")

    _, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(C_vals, [a*100 for a in tr_accs], 'o-', label='Train')
    ax.semilogx(C_vals, [a*100 for a in te_accs], 's-', label='Test')
    ax.set(xlabel='C', ylabel='Accuracy (%)', title='C Tuning'); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()

    best_C = C_vals[np.argmax(te_accs)]
    print(f"  Best C = {best_C} → {max(te_accs):.2%}")

    return best_C

class ResNet50Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        r.fc = nn.Identity()
        self.model = r

    def forward(self, x):
        return self.model(x)


def experiment_resnet(tr_p, y_tr, te_p, y_te, device='cpu', C=1.0):
    ext = ResNet50Extractor().eval().to(device)
    X_tr = normalize(extract_features(tr_p, ext, device), norm='l2')
    X_te = normalize(extract_features(te_p, ext, device), norm='l2')
    svm = LinearSVC(C=C, max_iter=10000); svm.fit(X_tr, y_tr)

    acc = svm.score(X_te, y_te)
    print(f"  ResNet50 (dim={X_tr.shape[1]}) -> {acc:.2%}")

    return acc

class SceneDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return preprocess_tensor_only(self.paths[i]), self.labels[i]


def experiment_finetune(tr_p, y_tr, te_p, y_te, n_classes=15, epochs=20, device='cpu'):
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    for p in model.parameters(): p.requires_grad = False
    model.classifier[6] = nn.Linear(4096, n_classes)
    model = model.to(device)

    loader_tr = DataLoader(SceneDataset(tr_p, y_tr), batch_size=32, shuffle=True)
    loader_te = DataLoader(SceneDataset(te_p, y_te), batch_size=32)
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    hist = {'loss': [], 'tr_acc': [], 'te_acc': []}
    for ep in range(epochs):
        model.train(); loss_sum = correct = total = 0
        for imgs, labs in loader_tr:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            out = model(imgs); loss = crit(out, labs)
            loss.backward(); opt.step()
            loss_sum += loss.item(); correct += (out.argmax(1)==labs).sum().item(); total += labs.size(0)
        hist['loss'].append(loss_sum/len(loader_tr)); hist['tr_acc'].append(correct/total)

        model.eval(); correct = total = 0
        with torch.no_grad():
            for imgs, labs in loader_te:
                imgs, labs = imgs.to(device), labs.to(device)
                correct += (model(imgs).argmax(1)==labs).sum().item(); total += labs.size(0)
        hist['te_acc'].append(correct/total)

        if (ep+1) % 5 == 0 or ep == 0:
            print(f"  Ep {ep+1:2d}/{epochs} Loss:{hist['loss'][-1]:.3f} "
                  f"Train:{hist['tr_acc'][-1]:.2%} Test:{hist['te_acc'][-1]:.2%}")

    _, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(hist['loss']); a1.set(title='Loss', xlabel='Epoch'); a1.grid(alpha=.3)
    a2.plot([a*100 for a in hist['tr_acc']], label='Train')
    a2.plot([a*100 for a in hist['te_acc']], label='Test')
    a2.set(title='Accuracy', xlabel='Epoch', ylabel='%'); a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    return hist

def experiment_pca(X_tr, y_tr, X_te, y_te, dims=None, C=1.0):
    if dims is None: dims = [64, 128, 256, 512, 1024, 2048]
    max_comp = min(X_tr.shape[0], X_tr.shape[1])
    dims = [d for d in dims if d <= max_comp]
    results = []
    for d in dims:
        pca = PCA(n_components=d)
        Xtr = normalize(pca.fit_transform(X_tr), norm='l2')
        Xte = normalize(pca.transform(X_te), norm='l2')
        svm = LinearSVC(C=C, max_iter=10000); svm.fit(Xtr, y_tr)
        acc = svm.score(Xte, y_te); var = pca.explained_variance_ratio_.sum()
        results.append({'dim': d, 'acc': acc, 'var': var})
        print(f"  PCA({d:>5d}) var:{var:.1%} acc:{acc:.2%}")
    return results
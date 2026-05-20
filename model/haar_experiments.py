"""
Module for model training and simulation.
"""

import os
import math
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available()
    else "cpu"
)
print(f"[haar_experiments] device = {DEVICE}")

class HaarCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block2(self.block1(x)))

def _affine_theta(angle_rad: float, batch_size: int, device: str) -> torch.Tensor:
    """Build (B, 2, 3) rotation matrix for a scalar angle."""
    a    = torch.tensor(angle_rad, dtype=torch.float32, device=device)
    c, s = a.cos().view(1), a.sin().view(1) 
    z    = torch.zeros(1, device=device)
    row0 = torch.stack([c, -s, z], dim=-1)   
    row1 = torch.stack([s,  c, z], dim=-1)
    theta = torch.stack([row0, row1], dim=-2) 
    return theta.expand(batch_size, -1, -1)

@torch.no_grad()
def rotate_batch(x: torch.Tensor, angle_rad: float) -> torch.Tensor:
    theta = _affine_theta(angle_rad, x.shape[0], x.device)
    grid  = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, mode="bilinear", padding_mode="zeros")

def sample_haar_angles(m: int, group: str, k: int = 4) -> list:
    """
    Sample m angles (radians) i.i.d. from the Haar measure of the group.
    """
    if group == "so2":
        return (torch.rand(m) * 2 * math.pi).tolist()
    elif group == "ck":
        idxs = torch.randint(0, k, (m,))
        return (idxs.float() * (2 * math.pi / k)).tolist()
    else:
        raise ValueError(f"Unknown group '{group}'. Use 'so2' or 'ck'.")

def get_loaders(
    n_train: int = 8_000,
    n_test:  int = 2_000,
    batch_train: int = 128,
    batch_test:  int = 256,
    seed: int = 0,
):
    tfm      = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST("data", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST("data", train=False, download=True, transform=tfm)

    rng      = np.random.default_rng(seed)
    tr_idx   = rng.choice(len(train_ds), n_train, replace=False)
    te_idx   = rng.choice(len(test_ds),  n_test,  replace=False)

    pin = DEVICE != "cpu"
    train_loader = DataLoader(Subset(train_ds, tr_idx), batch_size=batch_train, shuffle=True,  num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(Subset(test_ds,  te_idx), batch_size=batch_test, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, test_loader

# train
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    m: int,
    group: str,
    k: int = 4,
) -> tuple[float, float]:
    """
    Train for one epoch under m-sample Haar augmentation.
    """
    model.train()
    total_loss = 0.0
    orbit_vars: list[float] = []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        B = x.shape[0]

        if m == 0:
            #baseline
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
        else:
            # augmented
            angles = sample_haar_angles(m, group, k)

            x_aug = torch.cat([rotate_batch(x, a) for a in angles], dim=0)
            y_aug = y.repeat(m) 

            logits = model(x_aug)
            loss   = F.cross_entropy(logits, y_aug)
            
            with torch.no_grad():
                per_rot = [
                    F.cross_entropy(logits[i * B:(i + 1) * B], y).item()
                    for i in range(m)
                ]
            orbit_vars.append(float(np.var(per_rot)))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss  = total_loss / len(loader)
    avg_ovar  = float(np.mean(orbit_vars)) if orbit_vars else 0.0
    return avg_loss, avg_ovar


# eval
@torch.no_grad()
def _evaluate_with_transform(
    model: nn.Module,
    loader: DataLoader,
    rotate: bool,
    seed: int = 0,
) -> tuple[float, float]:
    model.eval()
    rng     = torch.Generator()
    rng.manual_seed(seed)
    correct = total = 0
    all_probs, all_labels = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        if rotate:
            angles = torch.rand(x.shape[0], generator=rng) * 2 * math.pi
            rotated = torch.stack([
                rotate_batch(x[i:i+1], float(angles[i]))[0]
                for i in range(x.shape[0])
            ])
            x = rotated

        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        all_probs.append(F.softmax(logits, -1).cpu())
        all_labels.append(y.cpu())

    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    return correct / total, _ece(probs, labels)

@torch.no_grad()
def evaluate_clean(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    return _evaluate_with_transform(model, loader, rotate=False)


@torch.no_grad()
def evaluate_rotated(
    model: nn.Module,
    loader: DataLoader,
    seed: int = 999,
) -> tuple[float, float]:
    return _evaluate_with_transform(model, loader, rotate=True, seed=seed)


@torch.no_grad()
def evaluate_rotation_robustness(
    model: nn.Module,
    loader: DataLoader,
    step_deg: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Accuracy at every rotation angle from 0 to 355 (step = step_deg).
    """
    model.eval()
    angles_deg = np.arange(0, 360, step_deg)
    accs: list[float] = []

    for deg in angles_deg:
        rad = math.radians(float(deg))
        correct = total = 0
        for x, y in loader:
            x, y= x.to(DEVICE), y.to(DEVICE)
            x_r = rotate_batch(x, rad)
            correct +=(model(x_r).argmax(1) == y).sum().item()
            total += y.size(0)
        accs.append(correct / total)

    return angles_deg, np.array(accs)

def _ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error (15 equal-width confidence bins)."""
    conf, preds = probs.max(1)
    correct = preds.eq(labels).float()
    ece = 0.0
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum():
            ece += mask.float().mean().item() * abs(correct[mask].mean().item() - conf[mask].mean().item())
    return ece

def run_condition(
    group:   str,
    m:       int,
    k:       int = 4,
    epochs:  int = 15,
    seed:    int = 42,
    n_train: int = 8_000,
    n_test:  int = 2_000,
) -> dict:
    """
    Train one model under (group, m) and return a metrics dictionary.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_loaders(n_train, n_test, seed=seed)
    model     = HaarCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict[str, list] = dict(
        train_loss=[], test_acc_clean=[], test_acc_rotated=[],
        orbit_var=[], epoch_time_s=[]
    )
    t_train_start = time.time()   # total wall-clock for entire training run

    label = "baseline" if m == 0 else f"{group.upper()}  m={m}"
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"Condition : {label}")
    print(f"Epochs    : {epochs}   |   Train N : {n_train}   |   Device : {DEVICE}")
    print(bar)

    for ep in range(1, epochs + 1):
        t_ep_start = time.time()
        loss, ovar = train_one_epoch(model, train_loader, optimizer, m, group, k)
        ep_train_time = time.time() - t_ep_start

        acc_clean, _ = evaluate_clean(model, test_loader)
        acc_rotated, _ = evaluate_rotated(model, test_loader, seed=ep * 17)
        scheduler.step()

        history["train_loss"].append(loss)
        history["test_acc_clean"].append(acc_clean)
        history["test_acc_rotated"].append(acc_rotated)
        history["orbit_var"].append(ovar)
        history["epoch_time_s"].append(ep_train_time)

        print(f"  ep {ep:02d}/{epochs}  "
              f"loss={loss:.4f}  "
              f"acc_clean={acc_clean:.4f}  acc_rotated={acc_rotated:.4f}  "
              f"ovar={ovar:.5f}  train={ep_train_time:.1f}s")

    final_acc_clean, ece_clean = evaluate_clean(model, test_loader)
    final_acc_rotated, ece_rotated = evaluate_rotated(model, test_loader, seed=999)
    angles_deg, rob = evaluate_rotation_robustness(model, test_loader)

    total_train_time_s = time.time() - t_train_start

    return dict(
        group=group, m=m, k=k,
        history=history,
        # PRIMARY metric: accuracy when test images have random SO(2) orientations
        final_acc=final_acc_rotated,
        ece=ece_rotated,
        # DIAGNOSTIC: accuracy on canonical (upright) MNIST images
        final_acc_clean=final_acc_clean,
        ece_clean=ece_clean,
        angles_deg=angles_deg,
        rot_robustness=rob,
        mean_robustness=float(rob.mean()),
        # Timing: total wall-clock for training (excludes eval), and per-epoch
        total_train_time_s=total_train_time_s,
        mean_epoch_time_s=float(np.mean(history["epoch_time_s"])),
    )


def run_all_experiments(
    m_values:    list  = None,
    ck_orders:   list  = None,
    epochs:      int   = 15,
    n_train:     int   = 8_000,
    n_test:      int   = 2_000,
    seed:        int   = 42,
    save_path:   str   = "results.pkl",
    resume:      bool  = True,
) -> dict:
    """
    Run the full suite:
      - Baseline (no augmentation)
      - SO(2) with m in m_values
      - C_k for each k in ck_orders, with m in m_values

    Parameters
    ----------
    m_values  : sampling densities to test, default [1, 2, 4, 8, 16]
    ck_orders : cyclic group orders to test, default [4, 8]
    epochs    : training epochs per condition
    n_train   : training subset size (use <60k for speed)
    n_test    : test subset size
    seed      : global RNG seed
    save_path : pickle file for incremental saves
    resume    : if True, skip already-cached conditions
    """
    if m_values is None:
        m_values = [1, 2, 4, 8, 16]
    if ck_orders is None:
        ck_orders = [4, 8]

    results: dict = {}
    if resume and os.path.exists(save_path):
        with open(save_path, "rb") as f:
            results = pickle.load(f)
        print(f"[resume] Loaded {len(results)} cached conditions from '{save_path}'")

    def _run(key: str, group: str, m: int, k: int = 4) -> None:
        if key in results:
            print(f"[skip]  '{key}' already cached — set resume=False to rerun")
            return
        results[key] = run_condition(
            group, m, k=k, epochs=epochs, seed=seed,
            n_train=n_train, n_test=n_test,
        )
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"[saved] '{save_path}'  ({len(results)} conditions total)")

    # baseline cnn
    _run("baseline", "none", 0)

    # so(2) aug
    for m in m_values:
        _run(f"so2_m{m}", "so2", m)

    # c_k aug
    for k in ck_orders:
        for m in m_values:
            _run(f"c{k}_m{m}", "ck", m, k=k)

    n_total = 1 + len(m_values) + len(ck_orders) * len(m_values)
    print(f"\n{'─'*60}")
    print(f"  All {len(results)}/{n_total} conditions complete.")
    print(f"  Results stored in '{save_path}'")
    return results

if __name__ == "__main__":
    run_all_experiments(ck_orders=[4, 8])

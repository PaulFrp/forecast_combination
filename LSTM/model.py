"""Small LSTM, its training loop, and a multi-seed ensemble (step 5).

LSTMEnsemble exposes the same fit / predict interface as har.HARModel, so the
forecasting and evaluation code treats both models identically.
"""
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

import config
from sequences import make_windows, prepare


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class VolLSTM(nn.Module):
    """1 LSTM layer -> dropout -> linear head on the last hidden state."""

    def __init__(self, n_features: int, units: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, units, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(units, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1])).squeeze(-1)


def _predict(net: nn.Module, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.from_numpy(x[i:i + batch])).numpy()
                               for i in range(0, len(x), batch)]) if len(x) else np.empty(0)


def train_one(x_tr, y_tr, params: dict, seed: int, x_val=None, y_val=None,
              epochs: int | None = None) -> tuple[nn.Module, int, float]:
    """Train one network (MSE on standardised log variance, Adam).

    With validation data: early stopping, returns the best-epoch weights.
    Without: trains for exactly `epochs` epochs (used for refits).
    Returns (net, best_epoch, best_val_loss).
    """
    set_seed(seed)
    net = VolLSTM(x_tr.shape[-1], params["units"], params["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=params["lr"], weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr.astype(np.float32))
    gen = torch.Generator().manual_seed(seed)

    use_val = x_val is not None
    n_epochs = config.MAX_EPOCHS if use_val else epochs
    best_loss, best_epoch, best_state, bad = np.inf, n_epochs, None, 0

    for epoch in range(1, n_epochs + 1):
        net.train()
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), config.BATCH_SIZE):
            b = perm[i:i + config.BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(net(xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        if use_val:
            val_loss = float(np.mean((_predict(net, x_val) - y_val) ** 2))
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch, bad = val_loss, epoch, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= config.PATIENCE:
                    break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_epoch, best_loss


class LSTMEnsemble:
    """Average of several seeds. Works on log average variance, like HARModel."""
    name = config.MODEL_NAME

    def __init__(self, h: int, feature_cols: list[str], params: dict | None = None,
                 seeds=config.SEEDS, epochs: int | None = None):
        self.h, self.feature_cols = h, feature_cols
        self.params = {**config.DEFAULT_PARAMS, **(params or {})}
        self.seeds, self.epochs = list(seeds), epochs
        self.nets, self.best_epochs, self.val_losses = [], [], []

    def fit(self, data: pd.DataFrame, train_pos: np.ndarray, val_pos: np.ndarray | None = None):
        w, ycol = self.params["window"], f"target_{self.h}"
        self.prep = prepare(data, self.feature_cols, self.h, train_pos)
        y = data[ycol].to_numpy()
        scale = lambda pos: ((y[pos] - self.prep.y_mean) / self.prep.y_std).astype(np.float32)

        x_tr, y_tr = self.prep.windows(train_pos, w), scale(train_pos)
        x_val = y_val = None
        if val_pos is not None:
            x_val, y_val = self.prep.windows(val_pos, w), scale(val_pos)
        elif self.epochs is None:
            raise ValueError("Need either validation data (early stopping) or a fixed epoch count.")

        self.nets, self.best_epochs, self.val_losses = [], [], []
        for seed in self.seeds:
            net, ep, vl = train_one(x_tr, y_tr, self.params, seed, x_val, y_val, self.epochs)
            self.nets.append(net)
            self.best_epochs.append(ep)
            self.val_losses.append(vl)
        return self

    def predict_per_seed(self, data: pd.DataFrame, positions: np.ndarray,
                         x_scaled: np.ndarray | None = None) -> np.ndarray:
        """(n_seeds, n_samples) forecasts of log average variance.
        `x_scaled` lets callers pass perturbed features (permutation importance)."""
        xs =self.prep.x_scaled if x_scaled is None else x_scaled
        x = make_windows(xs, positions, self.params["window"]).astype(np.float32)
        preds = np.stack([_predict(net, x) for net in self.nets])
        return preds * self.prep.y_std + self.prep.y_mean

    def predict(self, data: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
        return self.predict_per_seed(data, positions).mean(axis=0)

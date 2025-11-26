"""
Advanced Time Series Forecasting:
LSTM + Self-Attention (multi-head) with Optuna hyperparameter tuning,
synthetic dataset with trend/seasonality/heteroscedastic noise,
and SARIMAX baseline comparison.

Usage:
    pip install torch numpy pandas scikit-learn statsmodels optuna matplotlib tqdm
    python attention_lstm_optuna.py --n_steps 1400 --trials 40 --save_dir ./output
"""

import os
import argparse
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# ML libs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Baseline
import statsmodels.api as sm

# HPO
import optuna

# -------------------------
# Utilities & Logging
# -------------------------
def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(save_dir, "run.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

# -------------------------
# 1) DATA GENERATION
# -------------------------
def generate_synthetic_multivariate(n_steps=1400, n_features=5, seed=42):
    np.random.seed(seed)
    t = np.arange(n_steps)

    trend = 0.0005 * (t**1.5)
    seasonal1 = 0.5 * np.sin(2 * np.pi * t / 24)      # daily
    seasonal2 = 0.3 * np.sin(2 * np.pi * t / (24*7))  # weekly

    base = np.zeros((n_steps, n_features))
    for i in range(n_features):
        phase = np.random.rand() * 2 * np.pi
        ampl = 0.2 + 0.1 * np.random.randn()
        base[:, i] = ampl * np.sin(2*np.pi*t/(24*(1+i%3)) + phase)

    mixing = np.random.randn(n_features, n_features) * 0.1
    mixed = base @ (np.eye(n_features) + mixing)

    noise_scale = 0.02 + 0.05 * (np.abs(seasonal1) + np.abs(seasonal2))
    noise = np.random.randn(n_steps, n_features) * noise_scale[:, None]

    features = mixed + noise + trend[:, None] * (0.01 + 0.05*np.random.randn(n_features))

    weights = np.array([0.3, 0.2, -0.15, 0.25, 0.1])
    target = features @ weights + 0.6*seasonal1 + 0.2*seasonal2 + 0.1*trend
    target_noise = np.random.randn(n_steps) * (0.02 + 0.05 * np.abs(seasonal1))
    target = target + target_noise

    cols = [f"feat_{i}" for i in range(n_features)] + ["target"]
    df = pd.DataFrame(np.column_stack([features, target]), columns=cols)
    return df

# -------------------------
# 2) WINDOWING + DATASET
# -------------------------
def create_windows(df, lookback=24):
    X, y = [], []
    arr = df.values
    for i in range(len(arr) - lookback):
        X.append(arr[i:i+lookback, :-1])
        y.append(arr[i+lookback, -1])
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# 3) MODEL: LSTM + SELF-ATTENTION
# -------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim_per_head = hidden_dim // num_heads

        self.q_lin = nn.Linear(hidden_dim, hidden_dim)
        self.k_lin = nn.Linear(hidden_dim, hidden_dim)
        self.v_lin = nn.Linear(hidden_dim, hidden_dim)
        self.out_lin = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(self.dim_per_head)

    def forward(self, x):
        # x: (B, T, H)
        B, T, H = x.size()
        # Prepare Q,K,V: -> (B, num_heads, T, dim_per_head)
        Q = self.q_lin(x).view(B, T, self.num_heads, self.dim_per_head).transpose(1,2)
        K = self.k_lin(x).view(B, T, self.num_heads, self.dim_per_head).transpose(1,2)
        V = self.v_lin(x).view(B, T, self.num_heads, self.dim_per_head).transpose(1,2)

        # scores: (B, num_heads, T, T)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (B, num_heads, T, dim_per_head)
        context = context.transpose(1,2).contiguous().view(B, T, H)
        out = self.out_lin(context)
        # aggregated_attn: average across heads -> (B, T, T)
        aggregated_attn = attn.mean(dim=1)
        return out, aggregated_attn  # keep as tensors

class AttentionLSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64, num_heads=4, dropout=0.1):
        super().__init__()
        if hidden % num_heads != 0:
            hidden = (hidden // num_heads) * num_heads
        self.hidden = hidden

        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=False)
        self.attn = MultiHeadSelfAttention(hidden, num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        enc_out, (h_n, c_n) = self.lstm(x)           # enc_out: (B, T, hidden)
        attn_out, attn_weights = self.attn(enc_out) # attn_weights: (B, T, T)
        # importance per timestep: average attention that timestep receives across keys (mean over source dim)
        # we'll average the row or column depending on interpretation; here we average the attention given to each key over queries:
        importance = attn_weights.mean(dim=1)  # (B, T)
        importance_norm = importance / (importance.sum(dim=1, keepdim=True) + 1e-8)
        context = (importance_norm.unsqueeze(-1) * enc_out).sum(dim=1)  # (B, hidden)
        out = self.fc(self.dropout(context))
        return out, importance, attn_weights  # return tensors; conversion to numpy is done outside

# -------------------------
# 4) TRAIN / EVALUATE
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds, _, _ = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    ys, yps, atts = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb_cpu = yb  # keep copy on cpu for metrics
            preds, imp_tensor, _ = model(Xb)
            ys.append(yb_cpu.numpy().flatten())
            yps.append(preds.cpu().numpy().flatten())
            atts.append(imp_tensor.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(yps)
    return y_true, y_pred, np.concatenate(atts, axis=0)

# -------------------------
# 5) SARIMAX Baseline
# -------------------------
def sarimax_baseline_train_predict(train_series, test_series, order=(1,1,1), seasonal_order=(0,0,0,0)):
    # train_series and test_series should be 1-d numpy arrays on ORIGINAL scale
    mod = sm.tsa.statespace.SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                                    enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    pred = res.get_forecast(steps=len(test_series))
    y_pred = pred.predicted_mean
    return y_pred, res

# -------------------------
# Fix for Optuna dynamic choices error:
# -------------------------
hidden_candidates = [32, 48, 64, 96, 128]

def objective(trial, X_train, y_train, X_val, y_val, device, save_dir):
    # NOTE: lookback is not re-windowed here; keep a fixed lookback for this HPO run.
    num_heads = trial.suggest_int("num_heads", 1, 4)
    hidden_raw = trial.suggest_categorical("hidden_raw", hidden_candidates)
    # compute a hidden that is divisible by num_heads
    hidden = (hidden_raw // num_heads) * num_heads
    if hidden == 0:
        hidden = num_heads
    # store computed hidden for inspection
    trial.set_user_attr("hidden", int(hidden))

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # removed dynamic lookback (would require re-windowing inside objective)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch = trial.suggest_categorical("batch", [16, 32, 64])
    epochs = 15

    model = AttentionLSTMModel(n_features=X_train.shape[2], hidden=hidden, num_heads=num_heads, dropout=dropout).to(device)
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=batch, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = 1e9
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        yv_true, yv_pred, _ = evaluate(model, val_loader, device)
        val_rmse = rmse(yv_true, yv_pred)

        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_rmse < best_val:
            best_val = val_rmse

    trial.set_user_attr("best_val_rmse", float(best_val))
    return float(best_val)

# -------------------------
# 7) MAIN & Orchestration
# -------------------------
def inverse_transform_targets(scaler, y_scaled, n_features, columns):
    """
    scaler: fitted MinMaxScaler on full dataframe
    y_scaled: 1-d or 2-d array of scaled target(s) (shape (N,) or (N,1))
    return: y in original scale (1-d numpy)
    """
    y_arr = np.array(y_scaled).reshape(-1, 1)
    # build placeholder with zeros for all columns, then set last col to values, inverse transform, extract last col
    placeholder = np.zeros((len(y_arr), n_features + 1), dtype=float)
    placeholder[:, -1] = y_arr.flatten()
    inv = scaler.inverse_transform(placeholder)
    return inv[:, -1]

def run_full_pipeline(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    df = generate_synthetic_multivariate(n_steps=args.n_steps, n_features=5, seed=42)
    df.to_csv(save_dir / "dataset_preview.csv", index=False)
    logging.info(f"Generated dataset with shape: {df.shape}")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled, columns=df.columns)

    lookback = args.lookback
    X, y = create_windows(scaled_df, lookback)
    n = len(X)
    train_end = int(n*0.7)
    val_end = int(n*0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    logging.info(f"Windows: total={n}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    def opt_obj(trial):
        return objective(trial, X_train, y_train, X_val, y_val, device, save_dir)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    logging.info(f"Starting Optuna HPO: trials={args.trials}")
    study.optimize(opt_obj, n_trials=args.trials, timeout=args.timeout)

    logging.info("Best trial:")
    best_trial = study.best_trial
    logging.info(f"  Value (val_rmse - scaled): {best_trial.value}")
    logging.info(f"  Params: {best_trial.params}")

    with open(save_dir / "optuna_best_params.json", "w") as f:
        json.dump({"value": best_trial.value, "params": best_trial.params, "user_attrs": best_trial.user_attrs}, f, indent=2)

    best = best_trial.params
    # derive hidden same way objective did
    hidden_raw = int(best.get("hidden_raw", 64))
    num_heads = int(best.get("num_heads", 1))
    hidden = (hidden_raw // num_heads) * num_heads
    if hidden == 0:
        hidden = num_heads

    lr = float(best.get("lr", 0.001))
    dropout = float(best.get("dropout", 0.1))
    batch = int(best.get("batch", 32))

    final_model = AttentionLSTMModel(n_features=X.shape[2], hidden=hidden, num_heads=num_heads, dropout=dropout).to(device)
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=batch, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch, shuffle=False)

    optimizer = optim.Adam(final_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_rmse = 1e9
    epochs = args.epochs

    for epoch in range(epochs):
        tr_loss = train_one_epoch(final_model, train_loader, optimizer, criterion, device)
        yv_true, yv_pred, _ = evaluate(final_model, val_loader, device)
        val_rmse = rmse(yv_true, yv_pred)
        logging.info(f"[Final train] Epoch {epoch+1}/{epochs} | train_loss={tr_loss:.6f} | val_rmse={val_rmse:.6f}")
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(final_model.state_dict(), save_dir / "best_model.pt")
    logging.info(f"Final best val RMSE (scaled): {best_val_rmse:.6f}")

    y_test_true_scaled, y_test_pred_scaled, test_importances = evaluate(final_model, test_loader, device)

    # inverse transform scaled predictions & true targets to ORIGINAL scale for meaningful comparison
    n_features = X.shape[2]
    y_test_true_orig = inverse_transform_targets(scaler, y_test_true_scaled, n_features, df.columns)
    y_test_pred_orig = inverse_transform_targets(scaler, y_test_pred_scaled, n_features, df.columns)

    final_rmse = rmse(y_test_true_orig, y_test_pred_orig)
    final_mae = mean_absolute_error(y_test_true_orig, y_test_pred_orig)
    final_mape = mape(y_test_true_orig, y_test_pred_orig)
    logging.info(f"Final Test Metrics (Attention-LSTM) RMSE={final_rmse:.6f}, MAE={final_mae:.6f}, MAPE={final_mape:.3f}%")

    np.savetxt(save_dir / "y_test_true_scaled.txt", y_test_true_scaled)
    np.savetxt(save_dir / "y_test_pred_attention_scaled.txt", y_test_pred_scaled)
    np.savetxt(save_dir / "y_test_true_orig.txt", y_test_true_orig)
    np.savetxt(save_dir / "y_test_pred_attention_orig.txt", y_test_pred_orig)
    np.save(save_dir / "test_importances.npy", test_importances)

    avg_imp = test_importances.mean(axis=0)
    plt.figure(figsize=(10,4))
    plt.plot(avg_imp)
    plt.title("Average Attention Importance (test set)")
    plt.xlabel("Encoder timestep (older -> newer)")
    plt.ylabel("Importance (avg across samples)")
    plt.savefig(save_dir / "attention_importance_avg.png", bbox_inches="tight")
    logging.info("Saved attention_importance_avg.png")

    # Prepare train/test series on ORIGINAL scale for SARIMAX
    # Need to align indices: training on first (train_end + lookback) original target points
    train_target_orig = df['target'].values[:train_end + lookback]
    test_target_orig = df['target'].values[train_end + lookback: train_end + lookback + len(y_test_true_orig)]
    try:
        sarimax_pred_orig, sarimax_res = sarimax_baseline_train_predict(train_target_orig, test_target_orig, order=(2,0,2))
        sar_rmse = rmse(test_target_orig, sarimax_pred_orig)
        sar_mae = mean_absolute_error(test_target_orig, sarimax_pred_orig)
        sar_mape = mape(test_target_orig, sarimax_pred_orig)
        logging.info(f"SARIMAX Baseline Metrics RMSE={sar_rmse:.6f}, MAE={sar_mae:.6f}, MAPE={sar_mape:.3f}%")
        np.savetxt(save_dir / "y_test_pred_sarimax_orig.txt", sarimax_pred_orig)
    except Exception as e:
        logging.exception("SARIMAX baseline failed (likely convergence). Continuing without baseline.")
        sar_rmse = sar_mae = sar_mape = None
        sarimax_pred_orig = None

    # Save summary with consistent original-scale metrics where available
    summary = {
        "final_attention_lstm": {"rmse_orig": float(final_rmse), "mae_orig": float(final_mae), "mape_orig": float(final_mape)},
        "sarimax": {"rmse_orig": sar_rmse, "mae_orig": sar_mae, "mape_orig": sar_mape},
        "optuna_best": {"params": best_trial.params, "user_attrs": best_trial.user_attrs}
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # plot predictions (original scale)
    plt.figure(figsize=(12,4))
    plt.plot(y_test_true_orig, label="y_true (orig)")
    plt.plot(y_test_pred_orig, label="Attention-LSTM pred (orig)")
    if sarimax_pred_orig is not None:
        plt.plot(sarimax_pred_orig, label="SARIMAX pred (orig)")
    plt.legend()
    plt.title("Test Predictions (original scale)")
    plt.savefig(save_dir / "predictions_test_orig.png", bbox_inches="tight")
    logging.info("Saved predictions_test_orig.png")

    logging.info(f"All artifacts saved into: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=1400)
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--save_dir", type=str, default="./output")
    args = parser.parse_args()

    run_full_pipeline(args)

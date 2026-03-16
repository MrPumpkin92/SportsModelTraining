"""
Evaluates all three architectures on held-out validation data.
Selects the best model by RMSE on PTS/REB/AST prediction.
Writes selection result to data/local/model_selection.json.

Architectures
-------------
  neural_network  : LSTM (64) + 2x Dense with dropout (PyTorch)
  topology        : Vietoris-Rips persistence features + XGBoost (giotto-tda)
  field_theory    : Statistical-mechanics mean-field / Ising model (numpy/scipy)

Selection logic
---------------
  1. Compute validation RMSE for every successfully-loaded architecture.
  2. Pick the architecture with the lowest mean RMSE across PTS, REB, AST.
  3. Manual override: set model.force_architecture in config/model_config.yaml.

Usage
-----
  python -m src.ml.model_selector
  python -m src.ml.model_selector --force neural_network
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# All heavy ML dependencies (sklearn, scipy, yaml, torch, giotto-tda, xgboost)
# are imported lazily inside each function so the module can be imported when
# only numpy + pandas are installed.

# ---------------------------------------------------------------------------
# Minimal numpy-backed shims (replaced by sklearn when available)
# ---------------------------------------------------------------------------


class _NumpyScaler:
    """Minimal StandardScaler backed purely by numpy."""

    _mean: np.ndarray
    _std: np.ndarray

    def fit(self, X: np.ndarray) -> "_NumpyScaler":
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0) + 1e-8
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return (X - self._mean) / self._std

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std


def _make_scaler() -> Any:
    """Return sklearn StandardScaler when available, else the numpy fallback."""
    try:
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    except ImportError:
        return _NumpyScaler()


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE with sklearn when available, pure numpy otherwise."""
    try:
        from sklearn.metrics import mean_squared_error
        return float(mean_squared_error(y_true, y_pred))
    except ImportError:
        a = np.asarray(y_true, dtype=np.float64)
        b = np.asarray(y_pred, dtype=np.float64)
        return float(np.mean((a - b) ** 2))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHITECTURES = ["neural_network", "topology", "field_theory"]

DATA_DIR = Path("data/local")
RAW_DIR = DATA_DIR / "raw"
CONFIG_PATH = Path("config/model_config.yaml")
SELECTION_OUTPUT = DATA_DIR / "model_selection.json"

# 20 box-score stats used for rolling feature engineering
STAT_COLS: list[str] = [
    "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF",
    "PTS", "PLUS_MINUS", "MIN",
]

# 3 rolling windows x 20 stats = 60 features
ROLL_WINDOWS: list[int] = [3, 5, 10]
FEATURE_COLS: list[str] = [
    f"{s}_roll{w}" for w in ROLL_WINDOWS for s in STAT_COLS
]

TARGET_COLS: list[str] = ["PTS", "REB", "AST"]

# Number of past games fed as a sequence into the LSTM
SEQ_LEN = 10

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------


def load_raw_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load every date-partitioned box_scores.parquet into one DataFrame."""
    frames = [
        pd.read_parquet(p)
        for p in sorted(raw_dir.rglob("box_scores.parquet"))
    ]
    if not frames:
        raise FileNotFoundError(f"No box_scores.parquet files found under {raw_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each stat in STAT_COLS compute rolling means over ROLL_WINDOWS.
    Features are shifted by 1 game so they only use *previous* game data.
    Returns a DataFrame with 60 added feature columns.
    """
    df = df.copy()
    for w in ROLL_WINDOWS:
        for col in STAT_COLS:
            if col in df.columns:
                df[f"{col}_roll{w}"] = (
                    df.groupby("PLAYER_ID")[col]
                    .transform(lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).mean())
                )
    present_feats = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=present_feats + TARGET_COLS).reset_index(drop=True)
    return df


def _build_train_sequences(
    train_df: pd.DataFrame,
    stat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Sequences from training data only: (n, SEQ_LEN, n_stats) -> (n, 3)."""
    X_seqs: list[np.ndarray] = []
    y_seqs: list[np.ndarray] = []
    for _, pdf in train_df.groupby("PLAYER_ID"):
        pdf = pdf.sort_values("GAME_DATE").reset_index(drop=True)
        vals = pdf[stat_cols].values.astype(np.float32)
        tgts = pdf[TARGET_COLS].values.astype(np.float32)
        for i in range(SEQ_LEN, len(pdf)):
            X_seqs.append(vals[i - SEQ_LEN: i])
            y_seqs.append(tgts[i])
    if not X_seqs:
        n = len(stat_cols)
        return np.empty((0, SEQ_LEN, n), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


def _build_val_sequences(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    stat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build val sequences by prepending the last SEQ_LEN training games per
    player to their val games, giving full look-back context at eval time.
    """
    X_seqs: list[np.ndarray] = []
    y_seqs: list[np.ndarray] = []
    for pid, vpdf in val_df.groupby("PLAYER_ID"):
        vpdf = vpdf.sort_values("GAME_DATE").reset_index(drop=True)
        history = (
            train_df[train_df["PLAYER_ID"] == pid]
            .sort_values("GAME_DATE")
            .tail(SEQ_LEN)
        )
        combined = (
            pd.concat([history, vpdf])
            .sort_values("GAME_DATE")
            .reset_index(drop=True)
        )
        n_hist = len(history)
        vals = combined[stat_cols].values.astype(np.float32)
        tgts = combined[TARGET_COLS].values.astype(np.float32)
        for i in range(SEQ_LEN, len(combined)):
            if i >= n_hist:  # target game is in the validation window
                X_seqs.append(vals[i - SEQ_LEN: i])
                y_seqs.append(tgts[i])
    if not X_seqs:
        n = len(stat_cols)
        return np.empty((0, SEQ_LEN, n), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


def train_val_split(
    df: pd.DataFrame,
    val_days: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Temporal split: the last val_days of distinct game dates become the
    validation set; everything earlier is training.
    Returns (train_df, val_df, X_train, y_train, X_val, y_val).
    """
    cutoff = df["GAME_DATE"].max() - timedelta(days=val_days)
    train_df = df[df["GAME_DATE"] <= cutoff].copy()
    val_df = df[df["GAME_DATE"] > cutoff].copy()
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_train = train_df[feat_cols].values.astype(np.float32)
    X_val = val_df[feat_cols].values.astype(np.float32)
    y_train = train_df[TARGET_COLS].values.astype(np.float32)
    y_val = val_df[TARGET_COLS].values.astype(np.float32)
    return train_df, val_df, X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Selection Heuristics
# ---------------------------------------------------------------------------


def _test_temporal_autocorrelation(df: pd.DataFrame) -> bool:
    """
    Ljung-Box test on each player's PTS series.
    Returns True when mean p-value < 0.05 (significant autocorrelation -> prefer LSTM).
    Requires statsmodels; returns False gracefully if unavailable.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        LOG.warning("statsmodels not installed; Ljung-Box test skipped")
        return False

    p_values: list[float] = []
    for _, pdf in df.groupby("PLAYER_ID"):
        pts = pdf.sort_values("GAME_DATE")["PTS"].dropna()
        if len(pts) < 10:
            continue
        try:
            result = acorr_ljungbox(pts.values, lags=[5], return_df=True)
            p_values.append(float(result["lb_pvalue"].iloc[0]))
        except Exception:
            continue
    if not p_values:
        return False
    return float(np.mean(p_values)) < 0.05


def _test_topological_variance(X: np.ndarray, seed: int = 42) -> bool:
    """
    Compute global persistence entropy on a sample of training points.
    Returns True when mean entropy > 2.0 (high topological variance -> prefer TDA).
    Requires giotto-tda + sklearn; returns False gracefully if unavailable.
    """
    try:
        from gtda.homology import VietorisRipsPersistence
        from gtda.diagrams import PersistenceEntropy
        from sklearn.decomposition import PCA
    except ImportError:
        LOG.warning("giotto-tda / sklearn not installed; topological-variance heuristic skipped")
        return False

    rng = np.random.default_rng(seed)
    n_sample = min(200, len(X))
    idx = rng.choice(len(X), n_sample, replace=False)
    sample = X[idx]

    pca = PCA(n_components=min(5, sample.shape[1]))
    sample_low = pca.fit_transform(sample)            # (n, 5)
    cloud = sample_low.reshape(1, *sample_low.shape)  # (1, n, 5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vrp = VietorisRipsPersistence(homology_dimensions=[0, 1])
        diagrams = vrp.fit_transform(cloud)
        pe = PersistenceEntropy()
        entropies = pe.fit_transform(diagrams)[0]     # (2,)

    return float(np.nanmean(entropies)) > 2.0


def _test_team_synergy_variance(
    df: pd.DataFrame,
    y_train: np.ndarray,
) -> bool:
    """
    Fit Ridge on team-mean features to per-player residuals.
    Returns True when team synergy explains > 15% of residual variance (-> prefer Field Theory).
    Requires sklearn; returns False gracefully if unavailable.
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        LOG.warning("sklearn not installed; team-synergy heuristic skipped")
        return False

    player_means = df.groupby("PLAYER_ID")[TARGET_COLS].transform("mean").values
    residuals = y_train - player_means
    base_var = float(np.var(residuals)) + 1e-9

    team_means = df.groupby("TEAM_ID")[TARGET_COLS].transform("mean").values
    features = np.hstack([player_means, team_means]).astype(np.float32)
    scaler = _make_scaler()
    features_n = scaler.fit_transform(features)

    model = Ridge()
    model.fit(features_n, residuals)
    synergy_var = float(np.var(residuals - model.predict(features_n)))
    return (base_var - synergy_var) / base_var > 0.15


# ---------------------------------------------------------------------------
# Architecture A - Neural Network (LSTM + Dense)
# ---------------------------------------------------------------------------


def _eval_neural_network(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """
    LSTM (64 units) -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(64, ReLU)
    -> regression head for PTS / REB / AST.
    Input: rolling sequences of raw box-score stats (SEQ_LEN x 20).
    Requires: torch
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("PyTorch required for neural_network: pip install torch") from exc

    seed = int(config.get("random_seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    stat_cols = [c for c in STAT_COLS if c in df.columns]
    n_stats = len(stat_cols)

    X_tr, y_tr = _build_train_sequences(train_df, stat_cols)
    X_vl, y_vl = _build_val_sequences(train_df, val_df, stat_cols)

    if len(X_tr) < 32 or len(X_vl) == 0:
        LOG.warning("neural_network: insufficient sequence data - returning sentinel RMSE")
        return {"rmse": {c: 999.0 for c in TARGET_COLS}, "mean_rmse": 999.0}

    # Normalise per-stat across the time axis
    scaler = _make_scaler()
    scaler.fit(X_tr.reshape(-1, n_stats))
    X_tr_n = scaler.transform(X_tr.reshape(-1, n_stats)).reshape(X_tr.shape)
    X_vl_n = scaler.transform(X_vl.reshape(-1, n_stats)).reshape(X_vl.shape)

    class LSTMPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(n_stats, 64, batch_first=True)
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 64)
            self.dropout = nn.Dropout(0.3)
            self.out = nn.Linear(64, len(TARGET_COLS))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, (h, _) = self.lstm(x)
            z = h.squeeze(0)
            z = torch.relu(self.fc1(z))
            z = self.dropout(z)
            z = torch.relu(self.fc2(z))
            return self.out(z)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr_n, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        ),
        batch_size=64,
        shuffle=True,
    )

    model.train()
    for epoch in range(30):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            LOG.debug("neural_network  epoch %d/30", epoch + 1)

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_vl_n).to(device)).cpu().numpy()

    rmse = {
        col: float(np.sqrt(_mse(y_vl[:, i], preds[:, i])))
        for i, col in enumerate(TARGET_COLS)
    }
    mean_rmse = float(np.mean(list(rmse.values())))
    LOG.info("neural_network  RMSE %s  (mean %.4f)", rmse, mean_rmse)
    return {"rmse": rmse, "mean_rmse": mean_rmse}


# ---------------------------------------------------------------------------
# Architecture B - Topology (TDA + XGBoost)
# ---------------------------------------------------------------------------


def _local_persistence_entropy(
    X_ref_low: np.ndarray,
    X_query_low: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    """
    For every point in X_query_low compute H0 and H1 persistence entropy
    of its k-nearest-neighbour cloud in X_ref_low.
    Returns shape (len(X_query_low), 2).
    Requires: giotto-tda, sklearn
    """
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy
    from sklearn.neighbors import NearestNeighbors

    nn_model = NearestNeighbors(n_neighbors=k).fit(X_ref_low)
    _, indices = nn_model.kneighbors(X_query_low)    # (n_query, k)
    clouds = X_ref_low[indices]                       # (n_query, k, n_dims)

    vrp = VietorisRipsPersistence(homology_dimensions=[0, 1])
    diagrams = vrp.fit_transform(clouds)              # (n_query, n_gen, 3)
    pe = PersistenceEntropy()
    entropies = pe.fit_transform(diagrams)            # (n_query, 2)
    return np.nan_to_num(entropies, nan=0.0)


def _eval_topology(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
) -> dict[str, Any]:
    """
    Augments the 60 rolling features with local H0/H1 Vietoris-Rips
    persistence entropy, then trains an XGBoost regressor per target.
    Requires: giotto-tda, sklearn, xgboost
    """
    try:
        import xgboost as xgb
        from gtda.homology import VietorisRipsPersistence  # noqa: F401 - import check
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError(
            "giotto-tda and xgboost required: pip install giotto-tda xgboost"
        ) from exc

    seed = int(config.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    scaler = _make_scaler()
    X_tr_n = scaler.fit_transform(X_train)
    X_vl_n = scaler.transform(X_val)

    # PCA to low dims for VR complex (expensive in high dimensions)
    n_pca = min(5, X_tr_n.shape[1])
    pca = PCA(n_components=n_pca, random_state=seed)

    # Fixed-size reference point cloud for k-NN topology
    n_ref = min(400, len(X_tr_n))
    ref_idx = rng.choice(len(X_tr_n), n_ref, replace=False)
    X_ref_low = pca.fit_transform(X_tr_n[ref_idx])   # (n_ref, n_pca)
    X_tr_low = pca.transform(X_tr_n)                  # (n_tr, n_pca)
    X_vl_low = pca.transform(X_vl_n)                  # (n_val, n_pca)

    LOG.info(
        "topology: computing persistence entropy (ref=%d, train=%d, val=%d)...",
        n_ref, len(X_tr_low), len(X_vl_low),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topo_tr = _local_persistence_entropy(X_ref_low, X_tr_low)   # (n_tr, 2)
        topo_vl = _local_persistence_entropy(X_ref_low, X_vl_low)   # (n_val, 2)

    X_tr_aug = np.hstack([X_tr_n, topo_tr])    # (n_tr, 62)
    X_vl_aug = np.hstack([X_vl_n, topo_vl])    # (n_val, 62)

    rmse: dict[str, float] = {}
    for i, col in enumerate(TARGET_COLS):
        reg = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=0,
            n_jobs=-1,
        )
        reg.fit(X_tr_aug, y_train[:, i])
        preds = reg.predict(X_vl_aug)
        rmse[col] = float(np.sqrt(_mse(y_val[:, i], preds)))

    mean_rmse = float(np.mean(list(rmse.values())))
    LOG.info("topology        RMSE %s  (mean %.4f)", rmse, mean_rmse)
    return {"rmse": rmse, "mean_rmse": mean_rmse}


# ---------------------------------------------------------------------------
# Architecture C - Field Theory (Statistical Mechanics / Ising mean-field)
# ---------------------------------------------------------------------------


def _build_field_features(df: pd.DataFrame) -> np.ndarray:
    """
    Augment the 60 rolling features with interaction-field terms:

      Coupling field  : per-team-game rolling means for PTS, REB, AST, MIN
                        (captures teammate synergy: J_ij * <s_j>)
      Coupling product: individual x team-mean cross-terms
                        (Ising-style J_ij * s_i * s_j proxy)
      External field  : PLUS_MINUS_roll3 (encodes opponent defensive pressure h_i)
    """
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_base = df[feat_cols].values.astype(np.float32)

    group_key = ["TEAM_ID", "GAME_DATE"]
    parts: list[np.ndarray] = []

    for col in ["PTS_roll3", "REB_roll3", "AST_roll3", "MIN_roll3"]:
        if col not in df.columns:
            continue
        team_mean = df.groupby(group_key)[col].transform("mean").values.astype(np.float32)
        individual = df[col].values.astype(np.float32)
        parts.append(team_mean)               # coupling field: <s_j>
        parts.append(individual * team_mean)  # Ising term: s_i * <s_j>

    if "PLUS_MINUS_roll3" in df.columns:
        parts.append(df["PLUS_MINUS_roll3"].values.astype(np.float32))  # external field h_i

    if parts:
        return np.hstack([X_base, np.column_stack(parts)])
    return X_base


def _eval_field_theory(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """
    Mean-field Hamiltonian model:
      H = -sum J_ij s_i s_j  -  sum h_i s_i
    Predicted output ~= W . x_field + b, where W and b are fitted per target
    via L-BFGS-B maximum-likelihood (equivalent to least-squares on the field
    features).
    Requires: scipy
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("scipy required for field_theory: pip install scipy") from exc

    seed = int(config.get("random_seed", 42))
    np.random.seed(seed)

    X_tr = _build_field_features(train_df)
    X_vl = _build_field_features(val_df)

    scaler = _make_scaler()
    X_tr_n = scaler.fit_transform(X_tr).astype(np.float64)
    X_vl_n = scaler.transform(X_vl).astype(np.float64)

    y_tr = train_df[TARGET_COLS].values.astype(np.float64)
    y_vl = val_df[TARGET_COLS].values.astype(np.float64)

    n_features = X_tr_n.shape[1]
    rmse: dict[str, float] = {}

    for i, col in enumerate(TARGET_COLS):
        y = y_tr[:, i]

        def neg_log_likelihood(params: np.ndarray, _y: np.ndarray = y) -> float:
            pred = X_tr_n @ params[:n_features] + params[n_features]
            return float(np.mean((pred - _y) ** 2))

        def gradient(params: np.ndarray, _y: np.ndarray = y) -> np.ndarray:
            resid = X_tr_n @ params[:n_features] + params[n_features] - _y
            dW = 2.0 * (X_tr_n.T @ resid) / len(_y)
            db = 2.0 * float(np.mean(resid))
            return np.append(dW, db)

        result = minimize(
            neg_log_likelihood,
            x0=np.zeros(n_features + 1),
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9},
        )
        W_opt, b_opt = result.x[:n_features], result.x[n_features]
        preds = X_vl_n @ W_opt + b_opt
        rmse[col] = float(np.sqrt(_mse(y_vl[:, i], preds)))

    mean_rmse = float(np.mean(list(rmse.values())))
    LOG.info("field_theory    RMSE %s  (mean %.4f)", rmse, mean_rmse)
    return {"rmse": rmse, "mean_rmse": mean_rmse}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load model config YAML; falls back to safe defaults if absent."""
    defaults: dict[str, Any] = {
        "force_architecture": None,
        "train_test_split": 0.8,
        "validation_window_days": 14,
        "target_columns": list(TARGET_COLS),
        "random_seed": 42,
    }
    if not config_path.exists():
        LOG.warning("Config not found at %s; using defaults", config_path)
        return defaults

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
    except ImportError:
        # pyyaml not installed: best-effort line-by-line parse for simple k:v YAML
        import re as _re
        loaded = {}
        for line in config_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, _, val = line.partition(":")
            val = val.strip()
            if val and val.lower() != "null":
                loaded[key.strip()] = val
        loaded = {"model": loaded}

    defaults.update(loaded.get("model", {}))
    return defaults


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_selection(
    features_path: str | None = None,
    config: dict | None = None,
) -> str:
    """
    Evaluate all three architectures and return the name of the best one.

    Parameters
    ----------
    features_path:
        Optional override for the raw data directory (default: data/local/raw).
    config:
        Optional config dict that overrides config/model_config.yaml.

    Returns
    -------
    str
        The name of the selected architecture.

    Side effects
    ------------
    Writes data/local/model_selection.json with full evaluation results.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = load_config()
    if config:
        cfg.update(config)

    raw_dir = Path(features_path) if features_path else RAW_DIR

    LOG.info("Loading raw data from %s ...", raw_dir)
    df = load_raw_data(raw_dir)
    LOG.info("Loaded %d rows across %d players", len(df), df["PLAYER_ID"].nunique())

    df = build_rolling_features(df)
    LOG.info("Feature matrix: %d rows x %d rolling features", len(df), len(FEATURE_COLS))

    val_days = int(cfg.get("validation_window_days", 14))
    train_df, val_df, X_train, y_train, X_val, y_val = train_val_split(df, val_days)
    LOG.info(
        "Train: %d rows  (%s -> %s) | Val: %d rows  (%s -> %s)",
        len(train_df),
        train_df["GAME_DATE"].min().date(),
        train_df["GAME_DATE"].max().date(),
        len(val_df),
        val_df["GAME_DATE"].min().date(),
        val_df["GAME_DATE"].max().date(),
    )

    # ---- diagnostic heuristics ----
    LOG.info("Running selection heuristics ...")
    heuristics = {
        "temporal_autocorrelation_ljungbox": _test_temporal_autocorrelation(train_df),
        "topological_variance_entropy_gt2": _test_topological_variance(
            X_train, seed=int(cfg.get("random_seed", 42))
        ),
        "team_synergy_variance_gt15pct": _test_team_synergy_variance(train_df, y_train),
    }
    LOG.info(
        "Heuristics - NN favour: %s | TDA favour: %s | Field favour: %s",
        heuristics["temporal_autocorrelation_ljungbox"],
        heuristics["topological_variance_entropy_gt2"],
        heuristics["team_synergy_variance_gt15pct"],
    )

    # ---- manual override ----
    force = cfg.get("force_architecture")
    if force:
        if force not in ARCHITECTURES:
            raise ValueError(
                f"force_architecture must be one of {ARCHITECTURES}, got {force!r}"
            )
        LOG.info("Manual override: architecture = %s", force)
        result: dict[str, Any] = {
            "selected_architecture": force,
            "selection_method": "manual_override",
            "heuristics": heuristics,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "rmse_scores": {},
            "mean_rmse": {},
            "errors": {},
        }
        SELECTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        SELECTION_OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        LOG.info("Selection result saved to %s", SELECTION_OUTPUT)
        return force

    # ---- evaluate all architectures ----
    eval_results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    for arch in ARCHITECTURES:
        LOG.info("--- Evaluating: %s ---", arch)
        try:
            if arch == "neural_network":
                eval_results[arch] = _eval_neural_network(df, train_df, val_df, cfg)
            elif arch == "topology":
                eval_results[arch] = _eval_topology(X_train, y_train, X_val, y_val, cfg)
            elif arch == "field_theory":
                eval_results[arch] = _eval_field_theory(train_df, val_df, cfg)
        except ImportError as exc:
            LOG.warning("Skipping %s (missing dependency): %s", arch, exc)
            errors[arch] = str(exc)
            eval_results[arch] = {"mean_rmse": float("inf"), "rmse": {}}
        except Exception as exc:
            LOG.error("Error evaluating %s: %s", arch, exc, exc_info=True)
            errors[arch] = str(exc)
            eval_results[arch] = {"mean_rmse": float("inf"), "rmse": {}}

    finite = {k: v for k, v in eval_results.items() if v["mean_rmse"] != float("inf")}
    if not finite:
        raise RuntimeError(
            "All architectures failed. Install missing dependencies and retry.\n"
            f"Errors: {errors}"
        )

    best = min(finite, key=lambda k: finite[k]["mean_rmse"])
    LOG.info("Selected architecture: %s  (mean RMSE = %.4f)", best, finite[best]["mean_rmse"])

    for arch in ARCHITECTURES:
        v = eval_results[arch].get("mean_rmse")
        score_str = f"{v:.4f}" if isinstance(v, float) and v != float("inf") else str(v)
        marker = " <- selected" if arch == best else ""
        LOG.info("  %-16s  mean RMSE = %s%s", arch, score_str, marker)

    output: dict[str, Any] = {
        "selected_architecture": best,
        "selection_method": "rmse_comparison",
        "heuristics": heuristics,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "rmse_scores": {k: v.get("rmse", {}) for k, v in eval_results.items()},
        "mean_rmse": {k: v.get("mean_rmse") for k, v in eval_results.items()},
        "errors": errors,
    }
    SELECTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    LOG.info("Selection result saved to %s", SELECTION_OUTPUT)

    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cfg_override: dict[str, Any] = {}
    args = sys.argv[1:]
    if "--force" in args:
        idx = args.index("--force")
        if idx + 1 >= len(args):
            print("Usage: python -m src.ml.model_selector [--force <architecture>]")
            sys.exit(1)
        cfg_override["force_architecture"] = args[idx + 1]

    selected = run_selection(config=cfg_override)
    print(f"\nSelected architecture: {selected}")

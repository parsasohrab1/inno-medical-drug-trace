"""Drug demand forecasting, shortage/expiry alerts and stock-transfer suggestions.

Lab prototype on SYNTHETIC data only (TRL 4).
"""
import numpy as np

ITEMS, CENTERS, WEEKS = 8, 4, 156
ORIGINS = (112, 122, 132, 142, 150)
HORIZON = 4

CRITERIA = [
    ("wape_gain", ">=", 0.10, "forecast WAPE improvement over 4-week moving average"),
    ("expiry_recall", ">=", 0.80, "recall of batches that would expire unsold"),
    ("expiry_precision", ">=", 0.70, "precision of expiry alerts"),
    ("shortage_recall", ">=", 0.80, "recall of true 4-week shortages"),
    ("shortage_precision", ">=", 0.70, "precision of shortage alerts"),
]


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(WEEKS)
    base = rng.uniform(40, 200, (ITEMS, CENTERS))
    amp = rng.uniform(0.15, 0.35, (ITEMS, 1, 1))
    ph = rng.uniform(0, 2 * np.pi, (ITEMS, 1, 1))
    trend = rng.uniform(-0.001, 0.003, (ITEMS, CENTERS, 1))
    mu = base[:, :, None] * (1 + amp * np.sin(2 * np.pi * t / 52 + ph)) * (1 + trend * t)
    return mu, rng.poisson(mu)


def _design(t):
    t = np.asarray(t, float)
    return np.stack([np.ones_like(t), t, np.sin(2 * np.pi * t / 52), np.cos(2 * np.pi * t / 52)], axis=-1)


def forecast(history, horizon=HORIZON):
    """history: (..., T) weekly units -> (..., horizon) point forecast (harmonic regression + level correction)."""
    history = np.asarray(history, float)
    T = history.shape[-1]
    X = _design(np.arange(T))
    flat = history.reshape(-1, T)
    out = np.empty((flat.shape[0], horizon))
    Xf = _design(np.arange(T, T + horizon))
    for i, y in enumerate(flat):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fit = X @ beta
        level = np.mean(y[-8:] - fit[-8:]) * 0.5  # damped recent-level correction
        out[i] = np.maximum(Xf @ beta + level, 0)
    return out.reshape(history.shape[:-1] + (horizon,))


def baseline(history, horizon=HORIZON):
    h = np.asarray(history, float)
    return np.repeat(h[..., -4:].mean(axis=-1, keepdims=True), horizon, axis=-1)


def unsold_fraction(qty, cum_consumption):
    return np.maximum(qty - cum_consumption, 0) / np.maximum(qty, 1e-9)


def suggest_transfers(surplus, deficit):
    """surplus/deficit: dict center -> units for ONE item. Greedy largest-deficit-first. Never moves more than surplus."""
    surplus = {c: float(v) for c, v in surplus.items() if v > 0}
    moves = []
    for rc, need in sorted(((c, v) for c, v in deficit.items() if v > 0), key=lambda x: -x[1]):
        need = float(need)
        for dc in sorted(surplus, key=lambda c: -surplus[c]):
            if need <= 0:
                break
            q = min(surplus[dc], need)
            if q >= 1:
                moves.append({"from": dc, "to": rc, "units": int(q)})
                surplus[dc] -= q
                need -= q
    return moves


def _scenario(seed):
    mu, y = simulate(seed)
    rng = np.random.default_rng(seed + 1000)
    rows = []
    for T in ORIGINS:
        F = forecast(y[:, :, :T])
        B = baseline(y[:, :, :T])
        A = y[:, :, T:T + HORIZON]
        for i in range(ITEMS):
            for c in range(CENTERS):
                m = mu[i, c, T]
                weeks_to_exp = int(rng.integers(1, HORIZON + 1))
                qty = float(rng.uniform(0.5, 2.0) * m * weeks_to_exp)
                stock = float(rng.uniform(0.5, 2.0) * m * HORIZON)
                cf = F[i, c, :weeks_to_exp].sum()
                ca = A[i, c, :weeks_to_exp].sum()
                rows.append(dict(item=i, center=c, origin=T, weeks_to_exp=weeks_to_exp, batch_qty=round(qty, 1),
                                 stock=round(stock, 1), f=F[i, c], b=B[i, c], a=A[i, c], cf=cf, ca=ca))
    return rows


def evaluate(seed=0):
    rows = _scenario(seed)
    fa = np.array([r["a"] for r in rows]); ff = np.array([r["f"] for r in rows]); fb = np.array([r["b"] for r in rows])
    wape_m = np.abs(ff - fa).sum() / fa.sum(); wape_b = np.abs(fb - fa).sum() / fa.sum()
    ep = np.array([unsold_fraction(r["batch_qty"], r["cf"]) > 0.10 for r in rows])
    et = np.array([unsold_fraction(r["batch_qty"], r["ca"]) > 0.10 for r in rows])
    sp = np.array([r["stock"] < r["f"].sum() for r in rows]); st = np.array([r["stock"] < r["a"].sum() for r in rows])

    def pr(p, t):
        tp = (p & t).sum()
        return tp / max(t.sum(), 1), tp / max(p.sum(), 1)
    er, epr = pr(ep, et); sr, spr = pr(sp, st)
    return {"wape_model": float(wape_m), "wape_baseline": float(wape_b), "wape_gain": float(1 - wape_m / wape_b),
            "expiry_recall": float(er), "expiry_precision": float(epr),
            "shortage_recall": float(sr), "shortage_precision": float(spr), "n_cases": len(rows)}


def generate(seed=0):
    mu, y = simulate(seed)
    rng = np.random.default_rng(seed + 1000)
    demand = [dict(item=f"DRG{i:02d}", center=f"C{c}", week=int(t), units=int(y[i, c, t]))
              for i in range(ITEMS) for c in range(CENTERS) for t in range(WEEKS)]
    batches = []
    for i in range(ITEMS):
        for c in range(CENTERS):
            batches.append(dict(item=f"DRG{i:02d}", center=f"C{c}", batch=f"B{i}{c}",
                                qty=int(rng.uniform(0.5, 2.0) * mu[i, c, -1] * 3), expiry_week=int(WEEKS + rng.integers(1, 5))))
    return {"demand": demand, "batches": batches}


def demo(seed=0):
    mu, y = simulate(seed)
    F = forecast(y[0, 0, :150])
    moves = suggest_transfers({"C0": 120, "C1": 40}, {"C2": 100, "C3": 90})
    return f"Forecast next 4 weeks (item 0, center 0): {F.round(0).tolist()}\nTransfers: {moves}"

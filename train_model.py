"""
Clinic Wait-Time Simulator — data generation, training, and evaluation.

1. Simulates a walk-in clinic (synthetic data, discrete-event queue).
2. Trains a small neural network to predict a new patient's wait (regression).
3. Compares it with simple baselines on a held-out test set.
4. Finds the model's biggest test-set mistake.
5. Exports weights + results to docs/model.json for the website.

Run:  python train_model.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

SEED = 42
N_SAMPLES = 8000
URGENT_SHARE = 0.10  # share of new walk-ins who are urgent and jump ahead of you
OUT = Path(__file__).parent

FEATURES = ["doctors", "queue_ahead", "visit_minutes", "walkins_per_hour"]
RANGES = {  # (min, max) sampled uniformly; doctors and queue are integers
    "doctors": (1, 6),
    "queue_ahead": (0, 15),
    "visit_minutes": (5, 25),
    "walkins_per_hour": (2, 15),
}


# ---------------------------------------------------------------- simulation
def simulate_wait(doctors, queue_ahead, visit_minutes, walkins_per_hour, rng):
    """Minutes until a NON-urgent patient who just walked in is seen.

    - Each doctor is either busy (with some time left) or free.
    - `queue_ahead` non-urgent patients are already waiting (first come, first served).
    - New walk-ins keep arriving (Poisson). Urgent ones (10%) are seen before
      any waiting non-urgent patient, including you. Non-urgent ones queue behind you.
    - Visit lengths are random (lognormal, mean = visit_minutes, CV = 0.5).
    """
    sigma = np.sqrt(np.log(1 + 0.5**2))
    mu = np.log(visit_minutes) - sigma**2 / 2

    def visit():
        return rng.lognormal(mu, sigma)

    # time at which each doctor becomes free
    busy_p = 1.0 if queue_ahead > 0 else 0.7
    free_at = np.array(
        [rng.uniform(0, visit_minutes * 1.5) if rng.random() < busy_p else 0.0
         for _ in range(doctors)]
    )

    # future urgent arrivals (only urgent arrivals can affect your wait)
    urgent_rate = walkins_per_hour * URGENT_SHARE / 60.0  # per minute
    urgent_times, t = [], 0.0
    while urgent_rate > 0:
        t += rng.exponential(1 / urgent_rate)
        if t > 2000:
            break
        urgent_times.append(t)
    urgent_idx = 0
    ahead = queue_ahead

    while True:
        d = int(np.argmin(free_at))
        now = free_at[d]
        if urgent_idx < len(urgent_times) and urgent_times[urgent_idx] <= now:
            urgent_idx += 1          # an urgent patient takes this doctor
            free_at[d] = now + visit()
        elif ahead > 0:
            ahead -= 1               # next person in line ahead of you
            free_at[d] = now + visit()
        else:
            return now               # your turn


def make_dataset(n, rng):
    X = np.column_stack([
        rng.integers(1, 7, n),
        rng.integers(0, 16, n),
        rng.uniform(*RANGES["visit_minutes"], n),
        rng.uniform(*RANGES["walkins_per_hour"], n),
    ]).astype(float)
    y = np.array([simulate_wait(int(a), int(b), c, d, rng) for a, b, c, d in X])
    return X, y


# ---------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    X, y = make_dataset(N_SAMPLES, rng)

    (OUT / "data").mkdir(exist_ok=True)
    header = ",".join(FEATURES + ["wait_minutes"])
    np.savetxt(OUT / "data" / "clinic_waits.csv", np.column_stack([X, y]),
               delimiter=",", header=header, comments="", fmt="%.3f")

    # 70 / 15 / 15 split: train / validation (tuning) / test (touched once)
    idx = rng.permutation(N_SAMPLES)
    n_tr, n_va = int(0.70 * N_SAMPLES), int(0.15 * N_SAMPLES)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    x_scaler = StandardScaler().fit(X[tr])
    # train on log(1 + wait): waits are long-tailed, and the network fits
    # multiplicative effects (e.g. "half the doctors, double the wait") more easily
    ly = np.log1p(y)
    y_mean, y_std = ly[tr].mean(), ly[tr].std()
    Xs = x_scaler.transform(X)
    ys = (ly - y_mean) / y_std

    def to_minutes(z):
        return np.expm1(z * y_std + y_mean)

    # tune hidden-layer size on the validation set only
    candidates = [(4,), (8,), (16,), (32,), (16, 16)]
    best = None
    for h in candidates:
        m = MLPRegressor(hidden_layer_sizes=h, activation="relu", max_iter=2000,
                         early_stopping=True, random_state=SEED).fit(Xs[tr], ys[tr])
        mae = mean_absolute_error(y[va], to_minutes(m.predict(Xs[va])))
        print(f"hidden={h!s:9}  val MAE = {mae:6.2f} min")
        if best is None or mae < best[1]:
            best = (h, mae, m)
    hidden, _, model = best
    print(f"-> chose hidden={hidden}")

    # typical-range band from validation residuals (log scale): p10..p90
    res_va = np.log1p(y[va]) - (model.predict(Xs[va]) * y_std + y_mean)
    band = [float(np.percentile(res_va, 10)), float(np.percentile(res_va, 90))]
    print("log residual band p10/p90:", np.round(band, 3))

    # ---- test set (used once, for final numbers)
    pred_nn = np.maximum(to_minutes(model.predict(Xs[te])), 0)
    pred_median = np.full(len(te), np.median(y[tr]))
    lin = LinearRegression().fit(X[tr], y[tr])
    pred_lin = np.maximum(lin.predict(X[te]), 0)

    def metrics(p):
        return {"mae": round(float(mean_absolute_error(y[te], p)), 1),
                "r2": round(float(r2_score(y[te], p)), 3)}

    results = {
        "neural_net": metrics(pred_nn),
        "median_baseline": metrics(pred_median),
        "linear_regression": metrics(pred_lin),
        "median_train_wait": round(float(np.median(y[tr])), 1),
    }
    print(json.dumps(results, indent=2))

    # ---- biggest mistake on the test set
    err = pred_nn - y[te]
    k = int(np.argmax(np.abs(err)))
    xm = X[te][k]
    # re-run the simulator many times on the same inputs to show the spread
    rng2 = np.random.default_rng(7)
    reruns = np.array([simulate_wait(int(xm[0]), int(xm[1]), xm[2], xm[3], rng2)
                       for _ in range(500)])
    mistake = {
        "inputs": dict(zip(FEATURES, [int(xm[0]), int(xm[1]), round(xm[2], 1), round(xm[3], 1)])),
        "actual": round(float(y[te][k]), 1),
        "predicted": round(float(pred_nn[k]), 1),
        "rerun_median": round(float(np.median(reruns)), 1),
        "rerun_p10": round(float(np.percentile(reruns, 10)), 1),
        "rerun_p90": round(float(np.percentile(reruns, 90)), 1),
    }
    print("biggest mistake:", mistake)

    # ---- export for the browser
    export = {
        "features": FEATURES,
        "ranges": RANGES,
        "urgent_share": URGENT_SHARE,
        "hidden_layers": list(hidden),
        "x_mean": x_scaler.mean_.tolist(),
        "x_std": x_scaler.scale_.tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "target_transform": "log1p",
        "log_band_p10_p90": band,
        "weights": [w.tolist() for w in model.coefs_],   # [in][out]
        "biases": [b.tolist() for b in model.intercepts_],
        "n": {"total": N_SAMPLES, "train": len(tr), "val": len(va), "test": len(te)},
        "results": results,
        "mistake": mistake,
        "test_points": [[round(float(a), 1), round(float(p), 1)] + X[te][i].round(1).tolist()
                        for i, (a, p) in enumerate(zip(y[te], pred_nn))],
    }
    (OUT / "docs").mkdir(exist_ok=True)
    (OUT / "site" / "model.json").write_text(json.dumps(export))
    # sanity values for checking the JS forward pass
    check_x = np.array([[2, 10, 20, 10], [5, 3, 12, 15]], float)
    print("check preds:", to_minutes(model.predict(x_scaler.transform(check_x))).round(2))


if __name__ == "__main__":
    main()

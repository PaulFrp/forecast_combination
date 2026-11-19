"""End-to-end LSTM volatility pipeline.

    python LSTM/pipeline.py --mode quick   # tiny untuned LSTM, 1 seed, static fit
                                           # -> valid forecasts in the contract fast
    python LSTM/pipeline.py --mode full    # tuning on validation, 5 seeds,
                                           # expanding-window refits, full evaluation

Steps (see the module of each step):
  1-2  features.py    targets (log avg GK variance at h = 1, 5, 22) and lagged features
  3    sequences.py   train-only scaling, sliding windows, chronological split
  4    har.py         HAR baseline on the same split
  5    model.py       1-layer LSTM, Adam, MSE, early stopping, seed ensemble
  6    tuning.py      small search, validation set only
  7    forecast.py    test forecasts (static or expanding window)
  8    export.py      back to variance, bias correction, contract CSV
  9    evaluate.py    QLIKE / MSE vs HAR, regimes, seeds, feature importance
"""
import os

# Windows/conda: torch and numpy/MKL each ship libiomp5md.dll -> OMP Error #15 crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import config
import data_loader
import evaluate
import export
import features
import plots
from forecast import forecast_test
from har import HARModel
from model import LSTMEnsemble
from sequences import split_positions
from tuning import tune


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["quick", "full"], default="full")
    p.add_argument("--scheme", choices=["static", "expanding"], default=None,
                   help="test forecasting scheme (default: static in quick mode, config in full)")
    p.add_argument("--no-tune", action="store_true", help="full mode without the search (default params)")
    p.add_argument("--horizons", type=int, nargs="+", default=config.HORIZONS)
    return p.parse_args()


def val_report(name, y_log, pred_log, s2):
    rv = np.exp(y_log)
    f = export.to_variance(pred_log, s2)
    return {"model": name, "val_MSE_log": float(np.mean((pred_log - y_log) ** 2)),
            "val_QLIKE": float(evaluate.qlike(rv, f).mean()), "bias_s2": s2}


def run_horizon(h, data, feat_cols, splits, params, seeds, scheme, full):
    print(f"\n=== horizon h={h}  params={params}  seeds={seeds}  scheme={scheme}")
    pos = split_positions(data, h, params["window"], splits)
    y = data[f"target_{h}"].to_numpy()
    y_val = y[pos["val"]]
    print(f"  samples: train {len(pos['train'])}, val {len(pos['val'])}, test {len(pos['test'])}")

    # --- Step 4: HAR baseline on the same split ------------------------------
    har = HARModel(h).fit(data, pos["train"])
    har_val = har.predict(data, pos["val"])
    s2_har = export.residual_variance(har_val, y_val)

    # --- Step 5: LSTM trained on train, early-stopped on validation ----------
    t0 = time.time()
    ens = LSTMEnsemble(h, feat_cols, params, seeds=seeds).fit(data, pos["train"], pos["val"])
    lstm_val = ens.predict(data, pos["val"])
    s2_lstm = export.residual_variance(lstm_val, y_val)
    epochs = int(np.median(ens.best_epochs))
    print(f"  LSTM trained in {time.time() - t0:.0f}s, best epochs {ens.best_epochs}")

    val_rows = [{"horizon": h, **val_report("HAR", y_val, har_val, s2_har)},
                {"horizon": h, **val_report(config.MODEL_NAME, y_val, lstm_val, s2_lstm)}]
    for r in val_rows:
        print(f"  VALIDATION {r['model']:>5}: MSE(log) {r['val_MSE_log']:.4f}  QLIKE {r['val_QLIKE']:.4f}")
    if val_rows[1]["val_MSE_log"] > 1.05 * val_rows[0]["val_MSE_log"]:
        print("  !! LSTM is clearly worse than HAR on validation - check features / training.")

    importance = evaluate.permutation_importance(ens, data, pos["val"]) if full else None

    # --- Step 7: test forecasts ----------------------------------------------
    har_test = forecast_test(data, h, params["window"], splits, lambda: HARModel(h), scheme)
    lstm_test = forecast_test(data, h, params["window"], splits,
                              lambda: LSTMEnsemble(h, feat_cols, params, seeds=seeds, epochs=epochs),
                              scheme)

    # --- Step 8: back to variance with bias correction -----------------------
    dates = lstm_test.index
    var_lstm = export.to_variance(lstm_test["log_pred"], s2_lstm)
    var_har = export.to_variance(har_test.loc[dates, "log_pred"], s2_har)
    contract = {"lstm": export.contract_frame(dates, h, var_lstm, config.MODEL_NAME),
                "har": export.contract_frame(dates, h, var_har, "HAR")}

    # --- Step 9: evaluation on test dates whose target is observed -----------
    detail = pd.DataFrame({"horizon": h, "realized": np.exp(data.loc[dates, f"target_{h}"]),
                           "LSTM": var_lstm, "HAR": var_har}, index=dates)
    detail["crisis"] = evaluate.crisis_flag(data, dates, splits["train"][1])
    ok = detail["realized"].notna()
    d = detail[ok]
    metrics = evaluate.loss_table(d["realized"], {"HAR": d["HAR"], config.MODEL_NAME: d["LSTM"]},
                                  h, d["crisis"])
    seed_cols = [c for c in lstm_test.columns if c.startswith("seed_")]
    stability = evaluate.seed_stability(d["realized"], lstm_test.loc[ok, seed_cols], s2_lstm, h)

    return dict(val=val_rows, contract=contract, detail=detail, metrics=metrics,
                stability=stability, importance=importance, epochs=epochs)


def main():
    args = parse_args()
    full = args.mode == "full"
    scheme = args.scheme or (config.FORECAST_SCHEME if full else "static")
    seeds = config.SEEDS if full else config.QUICK_SEEDS
    out_dir = config.OUTPUT_DIR / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # --- Steps 1-2: data, target, features -----------------------------------
    df, splits = data_loader.load_all()
    data, feat_cols = features.build_dataset(df)
    print("Splits:", {k: f"{a.date()} -> {b.date()}" for k, (a, b) in splits.items()})
    print("Features:", feat_cols)

    # --- Step 6: tuning on validation only -----------------------------------
    if full and not args.no_tune:
        print("\n=== Tuning on validation set")
        best, tuning_results = tune(data, feat_cols, splits, args.horizons)
        tuning_results.to_csv(out_dir / "tuning_results.csv", index=False)
    else:
        best = {h: dict(config.DEFAULT_PARAMS) for h in args.horizons}
    (out_dir / "params.json").write_text(json.dumps(
        {"params": {str(h): p for h, p in best.items()}, "seeds": seeds, "scheme": scheme,
         "refit_freq": config.REFIT_FREQ if scheme == "expanding" else None,
         "features": feat_cols, "export_unit": config.EXPORT_UNIT, "annualized": config.ANNUALIZE},
        indent=2))

    results = {h: run_horizon(h, data, feat_cols, splits, best[h], seeds, scheme, full)
               for h in args.horizons}

    # --- Outputs ---------------------------------------------------------------
    print("\n=== Outputs")
    cat = lambda key: pd.concat([r[key] for r in results.values()], ignore_index=True)
    export.write_contract([r["contract"]["lstm"] for r in results.values()], out_dir / "forecasts_lstm.csv")
    export.write_contract([r["contract"]["har"] for r in results.values()], out_dir / "forecasts_har_baseline.csv")
    pd.DataFrame([row for r in results.values() for row in r["val"]]).to_csv(out_dir / "validation.csv", index=False)
    metrics = cat("metrics")
    metrics.to_csv(out_dir / "test_metrics.csv", index=False)
    cat("stability").to_csv(out_dir / "seed_stability.csv", index=False)
    pd.concat([r["detail"] for r in results.values()]).rename_axis("date").to_csv(out_dir / "test_detail.csv")
    if full:
        cat("importance").to_csv(out_dir / "feature_importance.csv", index=False)
    plots.make_all(results, out_dir)

    print("\n=== Test metrics (variance scale; ratio < 1 means LSTM beats HAR)")
    cols = ["horizon", "regime", "model", "n", "QLIKE", "MSE", "QLIKE_vs_HAR", "MSE_vs_HAR", "DM_QLIKE_p"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
        print(metrics[[c for c in cols if c in metrics]].to_string(index=False))
    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()

"""End-to-end LSTM volatility pipeline on the group's Alpaca SPY data.

    python LSTM/pipeline.py --mode quick   # tiny untuned LSTM, 1 seed, static fits
                                           # -> valid forecasts in the contract fast
    python LSTM/pipeline.py --mode full    # tuning on validation, 5 seeds,
                                           # expanding-window refits, full evaluation,
                                           # writes forecasts/{validation,test}/lstm.csv

Steps (see the module of each step):
  1-2  features.py    targets (log avg RV over the next 1, 5, 22 sessions) and features
  3    sequences.py   train-only scaling, sliding windows, matured-label filtering
  5    model.py       1-layer LSTM, Adam, MSE, early stopping, seed ensemble
  6    tuning.py      small search, validation set only
  7    forecast.py    walk-forward forecasts over the validation and test schedules
  8    export.py      back to variance, bias correction, contract CSVs
  9    evaluate.py    RMSE / QLIKE vs model_3 HAR-RV (forecasts/<phase>/har_rv.csv)
                      and persistence, regimes, seeds, importance
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
from forecast import forecast_phase
from model import LSTMEnsemble
from sequences import split_positions
from tuning import tune

PHASES = ("val", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["quick", "full"], default="full")
    p.add_argument("--scheme", choices=["static", "expanding"], default=None,
                   help="walk-forward scheme (default: static in quick mode, config in full)")
    p.add_argument("--no-tune", action="store_true", help="full mode without the search (default params)")
    p.add_argument("--horizons", type=int, nargs="+", default=config.HORIZONS)
    p.add_argument("--no-submit", action="store_true",
                   help="full mode: do not write the group files in forecasts/<phase>/")
    return p.parse_args()


def run_horizon(h, data, feat_cols, splits, params, seeds, scheme, full, schedules, actuals, benchmarks):
    print(f"\n=== horizon h={h}  params={params}  seeds={seeds}  scheme={scheme}")
    pos = split_positions(data, h, params["window"], splits)
    print(f"  initial fit samples: train {len(pos['train'])}, val {len(pos['val'])}")

    # --- Step 5: LSTM trained on train, early-stopped on validation ----------
    # Only used to pick the number of epochs (and for feature importance);
    # every submitted forecast comes from the walk-forward refits below.
    t0 = time.time()
    ens = LSTMEnsemble(h, feat_cols, params, seeds=seeds).fit(data, pos["train"], pos["val"])
    epochs = int(np.median(ens.best_epochs))
    print(f"  LSTM trained in {time.time() - t0:.0f}s, best epochs {ens.best_epochs}")
    importance = evaluate.permutation_importance(ens, data, pos["val"]) if full else None

    # --- Step 7: walk-forward LSTM forecasts on each phase's schedule ---------
    log_fc = {}
    for phase in PHASES:
        origins = schedules[phase].loc[schedules[phase].horizon == h, "origin_date"]
        print(f"  {phase}: {len(origins)} scheduled origins")
        log_fc[phase] = forecast_phase(data, h, params["window"], origins,
                                       lambda: LSTMEnsemble(h, feat_cols, params, seeds=seeds, epochs=epochs),
                                       scheme)

    # --- Step 8: back to variance; bias term from walk-forward validation ----
    y = data[f"target_{h}"]
    fc = log_fc["val"]
    ok = y.loc[fc.index].notna()
    s2 = export.residual_variance(fc.loc[ok, "log_pred"], y.loc[fc.index][ok])
    variance = {phase: pd.Series(export.to_variance(log_fc[phase]["log_pred"], s2), index=log_fc[phase].index)
                for phase in PHASES}
    contract = {phase: export.contract_frame(v.index, h, v.to_numpy(), config.MODEL_ID)
                for phase, v in variance.items()}

    # --- Step 9: scores against actuals.csv, vs model_3 HAR-RV and persistence,
    #             on the origins where all of them are present ------------------
    bench = config.BENCHMARK_NAME
    details, metrics = {}, {}
    for phase in PHASES:
        a = actuals[phase].loc[actuals[phase].horizon == h].set_index("origin_date")
        b = benchmarks[phase].loc[benchmarks[phase].horizon == h].set_index("origin_date")["prediction"]
        dates = log_fc[phase].index
        detail = pd.DataFrame({"horizon": h, "realized": a["actual"].reindex(dates),
                               "persistence": a["baseline"].reindex(dates),
                               "LSTM": variance[phase], bench: b.reindex(dates)},
                              index=dates)
        detail["crisis"] = evaluate.crisis_flag(data, dates, splits["train"][1])
        details[phase] = detail
        d = detail.dropna(subset=["realized", "persistence", "LSTM", bench])
        metrics[phase] = evaluate.loss_table(
            d["realized"], {"persistence": d["persistence"], bench: d[bench], config.MODEL_NAME: d["LSTM"]},
            h, d["crisis"]).assign(phase=phase,
                                  bias_s2=lambda t: t.model.map({config.MODEL_NAME: s2}))
        all_rows = metrics[phase][metrics[phase].regime == "all"].set_index("model")
        print(f"  {phase.upper():>4} RMSE  persistence {all_rows.RMSE['persistence']:.3e}  "
              f"{bench} {all_rows.RMSE[bench]:.3e}  LSTM {all_rows.RMSE[config.MODEL_NAME]:.3e}  (n={len(d)})")

    d = details["test"].dropna(subset=["realized", bench, "LSTM"])
    seed_cols = [c for c in log_fc["test"].columns if c.startswith("seed_")]
    stability = evaluate.seed_stability(d["realized"], log_fc["test"].loc[d.index, seed_cols],
                                        s2, h, d[bench], d["crisis"])

    return dict(contract=contract, detail=details["test"], val_detail=details["val"],
                metrics=metrics, stability=stability, importance=importance, epochs=epochs)


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
    schedules = {p: data_loader.load_schedule(p) for p in PHASES}
    schedules = {p: s[s.horizon.isin(args.horizons)].reset_index(drop=True) for p, s in schedules.items()}
    actuals = {p: data_loader.load_actuals(p) for p in PHASES}
    benchmarks = {p: data_loader.load_benchmark(p, schedules[p]) for p in PHASES}
    print("Splits:", {k: f"{a.date()} -> {b.date()}" for k, (a, b) in splits.items()})
    print("Features:", feat_cols)

    # --- Step 6: tuning on validation only -----------------------------------
    if full and not args.no_tune:
        print("\n=== Tuning on validation set")
        best, tuning_results = tune(data, feat_cols, splits, args.horizons)
        tuning_results.to_csv(out_dir / "tuning_results.csv", index=False)
    else:
        best = {h: dict(config.DEFAULT_PARAMS) for h in args.horizons}

    results = {h: run_horizon(h, data, feat_cols, splits, best[h], seeds, scheme, full, schedules, actuals,
                          benchmarks)
               for h in args.horizons}
    (out_dir / "params.json").write_text(json.dumps(
        {"params": {str(h): p for h, p in best.items()},
         "refit_epochs": {str(h): r["epochs"] for h, r in results.items()},
         "seeds": seeds, "scheme": scheme,
         "refit_freq": config.REFIT_FREQ if scheme == "expanding" else None,
         "features": feat_cols, "vix_lag": config.VIX_LAG, "bias_correction": config.BIAS_CORRECTION},
        indent=2))

    # --- Outputs ---------------------------------------------------------------
    print("\n=== Outputs")
    cat = lambda key: pd.concat([r[key] for r in results.values()], ignore_index=True)
    submit = full and not args.no_submit and sorted(args.horizons) == sorted(config.HORIZONS)
    for phase in PHASES:
        folder = config.PHASES[phase]
        lstm = [r["contract"][phase] for r in results.values()]
        export.write_contract(lstm, out_dir / "forecasts" / folder / f"{config.MODEL_ID}.csv", schedules[phase])
        if submit:
            export.write_contract(lstm, config.FORECASTS_DIR / folder / f"{config.MODEL_ID}.csv",
                                  schedules[phase])

    metrics = pd.concat([r["metrics"][p] for r in results.values() for p in PHASES], ignore_index=True)
    metrics[metrics.phase == "val"].to_csv(out_dir / "validation.csv", index=False)
    metrics[metrics.phase == "test"].to_csv(out_dir / "test_metrics.csv", index=False)
    cat("stability").to_csv(out_dir / "seed_stability.csv", index=False)
    pd.concat([r["detail"] for r in results.values()]).rename_axis("origin_date").to_csv(out_dir / "test_detail.csv")
    if full:
        cat("importance").to_csv(out_dir / "feature_importance.csv", index=False)
    plots.make_all(results, out_dir)

    cols = ["horizon", "regime", "model", "n", "RMSE", "RMSE_vs_pers", "RMSE_vs_HAR", "QLIKE",
            "QLIKE_vs_HAR", "DM_SE_pers_p"]
    for phase in PHASES:
        print(f"\n=== {phase} metrics (variance scale; ratio < 1 means better than the reference)")
        m = metrics[(metrics.phase == phase) & (metrics.regime == "all")]
        with pd.option_context("display.width", 180, "display.float_format", "{:.4g}".format):
            print(m[[c for c in cols if c in m]].to_string(index=False))
    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()

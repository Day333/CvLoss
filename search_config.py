import re
import argparse
import pandas as pd
import numpy as np

VALID_PATCH = {3, 6, 12, 24, 48}
EXPECTED_PREDS = [96, 192, 336, 720]


def parse_line(header_line, metric_line):
    header_line = header_line.strip()
    metric_line = metric_line.strip()

    if "addloss" not in header_line:
        return None

    metric_match = re.search(r"mse:([\d.eE+-]+),\s*(?:mae):([\d.eE+-]+)", metric_line)
    if metric_match is None:
        return None

    mse = float(metric_match.group(1))
    mae = float(metric_match.group(2))

    parts = header_line.split("_")
    dataset = parts[3]
    input_len = int(parts[4])
    pred_len = int(parts[5])

    alpha_match = re.search(r"alpha([0-9\.]+)", header_line)
    beta_match = re.search(r"beta([0-9\.]+)", header_line)
    patch_match = re.search(r"patch(?:len)?_?([0-9]+)", header_line)

    if not (alpha_match and beta_match and patch_match):
        return None

    patch = int(patch_match.group(1))
    if patch not in VALID_PATCH:
        return None

    beta = float(beta_match.group(1))

    return {
        "dataset": dataset,
        "input_len": input_len,
        "pred_len": pred_len,
        "patch": patch,
        "alpha": float(alpha_match.group(1)),
        "beta": beta,
        "mse": mse,
        "mae": mae,
    }


def parse_file(filepath):
    results = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("long_term_forecast"):
            header = lines[i]
            metric = lines[i + 1] if i + 1 < len(lines) else ""
            parsed = parse_line(header, metric)
            if parsed:
                results.append(parsed)
            i += 2
        else:
            i += 1

    return pd.DataFrame(results)


def analyze(df: pd.DataFrame, wide_mode: str = "full", metric: str = "mse"):
    assert wide_mode in {"full", "avg"}, "wide_mode must be 'full' or 'avg'"
    assert metric in {"mse", "mae", "mse+mae"}, "metric must be 'mse', 'mae', or 'mse+mae'"

    for dataset in sorted(df["dataset"].unique()):

        print("\n" + "=" * 120)
        print(f"DATASET: {dataset}")
        print("=" * 120)

        df_dataset = df[df["dataset"] == dataset]

        for input_len in sorted(df_dataset["input_len"].unique()):

            print("\n" + "#" * 120)
            print(f"INPUT_LEN: {input_len}")
            print("#" * 120)

            df_input = df_dataset[df_dataset["input_len"] == input_len]

            pred_list = sorted(df_input["pred_len"].unique())

            # ============================================================
            # Global best per pred_len
            # ============================================================
            print(f"\nGlobal Best per pred_len (across patch & beta, by {metric}):")

            best_rows = []

            for pl in pred_list:
                df_pl = df_input[df_input["pred_len"] == pl]
                if df_pl.empty:
                    continue

                if metric == "mse":
                    target_col = df_pl["mse"]
                elif metric == "mae":
                    target_col = df_pl["mae"]
                else:
                    target_col = df_pl["mse"] + df_pl["mae"]

                idx = target_col.idxmin()
                best = df_pl.loc[idx]

                print(
                    f"pred_len={pl} → "
                    f"target({metric})={target_col.loc[idx]:.6f} (mse={best['mse']:.6f}, mae={best['mae']:.6f}), "
                    f"patch={best['patch']}, "
                    f"beta={best['beta']}"
                )

                best_rows.append(best)

            if best_rows:
                best_df = pd.DataFrame(best_rows)
                print(f"\nAverage of best models across pred_len:")
                print(f"avg_best_mse = {best_df['mse'].mean():.6f}")
                print(f"avg_best_mae = {best_df['mae'].mean():.6f}")

            # ============================================================
            # patch 
            # ============================================================
            for patch in sorted(df_input["patch"].unique()):

                print("\n" + "-" * 120)
                print(f"PATCH = {patch}")
                print("-" * 120)

                df_patch = df_input[df_input["patch"] == patch].copy()

                mse_wide = df_patch.pivot_table(
                    index="beta", columns="pred_len", values="mse", aggfunc="mean"
                )
                mae_wide = df_patch.pivot_table(
                    index="beta", columns="pred_len", values="mae", aggfunc="mean"
                )

                for pl in pred_list:
                    if pl not in mse_wide.columns:
                        mse_wide[pl] = np.nan
                    if pl not in mae_wide.columns:
                        mae_wide[pl] = np.nan

                mse_wide = mse_wide[pred_list]
                mae_wide = mae_wide[pred_list]

                mse_wide.columns = [f"mse_{pl}" for pl in pred_list]
                mae_wide.columns = [f"mae_{pl}" for pl in pred_list]

                combined = pd.concat([mse_wide, mae_wide], axis=1)

                combined["avg_mse"] = combined[
                    [f"mse_{pl}" for pl in pred_list]
                ].mean(axis=1, skipna=True)

                combined["avg_mae"] = combined[
                    [f"mae_{pl}" for pl in pred_list]
                ].mean(axis=1, skipna=True)

                combined["coverage"] = (
                    combined[[f"mse_{pl}" for pl in pred_list]]
                    .notna()
                    .sum(axis=1)
                    .astype(str)
                    + f"/{len(pred_list)}"
                )

                combined = combined.sort_index()

                if wide_mode == "avg":
                    to_print = combined[["avg_mse", "avg_mae", "coverage"]]
                else:
                    to_print = combined

                print("\nWide table:")
                print(to_print.to_string())

                # best beta per pred_len
                print(f"\nBest beta per pred_len (by {metric}):")
                for pl in pred_list:
                    col_mse = f"mse_{pl}"
                    col_mae = f"mae_{pl}"
                    
                    # 动态选择评判标准 2
                    if metric == "mse":
                        target_series = combined[col_mse]
                    elif metric == "mae":
                        target_series = combined[col_mae]
                    else:
                        target_series = combined[col_mse] + combined[col_mae]
                        
                    target_series = target_series.dropna()
                    if target_series.empty:
                        continue
                        
                    best_beta = target_series.idxmin()
                    best_val = target_series.min()
                    best_mse = combined.loc[best_beta, col_mse]
                    best_mae = combined.loc[best_beta, col_mae]
                    
                    print(f"pred_len={pl} → beta={best_beta}, target({metric})={best_val:.6f} (mse={best_mse:.6f}, mae={best_mae:.6f})")

                if metric == "mse":
                    target_global = combined["avg_mse"]
                elif metric == "mae":
                    target_global = combined["avg_mae"]
                else:
                    target_global = combined["avg_mse"] + combined["avg_mae"]
                    
                best_global = target_global.idxmin()
                best_global_val = target_global.min()
                
                print(f"\nOverall best beta (by avg_{metric}):")
                print(
                    f"beta={best_global}, "
                    f"target({metric})={best_global_val:.6f}, "
                    f"avg_mse={combined.loc[best_global, 'avg_mse']:.6f}, "
                    f"avg_mae={combined.loc[best_global, 'avg_mae']:.6f}, "
                    f"coverage={combined.loc[best_global, 'coverage']}"
                )

        print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument(
        "--wide_mode",
        type=str,
        default="avg",
        choices=["full", "avg"],
        help="Wide table output: 'full' prints per-pred columns; 'avg' prints only avg_mse/avg_mae/coverage.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="mse",
        choices=["mse", "mae", "mse+mae"],
        help="Evaluation metric for selecting the best model.",
    )
    args = parser.parse_args()

    df = parse_file(args.file)

    if df.empty:
        print("No valid addloss experiments found.")
        return

    analyze(df, wide_mode=args.wide_mode, metric=args.metric)


if __name__ == "__main__":
    main()
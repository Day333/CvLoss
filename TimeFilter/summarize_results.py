import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    df["seq_len"] = df["seq_len"].astype(int)
    df["pred_len"] = df["pred_len"].astype(int)
    df["mse"] = df["mse"].astype(float)
    df["mae"] = df["mae"].astype(float)

    seq_order = [96, 192, 336, 720]
    pred_order = [96, 192, 336, 720]
    model_order = ["CvLoss", "TimeFilter"]

    rows = []
    index = []

    for seq_len in seq_order:
        sub = df[df["seq_len"] == seq_len]

        for pred_len in pred_order:
            row = []
            cur = sub[sub["pred_len"] == pred_len]

            for model in model_order:
                x = cur[cur["model_name"] == model]
                if len(x) == 0:
                    row.extend([None, None])
                else:
                    row.extend([
                        round(float(x.iloc[0]["mse"]), 3),
                        round(float(x.iloc[0]["mae"]), 3)
                    ])

            rows.append(row)
            index.append((seq_len, pred_len))

        avg_row = []
        for model in model_order:
            x = sub[sub["model_name"] == model]
            if len(x) == 0:
                avg_row.extend([None, None])
            else:
                avg_row.extend([
                    round(x["mse"].mean(), 3),
                    round(x["mae"].mean(), 3)
                ])

        rows.append(avg_row)
        index.append((seq_len, "Avg"))

    columns = pd.MultiIndex.from_tuples([
        ("CvLoss", "MSE"),
        ("CvLoss", "MAE"),
        ("TQNet", "MSE"),
        ("TQNet", "MAE"),
    ])

    result_df = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["Historical sequence length", "Prediction length"]),
        columns=columns
    )

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)

    print("\n================ Final Result Table ================\n")
    print(result_df.to_string())
    print("\n===================================================\n")

if __name__ == "__main__":
    main()
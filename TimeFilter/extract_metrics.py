import re
import csv
import argparse
from pathlib import Path

def extract_last_mse_mae(text: str):
    patterns = [
        re.compile(r"mse\s*[:=]\s*([0-9]*\.?[0-9]+)\s*[, ]+\s*mae\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I),
        re.compile(r"mae\s*[:=]\s*([0-9]*\.?[0-9]+)\s*[, ]+\s*mse\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I),
    ]

    matches = []
    for p in patterns:
        matches.extend(p.findall(text))

    if not matches:
        raise ValueError("No mse/mae pair found in log.")

    last = matches[-1]

    if "mae" in patterns[1].pattern.lower():
        pass

    pairs = re.findall(
        r"mse\s*[:=]\s*([0-9]*\.?[0-9]+).*?mae\s*[:=]\s*([0-9]*\.?[0-9]+)",
        text,
        flags=re.I | re.S
    )
    if pairs:
        mse, mae = pairs[-1]
        return float(mse), float(mae)

    pairs_rev = re.findall(
        r"mae\s*[:=]\s*([0-9]*\.?[0-9]+).*?mse\s*[:=]\s*([0-9]*\.?[0-9]+)",
        text,
        flags=re.I | re.S
    )
    if pairs_rev:
        mae, mse = pairs_rev[-1]
        return float(mse), float(mae)

    raise ValueError("No valid mse/mae pair found in log.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--add_loss", required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    args = parser.parse_args()

    log_text = Path(args.log_file).read_text(encoding="utf-8", errors="ignore")
    mse, mae = extract_last_mse_mae(log_text)

    with open(args.csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            args.model_name,
            args.add_loss,
            args.seq_len,
            args.pred_len,
            f"{mse:.3f}",
            f"{mae:.3f}",
            args.log_file,
        ])

    print(f"[OK] {args.model_name} | seq_len={args.seq_len} | pred_len={args.pred_len} | mse={mse:.3f} | mae={mae:.3f}")

if __name__ == "__main__":
    main()
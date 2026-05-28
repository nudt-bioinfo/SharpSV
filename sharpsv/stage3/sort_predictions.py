import argparse
import ast
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_CSV = str(Path("workdir") / "stage2_predictions.csv")
DEFAULT_OUTPUT_CSV = str(Path("workdir") / "stage3_refined_sv_results.csv")
OUTPUT_COLUMNS = ["chrom", "start", "end", "label"]


def parse_chrom(chrom):
    chrom = str(chrom).upper().replace("CHR", "")
    if chrom == "X":
        return 23
    if chrom == "Y":
        return 24
    if chrom in {"M", "MT"}:
        return 25
    try:
        return int(chrom)
    except ValueError:
        return 999


def extract_intervals(row):
    try:
        seq = ast.literal_eval(row["pred_sequence"])
    except Exception:
        return []

    if sum(seq) == 0:
        return []

    base_start = int(row["start"])
    window_results = []
    current_label = 0
    start_idx = -1

    for index, label in enumerate(seq):
        if label != current_label:
            if current_label != 0:
                sv_start = base_start + (start_idx * 50)
                sv_end = base_start + (index * 50)
                window_results.append(
                    {
                        "chrom": row["chrom"],
                        "start": sv_start,
                        "end": sv_end,
                        "sv_type": "DEL" if current_label == 1 else "INS",
                        "label": current_label,
                        "source_window_start": base_start,
                    }
                )

            current_label = label
            start_idx = index

    if current_label != 0:
        sv_start = base_start + (start_idx * 50)
        sv_end = base_start + (len(seq) * 50)
        window_results.append(
            {
                "chrom": row["chrom"],
                "start": sv_start,
                "end": sv_end,
                "sv_type": "DEL" if current_label == 1 else "INS",
                "label": current_label,
                "source_window_start": base_start,
            }
        )

    return window_results


def refine_predictions_csv(input_csv=DEFAULT_INPUT_CSV, output_csv=DEFAULT_OUTPUT_CSV, logger=print):
    input_path = Path(input_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger(f"reading stage-2 prediction artifact from {input_path}")
    df = pd.read_csv(input_path)
    if df.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_path, index=False)
        logger(f"stage-3 sort received an empty input CSV; wrote {output_path}")
        return str(output_path)

    df = df[df["pred_sequence"].astype(str).str.contains("1|2", regex=True, na=False)]
    logger(f"filtered all-zero windows: {len(df)} candidate windows remain")

    final_records = []
    for row in df.itertuples(index=False):
        final_records.extend(extract_intervals(row._asdict()))

    if not final_records:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_path, index=False)
        logger(f"no SV intervals remained after sequence decoding; wrote {output_path}")
        return str(output_path)

    result_df = pd.DataFrame(final_records)
    result_df["chrom_sort_key"] = result_df["chrom"].apply(parse_chrom)
    result_df = result_df.sort_values(by=["chrom_sort_key", "start"]).drop(columns=["chrom_sort_key"])
    result_df[OUTPUT_COLUMNS].to_csv(output_path, index=False)

    logger(f"decoded {len(result_df)} refined intervals")
    logger(f"saved refined intervals to {output_path}")
    return str(output_path)


def build_parser():
    parser = argparse.ArgumentParser(description="Decode SharpSV stage-2 sequence predictions into SV intervals")
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV, help="Input stage-2 prediction artifact CSV")
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="Output decoded stage-3 interval CSV")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    refine_predictions_csv(args.input_csv, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse

from .pipeline import export_stage3_csv_to_vcf


def build_parser():
    parser = argparse.ArgumentParser(description="Convert SharpSV validated CSV into an uncompressed VCF")
    parser.add_argument("-i", "--input_csv", default="./final_adaptive_validated.csv", help="Input validated CSV")
    parser.add_argument("-r", "--ref_fasta", required=True, help="Reference FASTA path")
    parser.add_argument("-o", "--output_vcf", default="./final_adaptive_validated.vcf", help="Output uncompressed VCF path")
    parser.add_argument("-s", "--sample_name", default="SharpSV", help="VCF sample name")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    written = export_stage3_csv_to_vcf(
        input_csv=args.input_csv,
        output_vcf=args.output_vcf,
        ref_fasta_path=args.ref_fasta,
        sample_name=args.sample_name,
    )
    print(f"[SUCCESS] wrote {written} records to {args.output_vcf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

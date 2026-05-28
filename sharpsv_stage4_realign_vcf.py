def main(argv=None):
    from sharpsv.stage4.realign_vcf import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

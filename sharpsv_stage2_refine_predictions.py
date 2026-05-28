def build_parser():
    from sharpsv.stage2.refine import build_parser as _build_parser

    return _build_parser()


def refine_intermediate_csv(*args, **kwargs):
    from sharpsv.stage2.refine import refine_intermediate_csv as _refine_intermediate_csv

    return _refine_intermediate_csv(*args, **kwargs)


def main(argv=None):
    from sharpsv.stage2.refine import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

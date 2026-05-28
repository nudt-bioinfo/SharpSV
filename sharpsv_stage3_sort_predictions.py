def build_parser():
    from sharpsv.stage3.sort_predictions import build_parser as _build_parser

    return _build_parser()


def refine_predictions_csv(*args, **kwargs):
    from sharpsv.stage3.sort_predictions import refine_predictions_csv as _refine_predictions_csv

    return _refine_predictions_csv(*args, **kwargs)


def main(argv=None):
    from sharpsv.stage3.sort_predictions import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

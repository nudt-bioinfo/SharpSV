def build_parser():
    from sharpsv.stage1.predict import build_parser as _build_parser

    return _build_parser()


def predict_workdir(*args, **kwargs):
    from sharpsv.stage1.predict import predict_workdir as _predict_workdir

    return _predict_workdir(*args, **kwargs)


def main(argv=None):
    from sharpsv.stage1.predict import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

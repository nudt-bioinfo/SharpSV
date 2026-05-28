def build_parser():
    from sharpsv.stage3.assembly_validator import build_parser as _build_parser

    return _build_parser()


def validate_assembly_candidates(*args, **kwargs):
    from sharpsv.stage3.assembly_validator import validate_assembly_candidates as _validate_assembly_candidates

    return _validate_assembly_candidates(*args, **kwargs)


def main(argv=None):
    from sharpsv.stage3.assembly_validator import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

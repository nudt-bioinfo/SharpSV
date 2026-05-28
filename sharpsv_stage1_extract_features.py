def build_parser():
    from sharpsv.stage1.features import build_parser as _build_parser

    return _build_parser()


def available_worker_count():
    from sharpsv.stage1.features import available_worker_count as _available_worker_count

    return _available_worker_count()


def baseinfo_main(*args, **kwargs):
    from sharpsv.stage1.features import baseinfo_main as _baseinfo_main

    return _baseinfo_main(*args, **kwargs)


def inspect_stage1_workdir(*args, **kwargs):
    from sharpsv.stage1.features import inspect_stage1_workdir as _inspect_stage1_workdir

    return _inspect_stage1_workdir(*args, **kwargs)


def write_stage1_completion_marker(*args, **kwargs):
    from sharpsv.stage1.features import write_stage1_completion_marker as _write_stage1_completion_marker

    return _write_stage1_completion_marker(*args, **kwargs)


def main(argv=None):
    from sharpsv.stage1.features import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

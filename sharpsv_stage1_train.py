def main(argv=None):
    from sharpsv.stage1.train import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

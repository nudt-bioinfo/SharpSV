from pathlib import Path

from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py as _build_py


class CleanBundledBuildPy(_build_py):
    """Keep repeated local builds from re-packaging stale reconstructed models."""

    def run(self):
        super().run()
        project_root = Path(__file__).resolve().parent
        source_models = project_root / "sharpsv" / "_bundle" / "models"
        built_models = Path(self.build_lib) / "sharpsv" / "_bundle" / "models"
        stale_stage2 = built_models / "stage2.model.bin"
        if stale_stage2.exists() and not (source_models / "stage2.model.bin").exists():
            stale_stage2.unlink()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    distclass=BinaryDistribution,
    cmdclass={"build_py": CleanBundledBuildPy},
)

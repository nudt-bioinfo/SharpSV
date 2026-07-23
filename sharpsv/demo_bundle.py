import json
from pathlib import Path


DEMO_BAM_BASENAME = "demo.bam"
DEMO_BAM_INDEX_BASENAME = "demo.bam.bai"
DEMO_FASTA_BASENAME = "demo.fa"
DEMO_FASTA_INDEX_BASENAME = "demo.fa.fai"
DEMO_BWA_INDEX_BASENAMES = [
    "demo.fa.amb",
    "demo.fa.ann",
    "demo.fa.bwt",
    "demo.fa.pac",
    "demo.fa.sa",
]
DEMO_METADATA_BASENAME = "demo_region.json"


def bundled_demo_dir():
    return Path(__file__).resolve().parent / "_bundle" / "demo"


def bundled_demo_metadata_path():
    return bundled_demo_dir() / DEMO_METADATA_BASENAME


def bundled_demo_bam_path():
    return bundled_demo_dir() / DEMO_BAM_BASENAME


def bundled_demo_fasta_path():
    return bundled_demo_dir() / DEMO_FASTA_BASENAME


def load_bundled_demo_metadata():
    with open(bundled_demo_metadata_path()) as handle:
        return json.load(handle)


def required_demo_paths():
    demo_dir = bundled_demo_dir()
    return [
        demo_dir / DEMO_BAM_BASENAME,
        demo_dir / DEMO_BAM_INDEX_BASENAME,
        demo_dir / DEMO_FASTA_BASENAME,
        demo_dir / DEMO_FASTA_INDEX_BASENAME,
        demo_dir / DEMO_METADATA_BASENAME,
        *[demo_dir / basename for basename in DEMO_BWA_INDEX_BASENAMES],
    ]


def validate_bundled_demo():
    missing = [path for path in required_demo_paths() if not path.exists()]
    if missing:
        missing_display = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Bundled demo assets are incomplete: {missing_display}")
    return True


def resolve_bundled_demo_inputs():
    validate_bundled_demo()
    return {
        "demo_dir": str(bundled_demo_dir()),
        "bam_path": str(bundled_demo_bam_path()),
        "fasta_path": str(bundled_demo_fasta_path()),
        "metadata": load_bundled_demo_metadata(),
    }

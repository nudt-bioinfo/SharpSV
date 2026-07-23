# SharpSV Bundled Demo

This directory contains the packaged `SharpSV --demo` input bundle.

- `demo.bam` and `demo.bam.bai`: a small HG002-derived BAM slice
- `demo.fa` plus `.fai` and BWA indexes: the synthetic 100 kb demo reference used by stage-3 local assembly
- `demo_region.json`: the mapping between the original HG002 source region and the local coordinates in the demo reference

The demo reference uses local coordinates on a single 100 kb contig named `1`. Final VCF positions from `--demo` are therefore relative to `demo.fa`, not to the full hs37d5 reference.

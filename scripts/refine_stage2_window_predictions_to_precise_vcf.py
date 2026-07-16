#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pysam


WINDOW_SIZE = 1000
DEFAULT_FLANK = 1200
MIN_MAPQ = 20
MIN_SOFTCLIP = 20
MIN_SPLIT_GAP = 20

_WORKER_BAM = None
_WORKER_FASTA = None
_WORKER_INSERT_MEAN = None
_WORKER_INSERT_STD = None
_WORKER_DEFAULT_FLANK = None


@dataclass
class EventCluster:
    event_id: str
    chrom: str
    svtype: str
    cluster_start: int
    cluster_end: int
    n_windows: int
    max_stage1_score: float
    mean_stage1_score: float
    max_confidence: float
    mean_confidence: float

    @property
    def midpoint(self) -> int:
        return int(round((self.cluster_start + self.cluster_end) / 2.0))


@dataclass
class AlignmentSegment:
    chrom: str
    ref_start1: int
    ref_end1: int
    strand: str
    qstart: int
    qend: int
    mapq: int
    source: str

    @property
    def entry_pos(self) -> int:
        return self.ref_start1 if self.strand == "+" else self.ref_end1

    @property
    def exit_pos(self) -> int:
        return self.ref_end1 if self.strand == "+" else self.ref_start1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Refine clustered stage-2 window-level SV predictions into breakpoint-level events "
            "using split-read, soft-clip, SA-tag, and discordant-pair evidence."
        )
    )
    parser.add_argument("--predictions", required=True, help="CSV from export_stage2_genomewide_vcf.py")
    parser.add_argument("--bam", required=True, help="Sorted/indexed BAM")
    parser.add_argument("--fasta", required=True, help="Reference FASTA")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--sample-name", default="SharpSV", help="Sample name written into the VCF")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 4) // 2, 1), help="CPU worker count")
    parser.add_argument("--cluster-gap", type=int, default=WINDOW_SIZE, help="Max gap for merging adjacent same-label windows")
    parser.add_argument("--flank", type=int, default=DEFAULT_FLANK, help="Extra search flank around each clustered event")
    parser.add_argument("--min-mapq", type=int, default=MIN_MAPQ, help="Minimum mapping quality")
    parser.add_argument("--min-softclip", type=int, default=MIN_SOFTCLIP, help="Minimum soft-clip length")
    parser.add_argument("--truth-bed", help="Optional BED-like truth file for summary evaluation only")
    return parser.parse_args()


def parse_chrom_key(chrom: str):
    c = str(chrom).replace("chr", "").replace("CHR", "")
    if c == "X":
        return 23
    if c == "Y":
        return 24
    if c in {"M", "MT"}:
        return 25
    try:
        return int(c)
    except ValueError:
        return 999


def sort_chrom_records(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda row: (parse_chrom_key(row["chrom"]), int(row["cluster_start"])))


def estimate_insert_stats(bam_path: str, sample_size: int = 100000):
    inserts = []
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if len(inserts) >= sample_size:
                break
            if (
                read.is_unmapped
                or read.is_duplicate
                or not read.is_paired
                or read.mate_is_unmapped
                or read.mapping_quality < MIN_MAPQ
            ):
                continue
            if read.reference_id != read.next_reference_id:
                continue
            tlen = abs(read.template_length)
            if 50 < tlen < 5000:
                inserts.append(tlen)
    if not inserts:
        return 500.0, 30.0
    return float(statistics.median(inserts)), float(statistics.pstdev(inserts) or 30.0)


def load_prediction_clusters(path: str, cluster_gap: int) -> list[EventCluster]:
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "chrom": str(row["chrom"]),
                    "window_start": int(row["window_start"]),
                    "window_end": int(row["window_end"]),
                    "svtype": str(row["pred_label"]),
                    "stage1_score": float(row["stage1_score"]),
                    "pred_confidence": float(row["pred_confidence"]),
                }
            )
    rows.sort(key=lambda row: (parse_chrom_key(row["chrom"]), row["window_start"]))

    clusters = []
    cluster_idx = 1
    current = None
    for row in rows:
        if current is None:
            current = {
                "chrom": row["chrom"],
                "svtype": row["svtype"],
                "cluster_start": row["window_start"],
                "cluster_end": row["window_end"],
                "stage1_scores": [row["stage1_score"]],
                "confidences": [row["pred_confidence"]],
                "n_windows": 1,
            }
            continue

        same_group = (
            row["chrom"] == current["chrom"]
            and row["svtype"] == current["svtype"]
            and row["window_start"] <= current["cluster_end"] + cluster_gap
        )
        if same_group:
            current["cluster_end"] = max(current["cluster_end"], row["window_end"])
            current["stage1_scores"].append(row["stage1_score"])
            current["confidences"].append(row["pred_confidence"])
            current["n_windows"] += 1
        else:
            clusters.append(
                EventCluster(
                    event_id=f"EV{cluster_idx:06d}",
                    chrom=current["chrom"],
                    svtype=current["svtype"],
                    cluster_start=current["cluster_start"],
                    cluster_end=current["cluster_end"],
                    n_windows=current["n_windows"],
                    max_stage1_score=max(current["stage1_scores"]),
                    mean_stage1_score=sum(current["stage1_scores"]) / len(current["stage1_scores"]),
                    max_confidence=max(current["confidences"]),
                    mean_confidence=sum(current["confidences"]) / len(current["confidences"]),
                )
            )
            cluster_idx += 1
            current = {
                "chrom": row["chrom"],
                "svtype": row["svtype"],
                "cluster_start": row["window_start"],
                "cluster_end": row["window_end"],
                "stage1_scores": [row["stage1_score"]],
                "confidences": [row["pred_confidence"]],
                "n_windows": 1,
            }

    if current is not None:
        clusters.append(
            EventCluster(
                event_id=f"EV{cluster_idx:06d}",
                chrom=current["chrom"],
                svtype=current["svtype"],
                cluster_start=current["cluster_start"],
                cluster_end=current["cluster_end"],
                n_windows=current["n_windows"],
                max_stage1_score=max(current["stage1_scores"]),
                mean_stage1_score=sum(current["stage1_scores"]) / len(current["stage1_scores"]),
                max_confidence=max(current["confidences"]),
                mean_confidence=sum(current["confidences"]) / len(current["confidences"]),
            )
        )
    return clusters


def _init_worker(bam_path: str, fasta_path: str, insert_mean: float, insert_std: float, default_flank: int):
    global _WORKER_BAM, _WORKER_FASTA, _WORKER_INSERT_MEAN, _WORKER_INSERT_STD, _WORKER_DEFAULT_FLANK
    _WORKER_BAM = pysam.AlignmentFile(bam_path, "rb")
    _WORKER_FASTA = pysam.FastaFile(fasta_path)
    _WORKER_INSERT_MEAN = insert_mean
    _WORKER_INSERT_STD = max(insert_std, 1.0)
    _WORKER_DEFAULT_FLANK = default_flank


_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")


def parse_cigar_string(cigar: str):
    return [(int(length), op) for length, op in _CIGAR_RE.findall(cigar)]


def query_bounds_from_ops(ops: list[tuple[int, str]], strand: str):
    total = 0
    left_clip = 0
    right_clip = 0
    i = 0
    while i < len(ops) and ops[i][1] in {"S", "H"}:
        left_clip += ops[i][0]
        i += 1
    j = len(ops) - 1
    while j >= 0 and ops[j][1] in {"S", "H"}:
        right_clip += ops[j][0]
        j -= 1
    for length, op in ops:
        if op in {"M", "I", "S", "=", "X", "H"}:
            total += length
    qstart = left_clip
    qend = total - right_clip
    if strand == "-":
        qstart, qend = total - qend, total - qstart
    return qstart, qend, total


def ref_length_from_ops(ops: list[tuple[int, str]]):
    return sum(length for length, op in ops if op in {"M", "D", "N", "=", "X"})


def read_to_primary_segment(read) -> AlignmentSegment | None:
    if read.is_unmapped or read.cigarstring is None:
        return None
    strand = "-" if read.is_reverse else "+"
    ops = parse_cigar_string(read.cigarstring)
    qstart, qend, _ = query_bounds_from_ops(ops, strand)
    return AlignmentSegment(
        chrom=str(read.reference_name),
        ref_start1=int(read.reference_start) + 1,
        ref_end1=int(read.reference_end),
        strand=strand,
        qstart=qstart,
        qend=qend,
        mapq=int(read.mapping_quality),
        source="PRIMARY",
    )


def parse_sa_segments(read) -> list[AlignmentSegment]:
    if not read.has_tag("SA"):
        return []
    segments = []
    for entry in read.get_tag("SA").split(";"):
        if not entry:
            continue
        fields = entry.split(",")
        if len(fields) < 6:
            continue
        chrom, pos1, strand, cigar, mapq, _nm = fields[:6]
        ops = parse_cigar_string(cigar)
        qstart, qend, _ = query_bounds_from_ops(ops, strand)
        ref_len = ref_length_from_ops(ops)
        ref_start1 = int(pos1)
        ref_end1 = ref_start1 + ref_len - 1
        segments.append(
            AlignmentSegment(
                chrom=str(chrom),
                ref_start1=ref_start1,
                ref_end1=ref_end1,
                strand=str(strand),
                qstart=qstart,
                qend=qend,
                mapq=int(mapq),
                source="SA",
            )
        )
    return segments


def finalize_position_cluster(items: list[dict]):
    qnames = {item["qname"] for item in items if item.get("qname")}
    positions = sorted(int(item["pos"]) for item in items)
    return {
        "pos": int(statistics.median(positions)),
        "count": len(qnames) if qnames else len(items),
        "n_items": len(items),
        "weight": float(sum(item.get("weight", 1.0) for item in items)),
        "sources": dict(Counter(item.get("source", "NA") for item in items)),
        "positions": positions,
    }


def cluster_positions(items: list[dict], tol: int = 10):
    if not items:
        return []
    items = sorted(items, key=lambda item: int(item["pos"]))
    clusters = []
    current = [items[0]]
    for item in items[1:]:
        center = statistics.median(int(x["pos"]) for x in current)
        if abs(int(item["pos"]) - int(center)) <= tol:
            current.append(item)
        else:
            clusters.append(finalize_position_cluster(current))
            current = [item]
    clusters.append(finalize_position_cluster(current))
    clusters.sort(key=lambda row: (-row["count"], -row["weight"], row["pos"]))
    return clusters


def finalize_pair_cluster(items: list[dict]):
    qnames = {item["qname"] for item in items if item.get("qname")}
    lefts = sorted(int(item["left"]) for item in items)
    rights = sorted(int(item["right"]) for item in items)
    return {
        "left": int(statistics.median(lefts)),
        "right": int(statistics.median(rights)),
        "count": len(qnames) if qnames else len(items),
        "n_items": len(items),
        "weight": float(sum(item.get("weight", 1.0) for item in items)),
        "sources": dict(Counter(item.get("source", "NA") for item in items)),
        "lengths": [max(0, int(item["right"]) - int(item["left"])) for item in items],
    }


def cluster_pairs(items: list[dict], tol: int = 20):
    if not items:
        return []
    items = sorted(items, key=lambda item: (int(item["left"]), int(item["right"])))
    clusters = []
    current = [items[0]]
    for item in items[1:]:
        left_center = statistics.median(int(x["left"]) for x in current)
        right_center = statistics.median(int(x["right"]) for x in current)
        if abs(int(item["left"]) - int(left_center)) <= tol and abs(int(item["right"]) - int(right_center)) <= tol:
            current.append(item)
        else:
            clusters.append(finalize_pair_cluster(current))
            current = [item]
    clusters.append(finalize_pair_cluster(current))
    clusters.sort(key=lambda row: (-row["count"], -row["weight"], row["left"], row["right"]))
    return clusters


def nearest_cluster_pos(clusters: list[dict], target: int, max_distance: int = 80):
    if not clusters:
        return None
    best = min(clusters, key=lambda row: abs(int(row["pos"]) - int(target)))
    if abs(int(best["pos"]) - int(target)) > max_distance:
        return None
    return int(best["pos"])


def choose_two_clip_peaks(left_clip_clusters: list[dict], right_clip_clusters: list[dict], midpoint: int):
    left_bp = None
    right_bp = None

    right_candidates = [row for row in right_clip_clusters if int(row["pos"]) <= midpoint + 200]
    left_candidates = [row for row in left_clip_clusters if int(row["pos"]) >= midpoint - 200]

    if right_candidates:
        right_candidates.sort(key=lambda row: (-row["count"], abs(int(row["pos"]) - midpoint)))
        left_bp = int(right_candidates[0]["pos"])
    elif right_clip_clusters:
        left_bp = int(right_clip_clusters[0]["pos"])

    if left_candidates:
        left_candidates.sort(key=lambda row: (-row["count"], abs(int(row["pos"]) - midpoint)))
        right_bp = int(left_candidates[0]["pos"])
    elif left_clip_clusters:
        right_bp = int(left_clip_clusters[0]["pos"])

    if left_bp is not None and right_bp is not None and left_bp > right_bp:
        left_bp, right_bp = min(left_bp, right_bp), max(left_bp, right_bp)
    return left_bp, right_bp


def build_event_evidence(event: EventCluster, min_mapq: int, min_softclip: int, flank: int):
    search_start = max(0, int(event.cluster_start) - flank)
    search_end = int(event.cluster_end) + flank
    midpoint = event.midpoint

    evidence = {
        "left_clip": [],
        "right_clip": [],
        "any_clip": [],
        "del_pairs": [],
        "ins_points": [],
        "inv_pairs": [],
        "dup_pairs": [],
        "tra_points": [],
        "pair_del": [],
        "pair_ins": [],
        "pair_inv": [],
        "pair_dup": [],
        "cross_pairs": [],
        "read_count": 0,
    }

    for read in _WORKER_BAM.fetch(event.chrom, search_start, search_end):
        if read.is_unmapped or read.is_duplicate or read.is_secondary or read.mapping_quality < min_mapq:
            continue
        evidence["read_count"] += 1
        qname = read.query_name

        if read.cigartuples:
            if read.cigartuples[0][0] == 4 and read.cigartuples[0][1] >= min_softclip:
                pos = int(read.reference_start) + 1
                evidence["left_clip"].append(
                    {"pos": pos, "qname": qname, "weight": float(read.cigartuples[0][1]), "source": "SOFTCLIP_L"}
                )
                evidence["any_clip"].append(
                    {"pos": pos, "qname": qname, "weight": float(read.cigartuples[0][1]), "source": "SOFTCLIP_L"}
                )
            if read.cigartuples[-1][0] == 4 and read.cigartuples[-1][1] >= min_softclip:
                pos = int(read.reference_end)
                evidence["right_clip"].append(
                    {"pos": pos, "qname": qname, "weight": float(read.cigartuples[-1][1]), "source": "SOFTCLIP_R"}
                )
                evidence["any_clip"].append(
                    {"pos": pos, "qname": qname, "weight": float(read.cigartuples[-1][1]), "source": "SOFTCLIP_R"}
                )

            curr_ref1 = int(read.reference_start) + 1
            for op, length in read.cigartuples:
                if op == 2 and length >= MIN_SPLIT_GAP:
                    evidence["del_pairs"].append(
                        {
                            "left": curr_ref1 - 1,
                            "right": curr_ref1 + int(length) - 1,
                            "qname": qname,
                            "weight": float(length),
                            "source": "CIGAR_D",
                        }
                    )
                elif op == 1 and length >= MIN_SPLIT_GAP:
                    evidence["ins_points"].append(
                        {
                            "pos": curr_ref1 - 1,
                            "qname": qname,
                            "weight": float(length),
                            "length": int(length),
                            "source": "CIGAR_I",
                        }
                    )
                if op in {0, 2, 3, 7, 8}:
                    curr_ref1 += int(length)

        primary = read_to_primary_segment(read)
        if primary is not None:
            for sa_seg in parse_sa_segments(read):
                ordered = sorted([primary, sa_seg], key=lambda seg: (seg.qstart, seg.qend, seg.chrom))
                first, second = ordered
                query_gap = int(second.qstart) - int(first.qend)
                boundary1 = int(first.exit_pos)
                boundary2 = int(second.entry_pos)
                if first.chrom == second.chrom:
                    if first.strand == second.strand:
                        ref_gap = int(boundary2) - int(boundary1)
                        if ref_gap >= MIN_SPLIT_GAP and ref_gap - max(query_gap, 0) >= MIN_SPLIT_GAP:
                            evidence["del_pairs"].append(
                                {
                                    "left": boundary1,
                                    "right": boundary2,
                                    "qname": qname,
                                    "weight": float(ref_gap),
                                    "source": "SA_DEL",
                                }
                            )
                        elif query_gap >= MIN_SPLIT_GAP and query_gap - max(ref_gap, 0) >= MIN_SPLIT_GAP:
                            pos = int(round((boundary1 + boundary2) / 2.0))
                            evidence["ins_points"].append(
                                {
                                    "pos": pos,
                                    "qname": qname,
                                    "weight": float(query_gap),
                                    "length": int(query_gap - max(ref_gap, 0)),
                                    "source": "SA_INS",
                                }
                            )
                        elif ref_gap <= -MIN_SPLIT_GAP:
                            evidence["dup_pairs"].append(
                                {
                                    "left": min(boundary1, boundary2),
                                    "right": max(boundary1, boundary2),
                                    "qname": qname,
                                    "weight": float(abs(ref_gap)),
                                    "source": "SA_DUP",
                                }
                            )
                    else:
                        if abs(boundary2 - boundary1) >= MIN_SPLIT_GAP:
                            evidence["inv_pairs"].append(
                                {
                                    "left": min(boundary1, boundary2),
                                    "right": max(boundary1, boundary2),
                                    "qname": qname,
                                    "weight": float(abs(boundary2 - boundary1)),
                                    "source": "SA_INV",
                                }
                            )
                else:
                    if primary.chrom == first.chrom:
                        local_break = boundary1
                        remote_break = boundary2
                        remote_chrom = second.chrom
                    else:
                        local_break = boundary2
                        remote_break = boundary1
                        remote_chrom = first.chrom
                    evidence["tra_points"].append(
                        {
                            "local_pos": int(local_break),
                            "remote_chrom": str(remote_chrom),
                            "remote_pos": int(remote_break),
                            "qname": qname,
                            "weight": 1.0,
                            "source": "SA_TRA",
                        }
                    )

        if not read.is_paired or read.mate_is_unmapped:
            continue

        same_chrom = read.reference_id == read.next_reference_id
        mate_start1 = int(read.next_reference_start) + 1 if read.next_reference_start >= 0 else None
        if same_chrom and mate_start1 is not None and read.reference_start < read.next_reference_start:
            tlen = abs(int(read.template_length))
            normal_fr = (not read.is_reverse) and read.mate_is_reverse
            rf = read.is_reverse and (not read.mate_is_reverse)
            same_orientation = read.is_reverse == read.mate_is_reverse

            if normal_fr and tlen >= _WORKER_INSERT_MEAN + (3.0 * _WORKER_INSERT_STD):
                evidence["pair_del"].append(
                    {
                        "left": int(read.reference_end),
                        "right": int(mate_start1),
                        "qname": qname,
                        "weight": float(tlen - _WORKER_INSERT_MEAN),
                        "source": "PAIR_DEL",
                    }
                )
            if normal_fr and tlen <= max(50.0, _WORKER_INSERT_MEAN - (3.0 * _WORKER_INSERT_STD)):
                pos = int(read.reference_end)
                evidence["pair_ins"].append(
                    {
                        "pos": pos,
                        "qname": qname,
                        "weight": float(_WORKER_INSERT_MEAN - tlen),
                        "length": int(round(_WORKER_INSERT_MEAN - tlen)),
                        "source": "PAIR_INS",
                    }
                )
            if rf:
                evidence["pair_dup"].append(
                    {
                        "left": min(int(read.reference_start) + 1, int(mate_start1)),
                        "right": max(int(read.reference_end), int(mate_start1)),
                        "qname": qname,
                        "weight": 1.0,
                        "source": "PAIR_DUP",
                    }
                )
            if same_orientation:
                evidence["pair_inv"].append(
                    {
                        "left": min(int(read.reference_start) + 1, int(mate_start1)),
                        "right": max(int(read.reference_end), int(mate_start1)),
                        "qname": qname,
                        "weight": 1.0,
                        "source": "PAIR_INV",
                    }
                )
        elif not same_chrom and mate_start1 is not None:
            local_pos = int(read.reference_end) if not read.is_reverse else (int(read.reference_start) + 1)
            evidence["cross_pairs"].append(
                {
                    "local_pos": local_pos,
                    "remote_chrom": str(read.next_reference_name),
                    "remote_pos": int(mate_start1),
                    "qname": qname,
                    "weight": 1.0,
                    "source": "PAIR_TRA",
                }
            )
            evidence["tra_points"].append(
                {
                    "local_pos": local_pos,
                    "remote_chrom": str(read.next_reference_name),
                    "remote_pos": int(mate_start1),
                    "qname": qname,
                    "weight": 1.0,
                    "source": "PAIR_TRA",
                }
            )
    return evidence


def refine_del(event: EventCluster, evidence: dict):
    direct = cluster_pairs(evidence["del_pairs"], tol=20)
    left_clips = cluster_positions(evidence["left_clip"], tol=10)
    right_clips = cluster_positions(evidence["right_clip"], tol=10)
    pair_del = cluster_pairs(evidence["pair_del"], tol=40)

    left_bp = None
    right_bp = None
    method = []
    split_support = 0
    pair_support = 0

    if direct:
        best = direct[0]
        left_bp = int(best["left"])
        right_bp = int(best["right"])
        split_support = int(best["count"])
        left_refined = nearest_cluster_pos(right_clips, left_bp, max_distance=60)
        right_refined = nearest_cluster_pos(left_clips, right_bp, max_distance=60)
        if left_refined is not None:
            left_bp = left_refined
        if right_refined is not None:
            right_bp = right_refined
        method.append("SPLIT")
    else:
        clip_left, clip_right = choose_two_clip_peaks(left_clips, right_clips, event.midpoint)
        if clip_left is not None and clip_right is not None and clip_left < clip_right:
            left_bp, right_bp = clip_left, clip_right
            method.append("SOFTCLIP")

    if (left_bp is None or right_bp is None) and pair_del:
        best_pair = pair_del[0]
        left_bp = int(best_pair["left"]) if left_bp is None else left_bp
        right_bp = int(best_pair["right"]) if right_bp is None else right_bp
        pair_support = int(best_pair["count"])
        method.append("PAIR")

    if left_bp is None or right_bp is None or left_bp >= right_bp:
        left_bp = max(1, int(event.cluster_start))
        right_bp = max(left_bp + 1, int(event.cluster_end))
        method.append("FALLBACK")

    left_clip_support = right_clips[0]["count"] if right_clips else 0
    right_clip_support = left_clips[0]["count"] if left_clips else 0
    filter_value = "PASS" if (split_support >= 2 or (left_clip_support >= 2 and right_clip_support >= 2)) else "LOWSUPP"
    if "FALLBACK" in method:
        filter_value = "UNRESOLVED"

    return {
        "svtype": "DEL",
        "chrom": event.chrom,
        "pos": int(left_bp),
        "end": int(right_bp),
        "svlen": int(left_bp - right_bp),
        "method": "+".join(dict.fromkeys(method)),
        "filter": filter_value,
        "split_support": split_support,
        "clip_left_support": int(left_clip_support),
        "clip_right_support": int(right_clip_support),
        "pair_support": int(pair_support),
        "partner_chrom": "",
        "partner_pos": "",
    }


def refine_ins(event: EventCluster, evidence: dict):
    direct = cluster_positions(evidence["ins_points"], tol=10)
    clips = cluster_positions(evidence["any_clip"], tol=10)
    pair_ins = cluster_positions(evidence["pair_ins"], tol=20)

    pos = None
    svlen = ""
    method = []
    split_support = 0
    pair_support = 0

    if direct:
        best = direct[0]
        pos = int(best["pos"])
        split_support = int(best["count"])
        lengths = [
            int(item["length"])
            for item in evidence["ins_points"]
            if abs(int(item["pos"]) - pos) <= 20 and int(item.get("length", 0)) > 0
        ]
        if lengths:
            svlen = int(round(statistics.median(lengths)))
        method.append("SPLIT")

    if pos is None and clips:
        clips.sort(key=lambda row: (-row["count"], abs(int(row["pos"]) - event.midpoint)))
        pos = int(clips[0]["pos"])
        method.append("SOFTCLIP")

    if pair_ins:
        near_pair_clusters = pair_ins
        if pos is not None:
            near_pair_clusters = sorted(pair_ins, key=lambda row: abs(int(row["pos"]) - pos))
        best_pair = near_pair_clusters[0]
        pair_support = int(best_pair["count"])
        if pos is None:
            pos = int(best_pair["pos"])
            method.append("PAIR")

    if pos is None:
        pos = max(1, int(event.midpoint))
        method.append("FALLBACK")

    clip_support = clips[0]["count"] if clips else 0
    filter_value = "PASS" if (split_support >= 1 or clip_support >= 3 or pair_support >= 3) else "LOWSUPP"
    if "FALLBACK" in method:
        filter_value = "UNRESOLVED"

    return {
        "svtype": "INS",
        "chrom": event.chrom,
        "pos": int(pos),
        "end": int(pos),
        "svlen": svlen,
        "method": "+".join(dict.fromkeys(method)),
        "filter": filter_value,
        "split_support": split_support,
        "clip_left_support": int(clip_support),
        "clip_right_support": int(clip_support),
        "pair_support": int(pair_support),
        "partner_chrom": "",
        "partner_pos": "",
    }


def refine_inv(event: EventCluster, evidence: dict):
    direct = cluster_pairs(evidence["inv_pairs"], tol=25)
    left_clips = cluster_positions(evidence["left_clip"], tol=10)
    right_clips = cluster_positions(evidence["right_clip"], tol=10)
    pair_inv = cluster_pairs(evidence["pair_inv"], tol=60)

    left_bp = None
    right_bp = None
    method = []
    split_support = 0
    pair_support = 0

    if direct:
        best = direct[0]
        left_bp = int(best["left"])
        right_bp = int(best["right"])
        split_support = int(best["count"])
        left_refined = nearest_cluster_pos(right_clips, left_bp, max_distance=80)
        right_refined = nearest_cluster_pos(left_clips, right_bp, max_distance=80)
        if left_refined is not None:
            left_bp = left_refined
        if right_refined is not None:
            right_bp = right_refined
        method.append("SPLIT")
    else:
        clip_left, clip_right = choose_two_clip_peaks(left_clips, right_clips, event.midpoint)
        if clip_left is not None and clip_right is not None and clip_left < clip_right:
            left_bp, right_bp = clip_left, clip_right
            method.append("SOFTCLIP")

    if (left_bp is None or right_bp is None) and pair_inv:
        best_pair = pair_inv[0]
        left_bp = int(best_pair["left"]) if left_bp is None else left_bp
        right_bp = int(best_pair["right"]) if right_bp is None else right_bp
        pair_support = int(best_pair["count"])
        method.append("PAIR")

    if left_bp is None or right_bp is None or left_bp >= right_bp:
        left_bp = max(1, int(event.cluster_start))
        right_bp = max(left_bp + 1, int(event.cluster_end))
        method.append("FALLBACK")

    filter_value = "PASS" if (split_support >= 2 or pair_support >= 3) else "LOWSUPP"
    if "FALLBACK" in method:
        filter_value = "UNRESOLVED"

    return {
        "svtype": "INV",
        "chrom": event.chrom,
        "pos": int(left_bp),
        "end": int(right_bp),
        "svlen": int(right_bp - left_bp),
        "method": "+".join(dict.fromkeys(method)),
        "filter": filter_value,
        "split_support": split_support,
        "clip_left_support": int(right_clips[0]["count"] if right_clips else 0),
        "clip_right_support": int(left_clips[0]["count"] if left_clips else 0),
        "pair_support": int(pair_support),
        "partner_chrom": "",
        "partner_pos": "",
    }


def refine_dup(event: EventCluster, evidence: dict):
    direct = cluster_pairs(evidence["dup_pairs"], tol=25)
    left_clips = cluster_positions(evidence["left_clip"], tol=10)
    right_clips = cluster_positions(evidence["right_clip"], tol=10)
    pair_dup = cluster_pairs(evidence["pair_dup"], tol=60)

    left_bp = None
    right_bp = None
    method = []
    split_support = 0
    pair_support = 0

    if direct:
        best = direct[0]
        left_bp = int(best["left"])
        right_bp = int(best["right"])
        split_support = int(best["count"])
        left_refined = nearest_cluster_pos(left_clips + right_clips, left_bp, max_distance=80)
        right_refined = nearest_cluster_pos(left_clips + right_clips, right_bp, max_distance=80)
        if left_refined is not None:
            left_bp = left_refined
        if right_refined is not None:
            right_bp = right_refined
        method.append("SPLIT")
    else:
        clip_left, clip_right = choose_two_clip_peaks(left_clips, right_clips, event.midpoint)
        if clip_left is not None and clip_right is not None and clip_left < clip_right:
            left_bp, right_bp = clip_left, clip_right
            method.append("SOFTCLIP")

    if (left_bp is None or right_bp is None) and pair_dup:
        best_pair = pair_dup[0]
        left_bp = int(best_pair["left"]) if left_bp is None else left_bp
        right_bp = int(best_pair["right"]) if right_bp is None else right_bp
        pair_support = int(best_pair["count"])
        method.append("PAIR")

    if left_bp is None or right_bp is None or left_bp >= right_bp:
        left_bp = max(1, int(event.cluster_start))
        right_bp = max(left_bp + 1, int(event.cluster_end))
        method.append("FALLBACK")

    filter_value = "PASS" if (split_support >= 2 or pair_support >= 3) else "LOWSUPP"
    if "FALLBACK" in method:
        filter_value = "UNRESOLVED"

    return {
        "svtype": "DUP",
        "chrom": event.chrom,
        "pos": int(left_bp),
        "end": int(right_bp),
        "svlen": int(right_bp - left_bp),
        "method": "+".join(dict.fromkeys(method)),
        "filter": filter_value,
        "split_support": split_support,
        "clip_left_support": int((left_clips + right_clips)[0]["count"] if (left_clips or right_clips) else 0),
        "clip_right_support": int((left_clips + right_clips)[0]["count"] if (left_clips or right_clips) else 0),
        "pair_support": int(pair_support),
        "partner_chrom": "",
        "partner_pos": "",
    }


def cluster_tra_by_partner(items: list[dict], pos_tol: int = 80):
    grouped = defaultdict(list)
    for item in items:
        grouped[str(item["remote_chrom"])].append(item)
    clusters = []
    for chrom, chrom_items in grouped.items():
        chrom_items = sorted(chrom_items, key=lambda row: int(row["remote_pos"]))
        current = [chrom_items[0]]
        for item in chrom_items[1:]:
            center = statistics.median(int(x["remote_pos"]) for x in current)
            if abs(int(item["remote_pos"]) - int(center)) <= pos_tol:
                current.append(item)
            else:
                qnames = {x["qname"] for x in current}
                clusters.append(
                    {
                        "remote_chrom": chrom,
                        "remote_pos": int(statistics.median(int(x["remote_pos"]) for x in current)),
                        "local_pos": int(statistics.median(int(x["local_pos"]) for x in current)),
                        "count": len(qnames) if qnames else len(current),
                        "sources": dict(Counter(x["source"] for x in current)),
                        "items": list(current),
                    }
                )
                current = [item]
        qnames = {x["qname"] for x in current}
        clusters.append(
            {
                "remote_chrom": chrom,
                "remote_pos": int(statistics.median(int(x["remote_pos"]) for x in current)),
                "local_pos": int(statistics.median(int(x["local_pos"]) for x in current)),
                "count": len(qnames) if qnames else len(current),
                "sources": dict(Counter(x["source"] for x in current)),
                "items": list(current),
            }
        )
    clusters.sort(key=lambda row: (-row["count"], parse_chrom_key(row["remote_chrom"]), row["remote_pos"]))
    return clusters


def collect_partner_breakpoints(source_chrom: str, source_pos: int, partner_chrom: str, rough_partner_pos: int, min_mapq: int, min_softclip: int):
    start = max(0, int(rough_partner_pos) - 1200)
    end = int(rough_partner_pos) + 1200
    clip_points = []
    mate_points = []
    try:
        iterator = _WORKER_BAM.fetch(partner_chrom, start, end)
    except ValueError:
        return [], []
    for read in iterator:
        if read.is_unmapped or read.is_duplicate or read.is_secondary or read.mapping_quality < min_mapq:
            continue
        qname = read.query_name
        if read.cigartuples:
            if read.cigartuples[0][0] == 4 and read.cigartuples[0][1] >= min_softclip:
                clip_points.append(
                    {"pos": int(read.reference_start) + 1, "qname": qname, "weight": float(read.cigartuples[0][1]), "source": "PARTNER_CLIP_L"}
                )
            if read.cigartuples[-1][0] == 4 and read.cigartuples[-1][1] >= min_softclip:
                clip_points.append(
                    {"pos": int(read.reference_end), "qname": qname, "weight": float(read.cigartuples[-1][1]), "source": "PARTNER_CLIP_R"}
                )
        if not read.is_paired or read.mate_is_unmapped or read.reference_id == read.next_reference_id:
            continue
        if str(read.next_reference_name) != str(source_chrom):
            continue
        mate_start1 = int(read.next_reference_start) + 1
        if abs(mate_start1 - int(source_pos)) <= 4000:
            local_pos = int(read.reference_end) if not read.is_reverse else (int(read.reference_start) + 1)
            mate_points.append({"pos": local_pos, "qname": qname, "weight": 1.0, "source": "PARTNER_PAIR"})
    return clip_points, mate_points


def refine_tra(event: EventCluster, evidence: dict, min_mapq: int, min_softclip: int):
    local_clips = cluster_positions(evidence["any_clip"], tol=10)
    partner_clusters = cluster_tra_by_partner(evidence["tra_points"], pos_tol=100)

    pos = None
    partner_chrom = ""
    partner_pos = None
    method = []
    split_support = 0

    if partner_clusters:
        best = partner_clusters[0]
        partner_chrom = str(best["remote_chrom"])
        partner_pos = int(best["remote_pos"])
        pos = int(best["local_pos"])
        split_support = int(best["count"])
        local_refined = nearest_cluster_pos(local_clips, pos, max_distance=120)
        if local_refined is not None:
            pos = local_refined
        partner_clip_points, partner_pair_points = collect_partner_breakpoints(
            event.chrom, pos, partner_chrom, partner_pos, min_mapq=min_mapq, min_softclip=min_softclip
        )
        partner_clusters_from_region = cluster_positions(partner_clip_points + partner_pair_points, tol=20)
        if partner_clusters_from_region:
            partner_pos = int(partner_clusters_from_region[0]["pos"])
            method.append("PARTNER_REFINE")
        method.append("TRA_SUPPORT")

    if pos is None and local_clips:
        local_clips.sort(key=lambda row: (-row["count"], abs(int(row["pos"]) - event.midpoint)))
        pos = int(local_clips[0]["pos"])
        method.append("SOFTCLIP")

    if pos is None:
        pos = max(1, int(event.midpoint))
        method.append("FALLBACK")

    if partner_pos is None:
        partner_pos = int(event.cluster_end)

    filter_value = "PASS" if split_support >= 2 else "LOWSUPP"
    if not partner_chrom:
        filter_value = "UNRESOLVED"
        method.append("NO_PARTNER")
    elif "FALLBACK" in method:
        filter_value = "UNRESOLVED"

    return {
        "svtype": "TRA",
        "chrom": event.chrom,
        "pos": int(pos),
        "end": int(pos),
        "svlen": 0,
        "method": "+".join(dict.fromkeys(method)),
        "filter": filter_value,
        "split_support": split_support,
        "clip_left_support": int(local_clips[0]["count"] if local_clips else 0),
        "clip_right_support": int(local_clips[0]["count"] if local_clips else 0),
        "pair_support": int(split_support),
        "partner_chrom": partner_chrom,
        "partner_pos": int(partner_pos) if partner_pos is not None else "",
    }


def _refine_event(task: dict):
    event = EventCluster(**task["event"])
    min_mapq = int(task["min_mapq"])
    min_softclip = int(task["min_softclip"])
    flank = int(task.get("flank", _WORKER_DEFAULT_FLANK))
    evidence = build_event_evidence(event, min_mapq=min_mapq, min_softclip=min_softclip, flank=flank)

    if event.svtype == "DEL":
        refined = refine_del(event, evidence)
    elif event.svtype == "INS":
        refined = refine_ins(event, evidence)
    elif event.svtype == "INV":
        refined = refine_inv(event, evidence)
    elif event.svtype == "DUP":
        refined = refine_dup(event, evidence)
    elif event.svtype == "TRA":
        refined = refine_tra(event, evidence, min_mapq=min_mapq, min_softclip=min_softclip)
    else:
        refined = {
            "svtype": event.svtype,
            "chrom": event.chrom,
            "pos": max(1, event.cluster_start),
            "end": max(event.cluster_start + 1, event.cluster_end),
            "svlen": event.cluster_end - event.cluster_start,
            "method": "UNKNOWN",
            "filter": "UNRESOLVED",
            "split_support": 0,
            "clip_left_support": 0,
            "clip_right_support": 0,
            "pair_support": 0,
            "partner_chrom": "",
            "partner_pos": "",
        }

    refined.update(
        {
            "event_id": event.event_id,
            "chrom": event.chrom,
            "pred_svtype": event.svtype,
            "cluster_start": event.cluster_start,
            "cluster_end": event.cluster_end,
            "n_windows": event.n_windows,
            "max_stage1_score": event.max_stage1_score,
            "mean_stage1_score": event.mean_stage1_score,
            "max_confidence": event.max_confidence,
            "mean_confidence": event.mean_confidence,
            "search_reads": int(evidence["read_count"]),
        }
    )
    return refined


def fetch_ref_base(chrom: str, pos1: int) -> str:
    try:
        base = _WORKER_FASTA.fetch(chrom, max(0, int(pos1) - 1), int(pos1)).upper()
    except Exception:
        return "N"
    return base if base else "N"


def write_refined_csv(records: list[dict], out_csv: Path):
    fieldnames = [
        "event_id",
        "chrom",
        "pred_svtype",
        "svtype",
        "pos",
        "end",
        "svlen",
        "partner_chrom",
        "partner_pos",
        "filter",
        "method",
        "split_support",
        "clip_left_support",
        "clip_right_support",
        "pair_support",
        "search_reads",
        "cluster_start",
        "cluster_end",
        "n_windows",
        "max_stage1_score",
        "mean_stage1_score",
        "max_confidence",
        "mean_confidence",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def build_vcf_lines(records: list[dict], sample_name: str):
    lines = [
        "##fileformat=VCFv4.2",
        "##source=SharpSV_direct_breakpoint_refiner",
        '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Structural variant type">',
        '##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">',
        '##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Structural variant length">',
        '##INFO=<ID=CHR2,Number=1,Type=String,Description="Partner chromosome for translocations">',
        '##INFO=<ID=METHOD,Number=1,Type=String,Description="Breakpoint refinement method">',
        '##INFO=<ID=SPLIT,Number=1,Type=Integer,Description="Unique split-read/SA support count">',
        '##INFO=<ID=CLIPL,Number=1,Type=Integer,Description="Left-breakpoint soft-clip support count">',
        '##INFO=<ID=CLIPR,Number=1,Type=Integer,Description="Right-breakpoint soft-clip support count">',
        '##INFO=<ID=PAIR,Number=1,Type=Integer,Description="Discordant-pair support count">',
        '##INFO=<ID=NWINDOWS,Number=1,Type=Integer,Description="Number of merged stage-2 windows in this event">',
        '##INFO=<ID=PREDTYPE,Number=1,Type=String,Description="Original stage-2 predicted class">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sample_name,
    ]

    body = []
    for row in records:
        chrom = str(row["chrom"])
        pos = int(row["pos"])
        ref = fetch_ref_base(chrom, pos)
        svtype = str(row["svtype"])
        if svtype == "DEL":
            alt = "<DEL>"
        elif svtype == "INS":
            alt = "<INS>"
        elif svtype == "INV":
            alt = "<INV>"
        elif svtype == "DUP":
            alt = "<DUP>"
        elif svtype == "TRA":
            alt = "<TRA>"
        else:
            alt = f"<{svtype}>"

        info_parts = [
            f"SVTYPE={svtype}",
            f"METHOD={row['method']}",
            f"SPLIT={int(row['split_support'])}",
            f"CLIPL={int(row['clip_left_support'])}",
            f"CLIPR={int(row['clip_right_support'])}",
            f"PAIR={int(row['pair_support'])}",
            f"NWINDOWS={int(row['n_windows'])}",
            f"PREDTYPE={row['pred_svtype']}",
        ]
        if svtype == "TRA" and row.get("partner_chrom"):
            info_parts.append(f"CHR2={row['partner_chrom']}")
            info_parts.append(f"END={int(row['partner_pos'])}")
        else:
            info_parts.append(f"END={int(row['end'])}")
        svlen_value = row.get("svlen")
        if svlen_value not in ("", None):
            info_parts.append(f"SVLEN={int(row['svlen'])}")

        body.append(
            "\t".join(
                [
                    chrom,
                    str(pos),
                    str(row["event_id"]),
                    ref,
                    alt,
                    ".",
                    str(row["filter"]),
                    ";".join(info_parts),
                    "GT",
                    "1/1",
                ]
            )
        )
    return lines + body


def write_vcf(path: Path, records: list[dict], sample_name: str):
    lines = build_vcf_lines(records, sample_name)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_truth_bed(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            rows.append(
                {
                    "chrom1": str(parts[0]),
                    "pos1": int(parts[1]),
                    "chrom2": str(parts[2]),
                    "pos2": int(parts[3]),
                    "svtype": str(parts[4]),
                }
            )
    return rows


def summarize_against_truth(records: list[dict], truth_rows: list[dict]):
    truth_by_type = defaultdict(list)
    for row in truth_rows:
        truth_by_type[str(row["svtype"])].append(row)

    match_rows = []
    for row in records:
        svtype = str(row["svtype"])
        candidates = truth_by_type.get(svtype, [])
        if not candidates:
            continue
        best = None
        best_score = None
        for truth in candidates:
            if str(truth["chrom1"]) != str(row["chrom"]):
                continue
            if svtype == "TRA" and str(truth["chrom2"]) != str(row.get("partner_chrom", "")):
                continue
            ref_end = int(row["partner_pos"]) if svtype == "TRA" and row.get("partner_pos") else int(row["end"])
            score = abs(int(row["pos"]) - int(truth["pos1"])) + abs(ref_end - int(truth["pos2"]))
            if best_score is None or score < best_score:
                best_score = score
                best = truth
        if best is not None:
            match_rows.append({"event_id": row["event_id"], "svtype": svtype, "distance_sum": int(best_score)})

    if not match_rows:
        return {}
    by_type = defaultdict(list)
    for row in match_rows:
        by_type[str(row["svtype"])].append(int(row["distance_sum"]))
    return {
        "matched_events": len(match_rows),
        "median_distance_sum": int(statistics.median(int(row["distance_sum"]) for row in match_rows)),
        "per_type_median_distance_sum": {svtype: int(statistics.median(vals)) for svtype, vals in by_type.items()},
    }


def main():
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    clusters = load_prediction_clusters(args.predictions, cluster_gap=args.cluster_gap)
    insert_mean, insert_std = estimate_insert_stats(args.bam)
    print(
        f"[init] loaded {len(clusters):,} clustered events from {args.predictions} "
        f"(insert mean={insert_mean:.2f}, std={insert_std:.2f})"
    )

    tasks = [
        {
            "event": event.__dict__,
            "min_mapq": args.min_mapq,
            "min_softclip": args.min_softclip,
            "flank": args.flank,
        }
        for event in clusters
    ]

    refined_records = []
    with mp.Pool(
        processes=max(int(args.workers), 1),
        initializer=_init_worker,
        initargs=(args.bam, args.fasta, insert_mean, insert_std, args.flank),
    ) as pool:
        for idx, row in enumerate(pool.imap_unordered(_refine_event, tasks, chunksize=20), start=1):
            refined_records.append(row)
            if idx == 1 or idx % 250 == 0 or idx == len(tasks):
                elapsed = time.time() - started_at
                rate = idx / elapsed if elapsed > 0 else 0.0
                print(f"[refine] {idx:,}/{len(tasks):,} events ({rate:.1f} events/s)")

    refined_records = sort_chrom_records(refined_records)
    all_csv = outdir / "refined_events.csv"
    write_refined_csv(refined_records, all_csv)

    pass_records = [row for row in refined_records if str(row["filter"]) == "PASS"]
    _init_worker(args.bam, args.fasta, insert_mean, insert_std, args.flank)
    write_vcf(outdir / "refined_events_all.vcf", refined_records, sample_name=args.sample_name)
    write_vcf(outdir / "refined_events_pass.vcf", pass_records, sample_name=args.sample_name)

    summary = {
        "input_predictions": str(Path(args.predictions).expanduser().resolve()),
        "clustered_events": len(clusters),
        "refined_events": len(refined_records),
        "pass_events": len(pass_records),
        "insert_mean": insert_mean,
        "insert_std": insert_std,
        "per_type_counts": dict(Counter(str(row["svtype"]) for row in refined_records)),
        "per_filter_counts": dict(Counter(str(row["filter"]) for row in refined_records)),
        "runtime_seconds": round(time.time() - started_at, 2),
    }
    if args.truth_bed:
        summary["truth_summary"] = summarize_against_truth(refined_records, load_truth_bed(args.truth_bed))

    (outdir / "refinement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {all_csv}")
    print(f"[done] wrote {outdir / 'refined_events_all.vcf'}")
    print(f"[done] wrote {outdir / 'refined_events_pass.vcf'}")
    print(f"[done] wrote {outdir / 'refinement_summary.json'}")


if __name__ == "__main__":
    main()

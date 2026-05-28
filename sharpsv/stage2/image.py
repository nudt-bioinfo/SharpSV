import numpy as np
import pysam
from enum import Enum


def _adapt_contig_id(ref_file, chr_id, prefer_numeric=False):
    if not hasattr(ref_file, "references"):
        return chr_id

    references = set(ref_file.references)
    if chr_id in references:
        return chr_id

    if prefer_numeric:
        if chr_id.startswith("chr"):
            no_chr = chr_id[3:]
            if no_chr in references:
                return no_chr
    else:
        if not chr_id.startswith("chr"):
            chr_prefixed = f"chr{chr_id}"
            if chr_prefixed in references:
                return chr_prefixed

    if chr_id.startswith("chr"):
        no_chr = chr_id[3:]
        if no_chr in references:
            return no_chr
    else:
        chr_prefixed = f"chr{chr_id}"
        if chr_prefixed in references:
            return chr_prefixed

    return None


def estimate_insert_sizes(sam_file_path, alignments=1000000):
    inserts = []
    mapping_qualities = []
    count = 0
    sam_file = pysam.AlignmentFile(sam_file_path, "rb")
    for read in sam_file:
        if (
            read.is_proper_pair
            and read.is_paired
            and read.is_read1
            and (not read.is_unmapped)
            and (not read.mate_is_unmapped)
            and (not read.is_duplicate)
            and (not read.is_secondary)
            and (not read.is_supplementary)
        ):
            if (
                (read.reference_start < read.next_reference_start and (not read.is_reverse) and read.mate_is_reverse)
                or (read.reference_start > read.next_reference_start and read.is_reverse and (not read.mate_is_reverse))
            ):
                count += 1
                if count <= alignments:
                    inserts.append(abs(read.tlen))
                    mapping_qualities.append(abs(read.mapping_quality))
                else:
                    break
    sam_file.close()

    inserts = sorted(inserts)
    mapping_qualities = sorted(mapping_qualities)
    total_num = len(inserts)
    if total_num == 0:
        raise ValueError("No proper paired reads were found for insert-size estimation.")

    l = int(0.05 * total_num)
    r = int(0.95 * total_num)
    inserts = inserts[l:r] or inserts
    mapping_qualities = mapping_qualities[l:r] or mapping_qualities

    insert_mean = int(np.mean(inserts))
    insert_std = int(np.std(inserts))
    mapping_qualities_mean = int(np.mean(mapping_qualities))
    return insert_mean, insert_std, mapping_qualities_mean


def calculate_gc_content_from_pileupread(pileupread):
    if not hasattr(pileupread, "query_sequence"):
        return 0

    read_sequence = pileupread.query_sequence
    if not read_sequence:
        return 0

    gc_count = 0
    for base in read_sequence:
        if base.upper() in {"G", "C"}:
            gc_count += 1

    return int(gc_count / len(read_sequence) * 255)


SV_SIGNAL_RP_TYPE = Enum("SV_SIGNAL_RP_TYPE", "LR LLRR RL")


def get_read_pair_type(read, rl_dist_thr=5):
    if read.is_reverse == read.mate_is_reverse:
        return SV_SIGNAL_RP_TYPE.LLRR
    if (
        ((read.reference_start + rl_dist_thr) < read.next_reference_start and read.is_read2 and read.is_reverse)
        or (read.reference_start > (read.next_reference_start + rl_dist_thr) and read.is_read1 and (not read.is_reverse))
    ):
        return SV_SIGNAL_RP_TYPE.RL
    return SV_SIGNAL_RP_TYPE.LR


def pipeup_column(
    sam_file,
    chr_id,
    pos_l,
    pos_r,
    mean_insert_size,
    sd_insert_size,
    fasta_file,
    clip_dict_record,
):
    pile_record = []
    bam_chr = _adapt_contig_id(sam_file, chr_id)
    fasta_chr = _adapt_contig_id(fasta_file, chr_id, prefer_numeric=True)

    if bam_chr is None or fasta_chr is None:
        return []

    for col in sam_file.pileup(bam_chr, pos_l - 1, pos_r - 1, truncate=True, stepper="all"):
        ref_pos0 = col.pos
        ref_pos1 = ref_pos0 + 1
        if not (pos_l <= ref_pos1 < pos_r):
            continue

        try:
            ref_base = fasta_file.fetch(fasta_chr, ref_pos0, ref_pos0 + 1)
        except Exception:
            ref_base = "N"

        for pr in col.pileups:
            read = pr.alignment
            if read is None or read.is_unmapped:
                continue

            if pr.query_position is not None:
                base = read.query_sequence[pr.query_position]
                base_qual = read.query_qualities[pr.query_position]
                base_equals_ref = base == ref_base
            else:
                base = "N"
                base_qual = 0
                base_equals_ref = False

            gc_content = calculate_gc_content_from_pileupread(read)
            read_pair_type = get_read_pair_type(read) if read.is_paired and not read.mate_is_unmapped else SV_SIGNAL_RP_TYPE.LR

            insert_signal = 0
            if read.is_paired and not read.mate_is_unmapped:
                tlen = abs(read.template_length)
                if tlen > mean_insert_size + 2 * sd_insert_size:
                    insert_signal = 1
                elif tlen < mean_insert_size - 2 * sd_insert_size:
                    insert_signal = 2

            map_type = 0
            ref_cursor = read.reference_start
            for op, length in read.cigartuples:
                if op in {0, 2, 3}:
                    if ref_cursor <= ref_pos0 < ref_cursor + length:
                        map_type = op
                        break
                    ref_cursor += length

            clip_value = clip_dict_record.get(ref_pos0, 0)
            if col.n > 0:
                clip_value = int((clip_value / col.n) * 255)

            ref_len = read.reference_length or 1
            qry_len = read.query_alignment_length or 1
            variants = abs(qry_len - ref_len) / ref_len >= 0.1

            pile_record.append(
                (
                    ref_pos1,
                    gc_content,
                    read.is_proper_pair,
                    read.mapping_quality,
                    map_type,
                    base,
                    base_qual,
                    read_pair_type,
                    read.is_reverse,
                    insert_signal,
                    base_equals_ref,
                    clip_value,
                    variants,
                )
            )

    return pile_record


def get_rgb(every_pile_record, mapping_qualities_mean):
    if every_pile_record[5] == "A":
        base_color = 60.0
    elif every_pile_record[5] == "T":
        base_color = 120.0
    elif every_pile_record[5] == "C":
        base_color = 180.0
    elif every_pile_record[5] == "G":
        base_color = 240.0
    else:
        base_color = 0.0

    gc_value_color = np.clip(every_pile_record[1], 0, 255)
    is_proper_color = 120.0 if every_pile_record[2] else 240.0
    normalized_value = int((every_pile_record[3] / max(mapping_qualities_mean, 1)) * 255)
    mapping_quality_color = np.clip(normalized_value, 0, 255)

    map_type = {0: 60.0, 1: 120.0, 2: 180.0, 4: 240.0}
    map_type_color = map_type.get(every_pile_record[4], 60.0)

    normalized_value = int((every_pile_record[6] / 40) * 255)
    base_quality_color = np.clip(normalized_value, 0, 255)

    read_pair_type = {
        SV_SIGNAL_RP_TYPE.LR: 80.0,
        SV_SIGNAL_RP_TYPE.LLRR: 240.0,
        SV_SIGNAL_RP_TYPE.RL: 160.0,
    }
    read_pair_type_color = read_pair_type.get(every_pile_record[7], 80.0)

    is_reverse_color = 120.0 if every_pile_record[8] else 240.0

    insert_size = {1: 240.0, 2: 160.0, 0: 80.0}
    insert_size_color = insert_size.get(every_pile_record[9], 80.0)

    equal_to_ref_color = 120.0 if every_pile_record[10] else 240.0
    clip_color = np.clip(int(every_pile_record[11]), 0, 255)
    variants_color = 120.0 if every_pile_record[12] is False else 240.0

    return (
        base_color,
        gc_value_color,
        is_proper_color,
        mapping_quality_color,
        map_type_color,
        base_quality_color,
        read_pair_type_color,
        is_reverse_color,
        insert_size_color,
        equal_to_ref_color,
        clip_color,
        variants_color,
    )


estimateInsertSizes = estimate_insert_sizes

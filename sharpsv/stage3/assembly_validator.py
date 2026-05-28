import pandas as pd
import pysam
import numpy as np
import argparse
import sys
import time
from numba.typed import List

try:
    import mamnet
except ImportError:
    try:
        from .. import native as mamnet
    except ImportError:
        print("[Error] Cannot import 'mamnet' or fallback 'sharpsv_native'.")
        sys.exit(1)


def _log(message, logger):
    if logger:
        logger(message)


def _render_bar(current, total, width=18):
    width = max(int(width), 8)
    try:
        current = float(current)
        total = float(total)
    except (TypeError, ValueError):
        return "░" * width
    if total <= 0:
        return "░" * width
    ratio = max(0.0, min(current / total, 1.0))
    filled = min(width, int(round(ratio * width)))
    return ("█" * filled) + ("░" * (width - filled))


def _emit_progress(progress_callback, logger, current, total, speed=None, eta_seconds=None, extra=None):
    if progress_callback is not None:
        progress_callback(current, total, speed=speed, eta_seconds=eta_seconds, extra=extra)
        return

    try:
        current_value = float(current)
        total_value = float(total)
    except (TypeError, ValueError):
        current_value = 0.0
        total_value = 0.0

    ratio = 0.0 if total_value <= 0 else max(0.0, min(current_value / total_value, 1.0))
    bar = _render_bar(current_value, total_value, width=18)
    message = f"[INFO] progress {ratio * 100:5.1f}% |{bar}| {int(current_value):,}/{int(total_value):,}"
    if speed is not None:
        message += f" | {float(speed):.1f}/s"
    if eta_seconds is not None and np.isfinite(float(eta_seconds)):
        message += f" | eta {max(int(round(float(eta_seconds))), 0)}s"
    if extra:
        message += f" | {extra}"
    _log(message, logger)


def estimate_bam_stats(bam, sample_reads=200000, logger=print):
    insert_sizes = []
    read_lengths = []
    sampled_count = 0

    for read in bam.fetch(until_eof=True):
        if sampled_count >= sample_reads: break
        if read.is_unmapped or read.is_duplicate: continue
        read_lengths.append(read.query_length)
        sampled_count += 1
        if read.is_paired and not read.mate_is_unmapped and 50 < abs(read.template_length) < 6000:
            insert_sizes.append(abs(read.template_length))

    if len(insert_sizes) < 100:
        return 557.0, 154.6, 148.0, 59.2

    isize_mean = np.mean(insert_sizes)
    isize_std = np.std(insert_sizes)
    avg_read_len = np.mean(read_lengths)

    try:
        total_mapped = sum(c.mapped for c in bam.get_index_statistics())
        genome_size = sum(bam.lengths)
        global_depth = (total_mapped * avg_read_len) / genome_size if genome_size > 0 else 59.2
    except:
        global_depth = 59.2

    _log(f"[Init] Insert Size={isize_mean:.1f}±{isize_std:.1f}, Read Len={avg_read_len:.1f}, Depth={global_depth:.1f}", logger)
    return isize_mean, isize_std, avg_read_len, global_depth


# ==========================================
# 1. Raw Signal
# ==========================================
def extract_raw_signal_adaptive(bam, chrom, start, end, sv_type, pad=370):
    win_start = max(0, int(start - pad))
    win_end = int(end + pad)
    win_size = max(290, win_end - win_start)

    local_count = bam.count(chrom, win_start, win_end)
    dynamic_max = max(390, local_count + 140)

    cigarlist = List()
    mdtaglist = List()
    corposlist = List()
    read_count = 0

    for read in bam.fetch(chrom, win_start, win_end):
        if read_count >= dynamic_max: break
        if read.is_unmapped or read.is_duplicate or not read.cigarstring: continue
        cigarlist.append(read.cigarstring)
        mdtaglist.append(read.get_tag('MD') if read.has_tag('MD') else "")
        corposlist.append(List([read.reference_start - win_start, read.reference_end - win_start]))
        read_count += 1

    if read_count == 0:
        return 0.0, 0

    dummy_quality = np.ones(read_count * 3, dtype=np.int64)
    try:
        matrix = mamnet.g_d(mdtaglist, cigarlist, corposlist, win_size, dummy_quality, dynamic_max)
    except:
        return 0.0, read_count

    rel_s = max(0, int(start - win_start))
    rel_e = min(win_size, int(end - win_start))
    if rel_s >= rel_e:
        return 0.0, read_count

    roi = matrix[rel_s:rel_e, :]

    def get_support(col, p=33):
        sig = roi[:, col]
        if len(sig) == 0: return 0.0
        thresh = np.percentile(sig, p)
        return float(np.sum(sig > max(thresh, 1.0)))

    clip_s = get_support(7, 33)

    if sv_type == 'DEL':
        del_s = get_support(1, 33)
        support = del_s * 0.65 + clip_s * 1.15
    elif sv_type == 'INS':
        ins_s = get_support(4, 33)
        support = ins_s * 0.65 + clip_s * 1.15
    else:
        support = clip_s

    return support, read_count


# ==========================================
# 2. RP - 锁定最佳的统计学边界
# ==========================================
def extract_rp_signal_adaptive(bam, chrom, start, end, isize_mean, isize_std, sv_type):
    abnormal = 0.0
    total = 0
    seen = set()

    # [调整] 1.95 倍标准差：比 2.0 敏感，比 1.8 准确，是最优平衡点
    thresh_high = isize_mean + 1.95 * isize_std
    thresh_low = isize_mean - 1.95 * isize_std

    try:
        reads = bam.fetch(chrom, max(0, start - 670), end + 670)
    except:
        return 0.0, 0

    for read in reads:
        if (not read.is_paired or read.is_unmapped or read.mate_is_unmapped or
                read.is_duplicate or read.mapping_quality < 9 or read.template_length == 0):
            continue
        if read.reference_start >= read.next_reference_start: continue

        qname = read.query_name
        if qname in seen: continue
        seen.add(qname)

        tlen = abs(read.template_length)
        total += 1

        if tlen > thresh_high:
            abnormal += 1.0 if sv_type == 'DEL' else 0.8
        elif tlen < thresh_low:
            abnormal += 1.0 if sv_type == 'INS' else 0.8

    return abnormal, total


# ==========================================
# 3. Assembly - 排除随机组装噪音
# ==========================================
def analyze_assembly_adaptive(bam, chrom, start, end, sv_type, predicted_len, avg_read_len):
    try:
        reads = list(bam.fetch(chrom, max(0, start - 370), end + 370))
    except:
        reads = []

    if not reads:
        return 'AMBIGUOUS', None

    has_sv = False
    best_refine = None
    clean_ref = 0
    total_valid = 0

    # [收紧] 要求至少 8bp 变异才能视为 SV 证据，防止将小 Indel 误判为大型 SV
    MIN_SV_LEN = max(8, int(predicted_len * 0.025), int(avg_read_len * 0.06))
    # [收紧] 容差范围回到 300，太宽容易引入随机错误
    slop = max(300, int(predicted_len * 0.20))

    for read in reads:
        if read.mapping_quality < 9: continue
        total_valid += 1
        ops = read.cigartuples or []
        curr_pos = read.reference_start
        read_has_sv = False

        for op, length in ops:
            found = False
            r_start = r_len = None

            if sv_type == 'DEL' and op == 2 and length >= MIN_SV_LEN:
                if abs(curr_pos - start) < slop or abs(curr_pos - end) < slop:
                    found = True
                    r_start = curr_pos
                    r_len = length
            elif sv_type == 'INS' and op == 1 and length >= MIN_SV_LEN:
                if abs(curr_pos - start) < slop:
                    found = True
                    r_start = curr_pos
                    r_len = length
            elif op == 4 and length >= MIN_SV_LEN:
                clip_pos = read.reference_start if (ops and ops[0][0] == 4) else read.reference_end
                if abs(clip_pos - start) < slop or abs(clip_pos - end) < slop:
                    found = True
                    r_start = clip_pos

            if found:
                read_has_sv = True
                has_sv = True
                if r_len is not None:
                    best_refine = (r_start, r_len)

            if op in (0, 2, 3, 7, 8):
                curr_pos += length

        if not read_has_sv:
            nm = read.get_tag("NM") if read.has_tag("NM") else 0
            if nm <= read.query_alignment_length * 0.095:
                if read.reference_start < start and read.reference_end > end:
                    clean_ref += 1

    if has_sv:
        return 'VERIFIED', best_refine

    # [严格界定 REF] 只要有超过 35% 的 reads 是干净比对，就先将其判为 REF，后续用极严逻辑排查
    if total_valid > 0 and clean_ref / total_valid > 0.35:
        return 'REF', None

    return 'AMBIGUOUS', None


# ==========================================
# 4. 主流程 - 双重证据验证 (Dual-Evidence)
# ==========================================
def validate_assembly_candidates(
    csv_path,
    raw_bam_path,
    asm_bam_path,
    output_path="final_adaptive_validated.csv",
    min_vaf=0.065,
    logger=print,
    progress_callback=None,
):
    raw_bam = pysam.AlignmentFile(raw_bam_path, "rb")
    asm_bam = pysam.AlignmentFile(asm_bam_path, "rb")

    isize_mean, isize_std, avg_read_len, global_depth = estimate_bam_stats(raw_bam, logger=logger)

    df = pd.read_csv(csv_path)
    if 'chrom' in df.columns:
        df['chrom'] = df['chrom'].astype(str).str.replace(r'\.0$', '', regex=True)

    validated_rows = []
    stats = {'total': len(df), 'kept_asm': 0, 'kept_rescue': 0, 'drop_ref': 0, 'drop_noise': 0}
    total_rows = len(df)
    started_at = time.time()
    progress_interval = max(1, min(200, total_rows // 100 if total_rows > 0 else 1))

    _log(f"Starting v10 Dual-Evidence Validation (Depth ~{global_depth:.1f})", logger)

    for idx, row in df.iterrows():
        chrom = str(row['chrom'])
        s_chrom = chrom
        if not chrom.startswith('chr') and any(r.startswith('chr') for r in raw_bam.references):
            s_chrom = 'chr' + chrom
        elif chrom.startswith('chr') and '1' in raw_bam.references:
            s_chrom = chrom.replace('chr', '')

        start = int(row['start'])
        end = int(row['end'])
        pred_len = abs(end - start)
        sv_type = row.get('sv_type', 'DEL' if int(row.get('label', 1)) == 1 else 'INS')

        asm_status, refinement = analyze_assembly_adaptive(asm_bam, s_chrom, start, end, sv_type, pred_len,
                                                           avg_read_len)

        raw_support, raw_depth = extract_raw_signal_adaptive(raw_bam, s_chrom, start, end, sv_type)
        raw_vaf = raw_support / raw_depth if raw_depth > 0 else 0.0

        rp_abnormal, rp_total = extract_rp_signal_adaptive(raw_bam, s_chrom, start, end, isize_mean, isize_std, sv_type)
        rp_ratio = rp_abnormal / rp_total if rp_total > 0 else 0.0

        final_decision = "DROP"
        new_row = row.copy()

        # 计算自适应高阈值 (上限 12%)
        adaptive_vaf = max(min_vaf, min(0.12, global_depth * 0.002))

        if asm_status == 'VERIFIED':
            # 【二次质控】即使组装认为有SV，如果底层测序数据连 1.0 的痕迹都没有，按假阳性Drop
            if raw_support < 1.0 and rp_abnormal < 1.0:
                final_decision = "DROP"
                stats['drop_noise'] += 1
            else:
                final_decision = "KEEP"
                stats['kept_asm'] += 1
                if refinement:
                    r_start, r_len = refinement
                    if abs(r_start - start) < 2000 and r_len and abs(r_len - pred_len) < pred_len * 0.83:
                        new_row['start'] = int(r_start)
                        if sv_type == 'DEL' and r_len:
                            new_row['end'] = int(r_start + r_len)
                        new_row['length_bp'] = int(r_len)

        elif asm_status == 'REF':
            # 【严苛斩杀】明确为 REF 的区域，必须具备极强的断点信号才予救援
            if raw_vaf >= 0.10 and raw_support >= 5.0:
                final_decision = "KEEP"
                stats['kept_rescue'] += 1
            else:
                final_decision = "DROP"
                stats['drop_ref'] += 1

        else:  # AMBIGUOUS
            # 策略 A: 极强的单维度 CIGAR/Clip 证据
            strong_raw = (raw_vaf >= adaptive_vaf and raw_support >= 5.0)

            # 策略 B: 极强的单维度 Read Pair 证据
            strong_rp = (rp_ratio >= 0.08 and rp_abnormal >= 3.0)

            # 策略 C: 双重证据 (Dual-Evidence) - 这是提升 Recall 且不掉 Precision 的核心
            # 当两个独立指标同时给出弱信号(>4.5%)时，合并判定为真
            dual_evidence = (raw_vaf >= 0.045 and raw_support >= 3.0) and (rp_ratio >= 0.045 and rp_abnormal >= 2.0)

            if strong_raw or strong_rp or dual_evidence:
                final_decision = "KEEP"
                stats['kept_rescue'] += 1
            else:
                final_decision = "DROP"
                stats['drop_noise'] += 1

        if final_decision == "KEEP":
            validated_rows.append(new_row)

        rows_done = idx + 1
        if rows_done == 1 or rows_done == total_rows or rows_done % progress_interval == 0:
            elapsed = time.time() - started_at
            row_rate = rows_done / elapsed if elapsed > 0 else 0.0
            eta_seconds = (total_rows - rows_done) / row_rate if row_rate > 0 else float("inf")
            _emit_progress(
                progress_callback,
                logger,
                rows_done,
                total_rows,
                speed=row_rate,
                eta_seconds=eta_seconds,
                extra=(
                    f"keep {len(validated_rows):,}  ·  asm {stats['kept_asm']:,}  ·  "
                    f"rescue {stats['kept_rescue']:,}  ·  drop {stats['drop_ref'] + stats['drop_noise']:,}"
                ),
            )

    _log("=== v10 DUAL-EVIDENCE SUMMARY ===", logger)
    kept = len(validated_rows)
    _log(f"Total: {stats['total']} | Kept: {kept}", logger)
    _log(f"   Asm Verified: {stats['kept_asm']}", logger)
    _log(f"   Raw Rescued : {stats['kept_rescue']}", logger)
    _log(f"   Dropped     : {stats['drop_ref'] + stats['drop_noise']}", logger)

    out_df = pd.DataFrame(validated_rows)
    if out_df.empty:
        pd.DataFrame(columns=list(df.columns)).to_csv(output_path, index=False)
    else:
        cols = [c for c in df.columns if c in out_df.columns]
        out_df[cols].to_csv(output_path, index=False)
    _log(f"[Output] Saved -> {output_path}", logger)
    raw_bam.close()
    asm_bam.close()
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(description="SV Validator v10 - Dual-Evidence Target (P≥0.80, R≥0.45)")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--raw_bam", required=True)
    parser.add_argument("--asm_bam", required=True)
    parser.add_argument("--output", default="final_adaptive_validated.csv")
    parser.add_argument("--min_vaf", type=float, default=0.065)  # 基础门槛定在 6.5%，保证精度
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_assembly_candidates(
        csv_path=args.csv,
        raw_bam_path=args.raw_bam,
        asm_bam_path=args.asm_bam,
        output_path=args.output,
        min_vaf=args.min_vaf,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

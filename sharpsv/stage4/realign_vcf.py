import sys
import pysam
import numpy as np
from Bio.Align import PairwiseAligner
import argparse
import collections
import multiprocessing as mp


# ==========================================
# 模块 1: 统计模块 (增强版)
# ==========================================
class BamStatCalculator:
    def __init__(self, bam_path, sample_size=50000):
        self.bam_path = bam_path
        self.mean = 350.0
        self.std = 50.0
        self.sample_size = sample_size
        self.calculate_stats()

    def calculate_stats(self):
        inserts = []
        try:
            with pysam.AlignmentFile(self.bam_path, "rb") as bam:
                count = 0
                for read in bam:
                    if count >= self.sample_size: break
                    # 必须是正常配对且不在同一位置
                    if read.is_proper_pair and not read.is_duplicate and read.mapping_quality > 20:
                        if read.next_reference_id == read.reference_id and read.next_reference_start > read.reference_start:
                            tlen = abs(read.template_length)
                            # 排除极其离谱的长片段，只统计正常分布
                            if tlen < 2000:
                                inserts.append(tlen)
                                count += 1
            if inserts:
                # 使用中位数和MAD估算，比Mean/Std更抗干扰
                self.mean = np.median(inserts)
                self.std = np.std(inserts)
                print(f"[Info] Bam Stats -> Mean: {self.mean:.2f}, STD: {self.std:.2f}")
            else:
                print("[Warning] Could not calc stats, using defaults.")
        except Exception as e:
            print(f"[Error] Stat calculation failed: {e}")


# ==========================================
# 模块 2: 终极重比对引擎 (混合模式)
# ==========================================
class SpritesRefiner:
    def __init__(self, bam_path, ref_path, insert_mean, insert_std):
        self.bam = pysam.AlignmentFile(bam_path, "rb")
        self.ref = pysam.FastaFile(ref_path)
        self.insert_mean = insert_mean
        self.insert_std = insert_std
        self.aligner = self._build_aligner()

        # 搜索参数
        self.search_windows = [100, 500, 1000] #100
        self.min_clip_len = 5
        self.min_mapq = 20  # 新增：MapQ 阈值

        # 判定 Discordant Pair 的阈值 (Mean + 3*STD)
        self.discordant_threshold = self.insert_mean + (3 * self.insert_std)

    def _build_aligner(self):
        aligner = PairwiseAligner(mode="local")
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -1
        return aligner

    def get_soft_clip_seq(self, read):
        if not read.cigartuples: return None, None, None

        # Right Clip (Forward Strand logic)
        if read.cigartuples[-1][0] == 4 and read.cigartuples[-1][1] >= self.min_clip_len:
            return 'right', read.query_sequence[-read.cigartuples[-1][1]:], read.reference_end

        # Left Clip (Reverse Strand logic mostly)
        if read.cigartuples[0][0] == 4 and read.cigartuples[0][1] >= self.min_clip_len:
            return 'left', read.query_sequence[:read.cigartuples[0][1]], read.reference_start

        return None, None, None

    def get_consensus_len(self, len_list, method="cluster"):
        """
        method: 'cluster' for SR (precise), 'median' for DP (imprecise)
        """
        if not len_list: return None

        if method == "median":
            # 对于 Insert Size 推断，直接取中位数
            return int(np.median(len_list))

        # 对于 Split Reads，使用聚类
        if len(len_list) == 1: return len_list[0]
        len_list.sort()
        clusters = []
        current_cluster = [len_list[0]]
        for i in range(1, len(len_list)):
            if len_list[i] - len_list[i - 1] <= 3:  #20
                current_cluster.append(len_list[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [len_list[i]]
        clusters.append(current_cluster)
        best_cluster = max(clusters, key=len)

        # 去噪逻辑
        if len(best_cluster) < 2 and len(len_list) > 3: return None
        return int(np.median(best_cluster))

    def refine_deletion(self, vcf_chrom, vcf_pos, vcf_end):
        """
        返回: (Length, Method)
        Method: 'SR' (Split Read), 'DP' (Discordant Pair), None
        """

        # --- 阶段 1: 尝试 Split Read (SR) 精修 ---
        sr_lens = []
        for radius in self.search_windows:
            if len(sr_lens) >= 5: break

            # 安全范围
            start = max(0, vcf_pos - radius)
            end = vcf_pos + radius

            try:
                reads = self.bam.fetch(vcf_chrom, start, end)
            except:
                continue

            for read in reads:
                # 基础过滤：去重、MapQ
                if read.is_unmapped or read.is_duplicate or read.mapping_quality < self.min_mapq:
                    continue
                if not read.is_proper_pair:
                    continue

                direction, clip_seq, anchor_pos = self.get_soft_clip_seq(read)
                if not anchor_pos: continue

                mate_pos = read.next_reference_start
                if read.next_reference_id != read.reference_id: continue

                # 逻辑与之前相同，寻找 target region
                target_start, target_end = 0, 0
                if direction == 'right' and abs(anchor_pos - vcf_pos) < radius:
                    if mate_pos < anchor_pos: continue
                    target_start = anchor_pos
                    target_end = mate_pos + read.query_length + 100
                    self.align_and_calc(vcf_chrom, clip_seq, target_start, target_end, anchor_pos, 'right', sr_lens)

                elif direction == 'left':
                    if anchor_pos < vcf_pos: continue
                    target_start = max(0, mate_pos - 100)
                    target_end = anchor_pos
                    self.align_and_calc(vcf_chrom, clip_seq, target_start, target_end, anchor_pos, 'left', sr_lens)

        consensus_sr = self.get_consensus_len(sr_lens, method="cluster")

        # 如果 Split Read 找到了可靠结果，直接返回 (PRECISE)
        if consensus_sr is not None:
            return consensus_sr, 'SR'

        # --- 阶段 2: 救回模式 - Discordant Pairs (DP) ---
        # 如果 SR 失败，尝试用 Insert Size 救回
        dp_lens = self.rescue_by_discordant_pairs(vcf_chrom, vcf_pos, vcf_end)
        consensus_dp = self.get_consensus_len(dp_lens, method="median")

        if consensus_dp is not None:
            return consensus_dp, 'DP'

        return None, None

    def rescue_by_discordant_pairs(self, chrom, start, end):
        """
        利用 Insert Size 异常大的 Read Pairs 推断缺失长度
        原理: Observed_TLEN = Physical_Insert + Deletion_Size
             Deletion_Size = Observed_TLEN - Mean_Insert
        """
        estimated_lens = []

        # 搜索范围：覆盖整个预估的 Deletion 区域
        # 我们只看 Start 附近的 Reads，因为它们会跨越到 End 后面
        search_start = max(0, start - 200) #200
        search_end = start + 200 #200

        try:
            reads = self.bam.fetch(chrom, search_start, search_end)
        except:
            return []

        for read in reads:
            # 1. 基础过滤
            if read.is_unmapped or read.is_duplicate or read.mapping_quality < self.min_mapq:
                continue
            if not read.is_proper_pair:  # 有些比对软件对大 DEL 仍标记为 proper，有些则否，视情况而定
                pass

                # 2. 必须是跨越断点的 Pair
            # Read 在 Start 左边，Mate 在 End 右边 (大概)
            # 或者简单判断：Template Length 是否显著异常
            tlen = abs(read.template_length)

            if tlen > self.discordant_threshold:
                # 3. 计算预估长度
                # 粗略公式：Del_Len = TLEN - Mean_Insert
                # 过滤掉过大的噪音 (比如 > 50kb，除非预期很大)
                est_len = tlen - self.insert_mean

                # 简单的合理性检查: 估算的长度应该和 VCF 里的原始长度在同一个数量级，或者至少 > 50bp
                if 50 < est_len < 100000:
                    estimated_lens.append(est_len)

        # 只有当发现一定数量的证据时才采信 (例如至少 3 对)
        if len(estimated_lens) >= 3:
            return estimated_lens
        return []

    def align_and_calc(self, chrom, clip_seq, t_start, t_end, anchor_pos, mode, results):
        if t_end - t_start > 50000: t_end = t_start + 50000
        if t_end <= t_start: return

        try:
            target_seq = self.ref.fetch(chrom, t_start, t_end)
        except:
            return
        if not target_seq: return

        try:
            alignments = self.aligner.align(clip_seq, target_seq)
        except Exception:
            return
        if len(alignments) == 0:
            return

        best_aln = alignments[0]
        if not best_aln.aligned or len(best_aln.aligned) < 2:
            return

        query_blocks, target_blocks = best_aln.aligned
        if len(target_blocks) == 0:
            return

        if mode == 'right':
            aln_start_in_target = int(target_blocks[0][0])
            realign_pos = t_start + aln_start_in_target
            svlen = realign_pos - anchor_pos
        else:
            aln_end_in_target = int(target_blocks[-1][1])
            realign_pos = t_start + aln_end_in_target
            svlen = anchor_pos - realign_pos

        if svlen > 30:
            results.append(svlen)


_WORKER_REFINER = None


def _init_refiner_worker(bam_path, ref_path, insert_mean, insert_std):
    global _WORKER_REFINER
    _WORKER_REFINER = SpritesRefiner(bam_path, ref_path, insert_mean, insert_std)


def _refine_del_task(task):
    idx, chrom, pos, stop = task
    refined_len, method = _WORKER_REFINER.refine_deletion(chrom, pos, stop)
    return idx, refined_len, method


def refine_del_records_parallel(del_tasks, bam_path, ref_path, insert_mean, insert_std, processes=1):
    if not del_tasks:
        return {}

    worker_count = max(1, int(processes or 1))
    if worker_count == 1:
        refiner = SpritesRefiner(bam_path, ref_path, insert_mean, insert_std)
        try:
            results = {}
            for idx, chrom, pos, stop in del_tasks:
                refined_len, method = refiner.refine_deletion(chrom, pos, stop)
                results[idx] = (refined_len, method)
            return results
        finally:
            refiner.bam.close()
            refiner.ref.close()

    with mp.Pool(
        processes=worker_count,
        initializer=_init_refiner_worker,
        initargs=(bam_path, ref_path, insert_mean, insert_std),
    ) as pool:
        mapped = pool.imap_unordered(_refine_del_task, del_tasks, chunksize=8)
        results = {}
        for idx, refined_len, method in mapped:
            results[idx] = (refined_len, method)
        return results


# ==========================================
# 模块 3: 主流程
# ==========================================
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vcf", default="./final_adaptive_validated.vcf")
    parser.add_argument("-b", "--bam", default="/mnt/HHD_16T_1/lyl/data/HG002_GRCh37/HG002.hs37d5.sorted.bam")
    parser.add_argument("-r", "--ref", default="/mnt/HHD_16T_1/lyl/data/HG002_GRCh37/hs37d5.fa")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-p", "--processes", type=int, default=1)
    args = parser.parse_args(argv)

    print(f"Analyzing BAM stats from {args.bam}...")
    stats = BamStatCalculator(args.bam)
    with pysam.VariantFile(args.vcf) as vcf_scan:
        records = list(vcf_scan)

    del_tasks = []
    for idx, record in enumerate(records):
        is_del = (record.info.get('SVTYPE') == 'DEL') or (record.alts and '<DEL>' in str(record.alts[0]))
        if is_del:
            del_tasks.append((idx, record.chrom, record.pos, record.stop))

    print(f"Starting Hybrid Refinement (SR + DP) with {max(1, int(args.processes or 1))} worker(s)...")
    refine_results = refine_del_records_parallel(
        del_tasks,
        bam_path=args.bam,
        ref_path=args.ref,
        insert_mean=stats.mean,
        insert_std=stats.std,
        processes=args.processes,
    )

    vcf_in = pysam.VariantFile(args.vcf)
    try:
        vcf_in.header.info.add("SPRITES_LEN", 1, "Integer", "Refined length by SpritesModel")
        vcf_in.header.info.add("EVIDENCE", 1, "String", "Evidence type: SR (SplitRead) or DP (DiscordantPair)")
        vcf_in.header.info.add("IMPRECISE", 0, "Flag", "Imprecise structural variation")
    except:
        pass

    vcf_out = pysam.VariantFile(args.out, 'w', header=vcf_in.header)
    counts = {'total': 0, 'kept': 0, 'drop_ins': 0, 'drop_fp': 0, 'rescued_dp': 0, 'passthrough_non_del': 0}

    for idx, record in enumerate(vcf_in):
        counts['total'] += 1

        # 1. 仅处理 DEL
        is_del = (record.info.get('SVTYPE') == 'DEL') or \
                 (record.alts and '<DEL>' in str(record.alts[0]))
        if not is_del:
            counts['drop_ins'] += 1
            counts['passthrough_non_del'] += 1
            vcf_out.write(record)
            counts['kept'] += 1
            continue

        refined_len, method = refine_results.get(idx, (None, None))

        # 3. 结果处理
        if refined_len is None:
            counts['drop_fp'] += 1
            vcf_out.write(record)
            counts['kept'] += 1
            continue

        # 更新 VCF 记录
        record.info['SVLEN'] = -refined_len
        record.info['SPRITES_LEN'] = -refined_len
        record.stop = record.pos + refined_len

        # 记录证据类型
        record.info['EVIDENCE'] = method

        if method == 'SR':
            # Split Read 证据确凿，移除 IMPRECISE 标记 (如果是 Precise)
            if 'IMPRECISE' in record.info:
                del record.info['IMPRECISE']
        elif method == 'DP':
            # Discordant Pair 救回的结果，标记为 IMPRECISE
            record.info['IMPRECISE'] = True
            counts['rescued_dp'] += 1

        vcf_out.write(record)
        counts['kept'] += 1

        if counts['total'] % 100 == 0:
            sys.stdout.write(
                f"\rProcessed {counts['total']} | Kept: {counts['kept']} (Rescued by DP: {counts['rescued_dp']})")
            sys.stdout.flush()

    vcf_out.close()
    vcf_in.close()
    print("\n" + "=" * 40)
    print(f"Final Statistics:")
    print(f"  Total Input:    {counts['total']}")
    print(f"  Non-DEL passthrough: {counts['passthrough_non_del']}")
    print(f"  DEL without added evidence kept: {counts['drop_fp']}")
    print(f"  Kept Total:     {counts['kept']}")
    print(f"    - Precise (SR): {counts['kept'] - counts['rescued_dp']}")
    print(f"    - Rescued (DP): {counts['rescued_dp']}")
    print("=" * 40)


if __name__ == "__main__":
    main()

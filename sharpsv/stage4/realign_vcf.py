import sys
import pysam
import numpy as np
from Bio.Align import PairwiseAligner
import argparse
import collections
import multiprocessing as mp


# ==========================================
# Module 1: BAM statistics
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
                    # Use only reliable intra-chromosomal proper pairs.
                    if read.is_proper_pair and not read.is_duplicate and read.mapping_quality > 20:
                        if read.next_reference_id == read.reference_id and read.next_reference_start > read.reference_start:
                            tlen = abs(read.template_length)
                            # Ignore extreme fragments when estimating the insert profile.
                            if tlen < 2000:
                                inserts.append(tlen)
                                count += 1
            if inserts:
                # Median-based insert estimation is more robust to noisy tails.
                self.mean = np.median(inserts)
                self.std = np.std(inserts)
                print(f"[Info] Bam Stats -> Mean: {self.mean:.2f}, STD: {self.std:.2f}")
            else:
                print("[Warning] Could not calc stats, using defaults.")
        except Exception as e:
            print(f"[Error] Stat calculation failed: {e}")


# ==========================================
# Module 2: hybrid DEL refiner
# ==========================================
class SpritesRefiner:
    def __init__(self, bam_path, ref_path, insert_mean, insert_std):
        self.bam = pysam.AlignmentFile(bam_path, "rb")
        self.ref = pysam.FastaFile(ref_path)
        self.insert_mean = insert_mean
        self.insert_std = insert_std
        self.aligner = self._build_aligner()

        # Search parameters.
        self.search_windows = [100, 500, 1000]
        self.min_clip_len = 5
        self.min_mapq = 20

        # Threshold for discordant-pair rescue.
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
            return int(np.median(len_list))

        # Cluster split-read estimates before taking the consensus.
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

        # Suppress isolated outliers when the evidence set is larger.
        if len(best_cluster) < 2 and len(len_list) > 3: return None
        return int(np.median(best_cluster))

    def refine_deletion(self, vcf_chrom, vcf_pos, vcf_end):
        """
        Return (refined_length, method).
        Method is 'SR' for split-read refinement or 'DP' for discordant-pair rescue.
        """

        # Stage 1: precise split-read refinement.
        sr_lens = []
        for radius in self.search_windows:
            if len(sr_lens) >= 5: break

            start = max(0, vcf_pos - radius)
            end = vcf_pos + radius

            try:
                reads = self.bam.fetch(vcf_chrom, start, end)
            except:
                continue

            for read in reads:
                # Basic read filtering.
                if read.is_unmapped or read.is_duplicate or read.mapping_quality < self.min_mapq:
                    continue
                if not read.is_proper_pair:
                    continue

                direction, clip_seq, anchor_pos = self.get_soft_clip_seq(read)
                if not anchor_pos: continue

                mate_pos = read.next_reference_start
                if read.next_reference_id != read.reference_id: continue

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

        if consensus_sr is not None:
            return consensus_sr, 'SR'

        # Stage 2: discordant-pair rescue when split-read evidence is absent.
        dp_lens = self.rescue_by_discordant_pairs(vcf_chrom, vcf_pos, vcf_end)
        consensus_dp = self.get_consensus_len(dp_lens, method="median")

        if consensus_dp is not None:
            return consensus_dp, 'DP'

        return None, None

    def rescue_by_discordant_pairs(self, chrom, start, end):
        """
        Estimate deletion length from abnormally large template lengths.
        Observed_TLEN = physical_insert + deletion_size
        """
        estimated_lens = []

        # Search around the left breakpoint where spanning pairs are expected.
        search_start = max(0, start - 200)
        search_end = start + 200

        try:
            reads = self.bam.fetch(chrom, search_start, search_end)
        except:
            return []

        for read in reads:
            # Basic filtering.
            if read.is_unmapped or read.is_duplicate or read.mapping_quality < self.min_mapq:
                continue
            if not read.is_proper_pair:
                pass

            tlen = abs(read.template_length)

            if tlen > self.discordant_threshold:
                # Approximate deletion length from the insert-size excess.
                est_len = tlen - self.insert_mean

                if 50 < est_len < 100000:
                    estimated_lens.append(est_len)

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
# Module 3: CLI flow
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

        is_del = (record.info.get('SVTYPE') == 'DEL') or \
                 (record.alts and '<DEL>' in str(record.alts[0]))
        if not is_del:
            counts['drop_ins'] += 1
            counts['passthrough_non_del'] += 1
            vcf_out.write(record)
            counts['kept'] += 1
            continue

        refined_len, method = refine_results.get(idx, (None, None))

        if refined_len is None:
            counts['drop_fp'] += 1
            vcf_out.write(record)
            counts['kept'] += 1
            continue

        record.info['SVLEN'] = -refined_len
        record.info['SPRITES_LEN'] = -refined_len
        record.stop = record.pos + refined_len

        record.info['EVIDENCE'] = method

        if method == 'SR':
            if 'IMPRECISE' in record.info:
                del record.info['IMPRECISE']
        elif method == 'DP':
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

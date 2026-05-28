import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import pysam
from numba.typed import List

from .. import native as sharpsv_native
from ..utils.console import emit, emit_banner, format_duration

def baseinfo_guess(bamfile, contig, start, end, feature_count, maxcountread):
    nooverlap = True
    cigarlist = List()
    mdtaglist = List()
    corposlist = List()
    readtrustedarray = []
    qualityarray = []
    for AlignedSegment in bamfile.fetch(contig, start, end):
        read_length = AlignedSegment.infer_read_length()
        if read_length is not None and read_length > 0: # 24 25 行 为新增代码
            qualityarray.append([AlignedSegment.query_alignment_length / AlignedSegment.infer_read_length(),
                             AlignedSegment.query_alignment_length, AlignedSegment.mapping_quality])
            mdtaglist.append(AlignedSegment.get_tag('MD'))
            cigarlist.append(AlignedSegment.cigarstring)
            corposlist.append(List([AlignedSegment.reference_start - start, AlignedSegment.reference_end - start]))
            readtrustedarray.append(
                [AlignedSegment.query_alignment_length, AlignedSegment.infer_read_length(), AlignedSegment.mapping_quality])
            nooverlap = False

    if (nooverlap):
        raise ValueError("No overlap reads detected in BAM file for the specified region.")

    qualityarray = np.array(qualityarray, dtype='float32')
    qualityarray = qualityarray - qualityarray.min(axis=0)
    qualityarray = qualityarray / (qualityarray.max(axis=0) + 0.000001)

    return sharpsv_native.g_d(
        mdtaglist,
        cigarlist,
        corposlist,
        end - start,
        np.argsort(qualityarray.sum(axis=1))[::-1],
        maxcountread,
    ).astype("float32"), np.array(readtrustedarray)


def closerone(floatnumber):
    floor = int(floatnumber)
    top = floor + 1
    return np.array([floor, top])[np.argsort(abs(np.array([floor, top]) - floatnumber))[0]]


def guess_summary_depth(bamfilepath, times, window_size=1000, feature_count=9):
    meanlist = []
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads=20)
    contig2length = {}
    for count in range(len(bamfile.get_index_statistics())):
        contig2length[bamfile.get_index_statistics()[count].contig] = bamfile.lengths[count]
    orderarray = np.argsort([contig2length[contig] for contig in contig2length])[::-1]
    contigLIST = []

    for contig in contig2length:
        contigLIST.append(contig)
    contigLIST = np.array(contigLIST)[orderarray][:20]

    for i in range(times):
        while (True):
            contig = contigLIST[np.random.randint(0, len(contigLIST))]
            start = np.random.randint(contig2length[contig])
            try:
                data, tmptrustsumary = baseinfo_guess(bamfile, contig, start, start + window_size, feature_count, 2)
                meanlist.append(data.flatten())
            except:
                # print('Bad luck')
                continue
            break

    meanlist = np.stack(meanlist)
    return meanlist.reshape(meanlist.size // feature_count, feature_count).astype('float64').mean(axis=0).astype('int32')[-1]


def guess_summary(bamfilepath, times, workdir, window_size=1000, feature_count=9):
    meanlist = []
    trustsummary = []
    maxcountread = closerone(guess_summary_depth(bamfilepath, times, window_size=1000, feature_count=9))
    emit("stage-1/features", f"estimated representative depth at {maxcountread}")
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads=20)
    contig2length = {}
    for count in range(len(bamfile.get_index_statistics())):
        contig2length[bamfile.get_index_statistics()[count].contig] = bamfile.lengths[count]
    orderarray = np.argsort([contig2length[contig] for contig in contig2length])[::-1]
    contigLIST = []

    for contig in contig2length:
        contigLIST.append(contig)
    contigLIST = np.array(contigLIST)[orderarray][:20]

    for i in range(times):
        while (True):
            contig = contigLIST[np.random.randint(0, len(contigLIST))]
            start = np.random.randint(contig2length[contig])

            try:
                data, tmptrustsumary = baseinfo_guess(bamfile, contig, start, start + window_size, feature_count, maxcountread)
                meanlist.append(data.flatten())

            except:
                # print('Bad luck')
                continue
            break

    meanlist = np.stack(meanlist)
    np.save(workdir + 'meanarray',
            meanlist.reshape(meanlist.size // feature_count, feature_count).astype('float64').mean(axis=0, keepdims=True).astype('float32'))


def logify_numpy(a):
    return (np.log(((a>0)*a)+1.)-np.log((np.abs(a)*(a<0))+1.))


def combinelist(listoflist):  # require same type
    combinelist = listoflist[0]
    for onelist in listoflist[1:]:
        for item in onelist:
            combinelist.append(item)
    return combinelist


def decode_flag(flag):
    """
    解析 BAM flag 以确定二代测序 reads 的比对方向
    - 1: 负链（reverse strand）
    - 2: 正链（forward strand）
    """
    return 1 if (flag & 16) else 2


def baseinfo_AlignedSegment_child(sapresent, qualityarray, mdtaglist, cigarlist, corposlist, contig, start, end,
                                  primaryreadidcontigandsa, primaryssee, maxcountread, window_size, meanvalue, workdir):
    qualityarray = np.array(qualityarray, dtype='float32')
    qualityarray = qualityarray - qualityarray.min(axis=0)
    qualityarray = qualityarray / (qualityarray.max(axis=0) + 1e-7)

    if (sapresent == True):

        data, cluster_result, cluster_readcount = sharpsv_native.c_cw(
            mdtaglist,
            cigarlist,
            corposlist,
            start,
            end - start,
            primaryreadidcontigandsa,
            primaryssee,
            "victory",
            np.argsort(qualityarray.sum(axis=1))[::-1],
            maxcountread,
        )

        data = data.reshape(((end - start) // window_size, 9 * window_size))

    else:

        data, cluster_result, cluster_readcount = sharpsv_native.c_cn(
            mdtaglist,
            cigarlist,
            corposlist,
            start,
            end - start,
            "victory",
            np.argsort(qualityarray.sum(axis=1))[::-1],
            maxcountread,
        )

        data = data.reshape(((end - start) // window_size, 9 * window_size))

    mask = (data.sum(axis=1) != 0)
    index = np.arange(start, end, window_size)[mask]

    data = logify_numpy((data.reshape(data.size // 9, 9) - meanvalue).reshape(data.shape))[mask].astype('float16')

    bp = np.array([[cluster_result[loc][0] + start, cluster_result[loc][1], cluster_readcount[loc]] for loc in range(len(cluster_result))])

    try:
        np.savez_compressed(
            os.path.join(workdir, f"{contig}:{start}:{end}"),
            data=data,
            index=index,
            bp=bp,
        )
        np.savez_compressed(
            os.path.join(workdir, "tmp", f"{contig}:{start}:{end}"),
            data=np.array(0),
            index=np.array(0),
        )
    except:
        emit("stage-1/features", f"failed to save feature block {contig}:{start}:{end}")

    return 0



def baseinfo_AlignedSegment(genotype, bamfilepath, contig, r_start, r_end, meanvalue, window_size, maxcountread, workdir, INTERVAL):
    start = None
    preend = None
    with pysam.AlignmentFile(bamfilepath, 'rb') as bamfile:
        for AlignedSegment in bamfile.fetch(contig, r_start, r_end):
            start = AlignedSegment.reference_start
            preend = AlignedSegment.reference_end
            break

    if start is None or preend is None:
        return 0

    cigarlist = List()
    mdtaglist = List()
    corposlist = List()
    primaryreadidcontigandsa = List()
    primaryssee = List()
    qualityarray = []
    sapresent = False

    with pysam.AlignmentFile(bamfilepath, 'rb') as bamfile:
        for AlignedSegment in bamfile.fetch(contig, r_start, r_end):
            if (AlignedSegment.is_secondary == True):
                continue
            reference_start, reference_end = AlignedSegment.reference_start, AlignedSegment.reference_end

            preend = reference_end

            read_length = AlignedSegment.infer_read_length()
            if read_length is not None and read_length > 0: # 198 199 行 为新增代码
                qualityarray.append([AlignedSegment.query_alignment_length / AlignedSegment.infer_read_length(),
                                 AlignedSegment.query_alignment_length, AlignedSegment.mapping_quality])
                if (AlignedSegment.has_tag('MD') == True):
                    mdtaglist.append(AlignedSegment.get_tag('MD'))
                else:
                    mdtaglist.append(str(reference_end - reference_start))
                cigarlist.append(AlignedSegment.cigarstring)
                corposlist.append(List([reference_start - start, reference_end - start]))
                strandcode = decode_flag(AlignedSegment.flag)
                contig = AlignedSegment.reference_name
                readid = AlignedSegment.query_name
                if (AlignedSegment.has_tag('SA') == True):

                    sapresent = True
                    if ((strandcode % 2) == 0):
                        strandcode = 2
                    else:
                        strandcode = 1
                    primaryreadidcontigandsa.append(List([readid, contig, str(strandcode), AlignedSegment.get_tag('SA')]))
                    refstart, refend, readstart, readend = AlignedSegment.reference_start, AlignedSegment.reference_end, AlignedSegment.query_alignment_start, AlignedSegment.query_alignment_end
                    primaryssee.append(List([refstart, refend, readstart, readend]))

    if len(mdtaglist) == 0:
        return 0

    while (len(mdtaglist) != 0):

        if (os.path.isfile(workdir + 'meanarray.npy') == True):

            meanvalue = np.load(os.path.join(workdir, 'meanarray.npy')).astype('float32')
            maxcountread = closerone(meanvalue[0][-1])
            try:
                baseinfo_AlignedSegment_child(sapresent, qualityarray, mdtaglist, cigarlist, corposlist, contig, start,
                                              start + (((preend - start) // window_size) + 1) * window_size,
                                              primaryreadidcontigandsa, primaryssee, maxcountread, window_size,
                                              meanvalue, workdir)
            except:
                emit("stage-1/features", f"feature extraction error on {contig}:{r_start}-{r_end}")
            break
        else:
            time.sleep(2)


def available_worker_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


STAGE1_COMPLETE_MARKER = "stage1_npz.complete.json"


def _stage1_marker_path(workdir):
    return Path(workdir).expanduser().resolve() / STAGE1_COMPLETE_MARKER


def inspect_stage1_workdir(workdir):
    workdir_path = Path(workdir).expanduser().resolve()
    tmp_path = workdir_path / "tmp"
    root_npz = sorted(workdir_path.glob("*.npz"))
    tmp_npz = sorted(tmp_path.glob("*.npz")) if tmp_path.exists() else []
    root_names = {path.name for path in root_npz}
    tmp_names = {path.name for path in tmp_npz}
    marker_path = _stage1_marker_path(workdir_path)
    meanarray_path = workdir_path / "meanarray.npy"

    marker_exists = marker_path.exists()
    meanarray_exists = meanarray_path.exists()
    root_npz_count = len(root_npz)
    tmp_npz_count = len(tmp_npz)
    tmp_names_match = bool(root_names) and root_names == tmp_names

    reusable = False
    reason = "stage-1 outputs are missing"
    if marker_exists and meanarray_exists and root_npz_count > 0:
        reusable = True
        reason = "completion marker found"
    elif meanarray_exists and root_npz_count > 0 and tmp_names_match:
        reusable = True
        reason = "legacy outputs look complete (meanarray + matching tmp markers)"
    elif root_npz_count > 0:
        reason = "existing stage-1 outputs look incomplete; regeneration required"

    return {
        "workdir": str(workdir_path),
        "marker_path": str(marker_path),
        "marker_exists": marker_exists,
        "meanarray_exists": meanarray_exists,
        "root_npz_count": root_npz_count,
        "tmp_npz_count": tmp_npz_count,
        "tmp_names_match": tmp_names_match,
        "reusable": reusable,
        "reason": reason,
    }


def write_stage1_completion_marker(workdir):
    status = inspect_stage1_workdir(workdir)
    marker_path = Path(status["marker_path"])
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meanarray_exists": status["meanarray_exists"],
        "root_npz_count": status["root_npz_count"],
        "tmp_npz_count": status["tmp_npz_count"],
    }
    marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(marker_path)


def prepare_workdir(workdir):
    workdir_path = Path(workdir).expanduser().resolve()
    workdir_path.mkdir(parents=True, exist_ok=True)
    (workdir_path / "tmp").mkdir(exist_ok=True)

    for path in workdir_path.glob("*.npz"):
        path.unlink()

    meanarray = workdir_path / "meanarray.npy"
    if meanarray.exists():
        meanarray.unlink()

    for path in (workdir_path / "tmp").glob("*.npz"):
        path.unlink()

    marker_path = _stage1_marker_path(workdir_path)
    if marker_path.exists():
        marker_path.unlink()

    return str(workdir_path) + os.sep


def baseinfo_main(
    bamfilepath,
    workdir,
    max_worker=None,
    window_size=1000,
    interval=int(1e7),
    includecontig=None,
    guesstime=400,
    genotype=False,
    minsize=0,
):
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb')
    contig2length = {}
    for count in range(len(bamfile.get_index_statistics())):
        contig2length[bamfile.get_index_statistics()[count].contig] = bamfile.lengths[count]
    meanvalue, maxcountread = 0, 0
    bamfile.close()

    if max_worker is None:
        max_worker = available_worker_count()
    max_worker = max(min(max_worker, available_worker_count()), 1)
    emit("stage-1/features", f"CPU worker limit set to {max_worker}")
    workdir = prepare_workdir(workdir)

    emit("stage-1/features", f"feature corpus workdir: {workdir}")
    emit_banner(
        "Stage-1 Feature Synthesis",
        details=[
            ("bam", bamfilepath),
            ("workdir", workdir),
            ("cpu workers", max_worker),
            ("window size", window_size),
            ("target region stride", minsize if minsize else "auto"),
        ],
    )

    summary_process = multiprocessing.Process(target=guess_summary, args=(bamfilepath, guesstime, workdir))
    summary_process.start()

    if not includecontig:
        includecontig = [str(contig) for contig in contig2length]
    else:
        includecontig = [str(contig) for contig in includecontig if str(contig) in contig2length]

    if (minsize == 0):
        totalsize = 0
        for contig in includecontig:
            totalsize += contig2length[contig]
        minsize = max(int(min((totalsize // max_worker), interval)), 1)
    emit("stage-1/features", f"region stride set to {minsize}")

    worker_processes = []
    orderarray = np.argsort([contig2length[contig] for contig in includecontig])[::-1]
    for contigiloc in orderarray:
        contig = includecontig[contigiloc]
        if (contig2length[contig] < 100000):
            continue
        r_start = 0
        if (contig2length[contig] < 200000 or (max_worker == 1)):
            while (r_start < contig2length[contig]):
                emit("stage-1/features", f"processing region {contig}:{r_start}-{min(r_start + minsize, contig2length[contig])}")
                if ((r_start + int(interval)) > contig2length[contig]):
                    baseinfo_AlignedSegment(genotype, bamfilepath, contig, r_start, contig2length[contig] + 10000,
                                            meanvalue, window_size, maxcountread, workdir, interval)
                    r_start = contig2length[contig] + 10000
                else:
                    baseinfo_AlignedSegment(genotype, bamfilepath, contig, r_start, r_start + minsize, meanvalue,
                                            window_size, maxcountread, workdir, interval)
                    r_start += minsize
            continue

        while (r_start < contig2length[contig]):

            while (True):
                if (len(multiprocessing.active_children()) < (max_worker + 1)):
                    emit("stage-1/features", f"dispatching region {contig}:{r_start}-{min(r_start + minsize, contig2length[contig])}")
                    if ((r_start + int(interval)) > contig2length[contig]):
                        worker = multiprocessing.Process(
                            target=baseinfo_AlignedSegment,
                            args=(
                                genotype,
                                bamfilepath,
                                contig,
                                r_start,
                                contig2length[contig] + 10000,
                                meanvalue,
                                window_size,
                                maxcountread,
                                workdir,
                                interval,
                            ),
                        )
                        worker.start()
                        worker_processes.append(worker)
                        r_start = contig2length[contig] + 10000
                    else:
                        worker = multiprocessing.Process(
                            target=baseinfo_AlignedSegment,
                            args=(
                                genotype,
                                bamfilepath,
                                contig,
                                r_start,
                                r_start + minsize,
                                meanvalue,
                                window_size,
                                maxcountread,
                                workdir,
                                interval,
                            ),
                        )
                        worker.start()
                        worker_processes.append(worker)
                        r_start += minsize
                    break
                else:
                    time.sleep(2)

    summary_process.join()
    if summary_process.exitcode != 0:
        raise RuntimeError(f"Stage-1 summary worker exited with code {summary_process.exitcode}")

    for worker in worker_processes:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"Stage-1 worker exited with code {worker.exitcode}")

    write_stage1_completion_marker(workdir)

    return workdir


def build_parser():
    parser = argparse.ArgumentParser(description="SharpSV structural variant discovery pipeline")
    parser.add_argument("-bamfilepath", "--bamfilepath", required=True, help="Input sorted and indexed BAM file")
    parser.add_argument("-workdir", "--workdir", required=True, help="Directory for SharpSV temporary outputs")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    st = time.time()
    baseinfo_main(
        bamfilepath=args.bamfilepath,
        workdir=args.workdir,
    )
    emit("stage-1/features", f"feature generation completed in {format_duration(time.time() - st)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



















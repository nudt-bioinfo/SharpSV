import argparse
import csv
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sharpsv-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sharpsv-cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import numpy as np
import pandas as pd
import pysam
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d, label
try:
    from PIL import Image
except ImportError:
    Image = None

from ..bundled_models import resolve_runtime_checkpoint_path
from ..utils.console import emit, emit_banner, emit_progress, format_count, format_duration, render_bar
from .features import get_clip_num
from .image import _adapt_contig_id, estimate_insert_sizes, get_rgb, pipeup_column


torch.multiprocessing.set_sharing_strategy("file_system")

SLICE_SIZE = 50
SEQ_LEN = 20
WINDOW_SIZE = SLICE_SIZE * SEQ_LEN
DEFAULT_BATCH_SIZE = 8
DEFAULT_SV_LENGTH = "sv_1000"
FINAL_CSV_COLUMNS = ["chrom", "start", "end", "pred_sequence"]


def _safe_stage2_artifact_stem(chrom, start, end):
    chrom = str(chrom).replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{chrom}_{int(start)}_{int(end)}"


def _build_stage2_channel_contact_sheet(channel_frames, thumb_size=128, columns=5):
    channel_frames = np.asarray(channel_frames, dtype=np.uint8)
    rows = (len(channel_frames) + columns - 1) // columns
    sheet = np.zeros((rows * thumb_size, columns * thumb_size), dtype=np.uint8)
    step = max(channel_frames.shape[1] // thumb_size, 1)

    for idx, frame in enumerate(channel_frames):
        row = idx // columns
        col = idx % columns
        thumb = frame[::step, ::step][:thumb_size, :thumb_size]
        y0 = row * thumb_size
        x0 = col * thumb_size
        sheet[y0 : y0 + thumb.shape[0], x0 : x0 + thumb.shape[1]] = thumb

    return sheet


def _save_stage2_contact_sheet(seq_tensor, chrom, start, end, image_output_dir):
    if not image_output_dir:
        return

    image_output_dir = Path(image_output_dir).expanduser().resolve()
    image_output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stage2_artifact_stem(chrom, start, end)
    seq_array = seq_tensor.numpy()

    for channel_idx in range(seq_array.shape[1]):
        output_path = image_output_dir / f"{stem}.ch{channel_idx + 1}.png"
        sheet = _build_stage2_channel_contact_sheet(seq_array[:, channel_idx, :, :])

        if Image is not None:
            Image.fromarray(sheet, mode="L").save(output_path)
            continue

        from matplotlib import image as mpl_image

        mpl_image.imsave(output_path, sheet, cmap="gray", vmin=0, vmax=255)


def available_process_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _counter_value(counter):
    with counter.get_lock():
        return counter.value


def _counter_increment(counter, amount=1):
    with counter.get_lock():
        counter.value += amount
        return counter.value


def _safe_queue_size(queue):
    try:
        return queue.qsize()
    except Exception:
        return None


def _stage2_progress_payload(total_candidates, produced, consumed, skipped, failed, queue_size, started_at):
    elapsed = time.time() - started_at
    build_speed = produced / elapsed if elapsed > 0 else 0.0
    refine_speed = consumed / elapsed if elapsed > 0 else 0.0
    refine_eta = (total_candidates - consumed) / refine_speed if refine_speed > 0 else float("inf")
    staged = max(produced - consumed, 0)
    extras = [f"build {render_bar(produced, total_candidates, width=10)}"]
    extras.append(f"prep {build_speed:.1f}/s")
    extras.append(f"buffer {format_count(staged)}")
    extras.append(f"queue {queue_size}" if queue_size is not None else "queue n/a")
    if skipped:
        extras.append(f"skipped {format_count(skipped)}")
    if failed:
        extras.append(f"errors {format_count(failed)}")
    return refine_speed, refine_eta, "  ·  ".join(extras)


def _emit_stage2_progress(total_candidates, produced, consumed, skipped, failed, queue_size, started_at):
    refine_speed, refine_eta, extra = _stage2_progress_payload(
        total_candidates,
        produced,
        consumed,
        skipped,
        failed,
        queue_size,
        started_at,
    )
    emit_progress(
        "stage-2",
        consumed,
        total_candidates,
        speed=refine_speed,
        eta_seconds=refine_eta,
        extra=extra,
        width=18,
    )


def _monitor_stage2_progress(
    total_candidates,
    queue,
    produced_counter,
    consumed_counter,
    skipped_counter,
    failed_counter,
    stop_event,
    started_at,
    interval_seconds=5,
):
    while not stop_event.wait(interval_seconds):
        produced = _counter_value(produced_counter)
        consumed = _counter_value(consumed_counter)
        skipped = _counter_value(skipped_counter)
        failed = _counter_value(failed_counter)
        queue_size = _safe_queue_size(queue)
        _emit_stage2_progress(
            total_candidates,
            produced,
            consumed,
            skipped,
            failed,
            queue_size,
            started_at,
        )


def _write_empty_final_csv(output_path):
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=FINAL_CSV_COLUMNS).to_csv(output_path, index=False)
    return str(output_path)


def _load_candidate_sites(input_csv_path):
    input_csv_path = Path(input_csv_path).expanduser().resolve()
    if not input_csv_path.exists():
        raise FileNotFoundError(f"Intermediate CSV not found: {input_csv_path}")

    df = pd.read_csv(
        input_csv_path,
        low_memory=False,
        dtype={
            "chr": "string",
            "chrom": "string",
        },
    )
    if df.empty:
        return []

    candidates = []
    skipped_invalid = 0
    if {"chr", "position"}.issubset(df.columns):
        for row in df.itertuples(index=False):
            chrom_value = getattr(row, "chr")
            position_value = getattr(row, "position")
            if pd.isna(chrom_value) or pd.isna(position_value):
                skipped_invalid += 1
                continue
            chrom = str(chrom_value).strip()
            position = int(position_value)
            if not chrom or position <= 0:
                skipped_invalid += 1
                continue
            candidates.append((chrom, position, position + WINDOW_SIZE))
        if skipped_invalid:
            emit("stage-2", f"ignored {format_count(skipped_invalid)} invalid stage-1 candidate rows from {input_csv_path}")
        return candidates

    if {"chrom", "start", "end"}.issubset(df.columns):
        for row in df.itertuples(index=False):
            chrom_value = getattr(row, "chrom")
            start_value = getattr(row, "start")
            end_value = getattr(row, "end")
            if pd.isna(chrom_value) or pd.isna(start_value) or pd.isna(end_value):
                skipped_invalid += 1
                continue
            chrom = str(chrom_value).strip()
            start = int(start_value)
            end = int(end_value)
            if not chrom or start <= 0 or end <= start:
                skipped_invalid += 1
                continue
            candidates.append((chrom, start, end))
        if skipped_invalid:
            emit("stage-2", f"ignored {format_count(skipped_invalid)} invalid refinement candidate rows from {input_csv_path}")
        return candidates

    raise ValueError(
        f"Unsupported intermediate CSV columns in {input_csv_path}. Expected ['chr','position'] or ['chrom','start','end']."
    )


class PredictionRefiner:
    def __init__(
        self,
        smooth_sigma=0.6,
        del_anchor_conf=0.70,
        del_extend_conf=0.40,
        ins_conf_thresh=0.85,
        min_del_len=2,
        max_ins_width=10,
    ):
        self.sigma = smooth_sigma
        self.del_anchor = del_anchor_conf
        self.del_extend = del_extend_conf
        self.ins_thresh = ins_conf_thresh
        self.min_del_len = min_del_len
        self.max_ins_width = max_ins_width

    def process(self, logits):
        probs = F.softmax(logits, dim=2).cpu().numpy()
        batch_size, seq_len, _ = probs.shape
        refined_results = []

        for b in range(batch_size):
            p_del = gaussian_filter1d(probs[b, :, 1], sigma=self.sigma)
            p_ins = gaussian_filter1d(probs[b, :, 2], sigma=self.sigma * 0.5)
            final_seq = np.zeros(seq_len, dtype=int)

            candidates = (p_del > self.del_extend).astype(int)
            labeled, n_components = label(candidates)
            for i in range(1, n_components + 1):
                indices = np.where(labeled == i)[0]
                if len(indices) >= self.min_del_len and np.max(p_del[indices]) >= self.del_anchor:
                    final_seq[indices] = 1

            ins_candidates_mask = (p_ins > self.ins_thresh).astype(int)
            if np.sum(ins_candidates_mask) > 0:
                labeled_ins, n_ins = label(ins_candidates_mask)
                for i in range(1, n_ins + 1):
                    indices = np.where(labeled_ins == i)[0]
                    region_width = len(indices)
                    max_conf = np.max(p_ins[indices])
                    max_idx = indices[np.argmax(p_ins[indices])]
                    conflict = np.any(final_seq[indices] == 1)
                    if not conflict:
                        if region_width <= self.max_ins_width:
                            final_seq[max_idx] = 2
                    else:
                        if max_conf > 0.92 and region_width <= self.max_ins_width:
                            final_seq[max_idx] = 2

            refined_results.append(final_seq.tolist())
        return refined_results


def draw_pic_tensor(clip_dict_record, pile_record, del_pos_np_start, mapping_qualities_mean):
    image_size = (256, 256)
    original_multi_channel_image = np.zeros((12, image_size[0], image_size[1]), dtype=np.uint8)

    if not pile_record:
        compressed_multi_channel_image = np.zeros((3, 512, 512), dtype=np.uint8)
        return torch.tensor(compressed_multi_channel_image, dtype=torch.uint8)

    rgb_array = np.array([get_rgb(p, mapping_qualities_mean) for p in pile_record])
    x_starts = np.array([(p[0] - del_pos_np_start) * 5 + 5 for p in pile_record], dtype=int)

    y_start_index = 0
    old_x_start = 5
    for j, x_start in enumerate(x_starts):
        if x_start < 0 or x_start >= image_size[1]:
            continue

        if old_x_start == x_start:
            y_start = 5 + y_start_index * 5
            y_start_index += 1
        else:
            old_x_start = x_start
            y_start_index = 1
            y_start = 5

        x_end = x_start + 5
        y_end = y_start + 5
        if y_end > image_size[0]:
            continue

        y_slice = slice(y_start, y_end)
        x_slice = slice(x_start, x_end)
        original_multi_channel_image[:, y_slice, x_slice] = rgb_array[j][:, None, None]

    compressed_multi_channel_image = np.zeros((3, 512, 512), dtype=np.uint8)
    quadrant_size = 256
    for i in range(3):
        start_channel = i * 4
        compressed_multi_channel_image[i, :quadrant_size, :quadrant_size] = original_multi_channel_image[start_channel]
        compressed_multi_channel_image[i, :quadrant_size, quadrant_size:] = original_multi_channel_image[start_channel + 1]

        rotated_channel_3 = np.rot90(original_multi_channel_image[start_channel + 2], 2)
        rotated_channel_3 = np.fliplr(rotated_channel_3)
        compressed_multi_channel_image[i, quadrant_size:, :quadrant_size] = rotated_channel_3

        rotated_channel_4 = np.rot90(original_multi_channel_image[start_channel + 3], 2)
        rotated_channel_4 = np.fliplr(rotated_channel_4)
        compressed_multi_channel_image[i, quadrant_size:, quadrant_size:] = rotated_channel_4

    return torch.tensor(compressed_multi_channel_image, dtype=torch.uint8)


def _resolve_contigs(original_chr_id, sam_file, fasta_file):
    bam_chr = _adapt_contig_id(sam_file, original_chr_id)
    fasta_chr = _adapt_contig_id(fasta_file, original_chr_id, prefer_numeric=True)
    return bam_chr, fasta_chr


def process_candidate_chunk(
    chunk,
    bam_path,
    fasta_path,
    queue,
    mean_insert_size,
    sd_insert_size,
    mapping_qualities_mean,
    produced_counter,
    skipped_counter,
    failed_counter,
    image_output_dir=None,
):
    sam_file = pysam.AlignmentFile(bam_path, "rb")
    fasta_file = pysam.FastaFile(fasta_path)

    try:
        for candidate in chunk:
            original_chr_id, pos, _ = candidate
            pos = int(pos)
            win_start = pos
            win_end = pos + WINDOW_SIZE

            bam_chr, fasta_chr = _resolve_contigs(original_chr_id, sam_file, fasta_file)
            if bam_chr is None or fasta_chr is None:
                _counter_increment(skipped_counter)
                continue

            try:
                clip_record = get_clip_num(sam_file, bam_chr, win_start, win_end)
                clip_dict_record = dict(clip_record)

                seq_tensors = []
                for i in range(SEQ_LEN):
                    slice_start = win_start + i * SLICE_SIZE
                    slice_end = slice_start + SLICE_SIZE
                    pile_record = pipeup_column(
                        sam_file,
                        bam_chr,
                        slice_start,
                        slice_end,
                        mean_insert_size,
                        sd_insert_size,
                        fasta_file,
                        clip_dict_record,
                    )
                    tensor = draw_pic_tensor(clip_dict_record, pile_record, slice_start, mapping_qualities_mean)
                    seq_tensors.append(tensor)

                if len(seq_tensors) == SEQ_LEN:
                    seq_stack = torch.stack(seq_tensors)
                    _save_stage2_contact_sheet(seq_stack, original_chr_id, pos, win_end, image_output_dir)
                    queue.put((seq_stack, (original_chr_id, pos, win_end)))
                    _counter_increment(produced_counter)
            except Exception as exc:
                failure_count = _counter_increment(failed_counter)
                if failure_count <= 5:
                    emit("stage-2/cpu", f"candidate {original_chr_id}:{pos} skipped ({exc})")
                continue
    finally:
        sam_file.close()
        fasta_file.close()


def _get_devices(max_gpu_workers=2):
    if not torch.backends.cuda.is_built():
        emit(
            "stage-2",
            "CUDA unavailable because this PyTorch build has no CUDA support "
            f"(torch {torch.__version__}, torch.version.cuda={torch.version.cuda}). "
            "Image generation and refinement inference both run on CPU.",
        )
        return [torch.device("cpu")]

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        count = min(torch.cuda.device_count(), max_gpu_workers)
        devices = [torch.device(f"cuda:{i}") for i in range(count)]
        gpu_names = [torch.cuda.get_device_name(i) for i in range(count)]
        emit(
            "stage-2",
            f"image generation stays on CPU workers; inference uses GPU devices {devices}: {gpu_names}",
        )
        return devices
    emit(
        "stage-2",
        "CUDA runtime is unavailable to PyTorch "
        f"(torch {torch.__version__}, torch.version.cuda={torch.version.cuda}, "
        f"device_count={torch.cuda.device_count()}). Image generation and refinement inference both run on CPU.",
    )
    return [torch.device("cpu")]


def _load_refinement_model(checkpoint_path, device):
    from .model import SharpSVRefineSequenceModel

    runtime_checkpoint_path = resolve_runtime_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(runtime_checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    hyper_parameters = checkpoint.get("hyper_parameters") or {}
    state_dict.pop("criterion.weight", None)

    model = SharpSVRefineSequenceModel(
        data_root=hyper_parameters.get("data_root", "."),
        source_types=hyper_parameters.get("source_types", []),
        input_channels=hyper_parameters.get("input_channels", 3),
        num_classes=hyper_parameters.get("num_classes", 3),
        d_model=hyper_parameters.get("d_model", 512),
        nhead=hyper_parameters.get("nhead", 8),
        num_layers=hyper_parameters.get("num_layers", 4),
        lr=hyper_parameters.get("lr", 1e-4),
        batch_size=hyper_parameters.get("batch_size", DEFAULT_BATCH_SIZE),
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        emit("stage-2", f"checkpoint compatibility note: missing={missing}, unexpected={unexpected}")

    model.to(device)
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    return model


def _predict_batches(model, device, refiner, batch_tensors):
    batch = torch.stack(batch_tensors).to(device).float() / 255.0
    padding_mask = torch.zeros((len(batch_tensors), SEQ_LEN), dtype=torch.bool, device=device)
    with torch.no_grad():
        logits = model(batch, padding_mask)
    return refiner.process(logits)


def prediction_consumer(model_ckpt, batch_size, queue, output_csv, consumed_counter, max_gpu_workers=2):
    devices = _get_devices(max_gpu_workers=max_gpu_workers)
    models = [_load_refinement_model(model_ckpt, device) for device in devices]
    refiner = PredictionRefiner()

    output_csv = Path(output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FINAL_CSV_COLUMNS)

        buffers = [[] for _ in devices]
        infos = [[] for _ in devices]
        device_turn = 0

        while True:
            try:
                item = queue.get(timeout=20)
            except Exception:
                break

            if item == "DONE":
                break

            seq_tensor, info = item
            buffers[device_turn].append(seq_tensor)
            infos[device_turn].append(info)

            if len(buffers[device_turn]) >= batch_size:
                preds = _predict_batches(models[device_turn], devices[device_turn], refiner, buffers[device_turn])
                for (chrom, start, end), pred_seq in zip(infos[device_turn], preds):
                    writer.writerow([chrom, start, end, str(pred_seq)])
                    _counter_increment(consumed_counter)

                buffers[device_turn].clear()
                infos[device_turn].clear()
                device_turn = (device_turn + 1) % len(devices)

        for idx, buffer in enumerate(buffers):
            if not buffer:
                continue
            preds = _predict_batches(models[idx], devices[idx], refiner, buffer)
            for (chrom, start, end), pred_seq in zip(infos[idx], preds):
                writer.writerow([chrom, start, end, str(pred_seq)])
                _counter_increment(consumed_counter)


def refine_intermediate_csv(
    fasta_path,
    bam_path,
    input_csv_path,
    checkpoint_path,
    output_path,
    processes=None,
    batch_size=DEFAULT_BATCH_SIZE,
    sv_length=DEFAULT_SV_LENGTH,
    image_output_dir=None,
):
    started_at = time.time()
    fasta_path = str(Path(fasta_path).expanduser().resolve())
    bam_path = str(Path(bam_path).expanduser().resolve())
    runtime_checkpoint_path = resolve_runtime_checkpoint_path(checkpoint_path)
    output_path = str(Path(output_path).expanduser().resolve())

    if sv_length != DEFAULT_SV_LENGTH:
        raise ValueError(f"Only {DEFAULT_SV_LENGTH} is supported in the current SharpSV refinement pipeline.")

    for required_path in [fasta_path, bam_path, runtime_checkpoint_path]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required path not found: {required_path}")

    if processes is None:
        processes = available_process_count()
    processes = max(int(processes), 1)

    candidates = _load_candidate_sites(input_csv_path)
    if not candidates:
        emit("stage-2", "no stage-1 candidates were retained; writing an empty refinement CSV")
        return _write_empty_final_csv(output_path)

    mean_insert_size, sd_insert_size, mapping_qualities_mean = estimate_insert_sizes(bam_path)
    emit_banner(
        "Stage-2 Refinement",
        details=[
            ("candidates", format_count(len(candidates))),
            ("cpu workers", processes),
            ("batch size", batch_size),
            ("insert mean", mean_insert_size),
            ("insert std", sd_insert_size),
            ("mapq mean", mapping_qualities_mean),
            ("image export", image_output_dir or "disabled"),
        ],
    )
    chunk_size = max(1, len(candidates) // processes)
    chunks = [candidates[i : i + chunk_size] for i in range(0, len(candidates), chunk_size)]
    emit(
        "stage-2",
        f"launching {len(chunks)} CPU image builders and a GPU refinement consumer over {format_count(len(candidates))} candidate windows",
    )

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=min(max(len(chunks) * batch_size, 128), 2000))
    produced_counter = ctx.Value("i", 0)
    consumed_counter = ctx.Value("i", 0)
    skipped_counter = ctx.Value("i", 0)
    failed_counter = ctx.Value("i", 0)

    producers = []
    for chunk in chunks:
        producer = ctx.Process(
            target=process_candidate_chunk,
            args=(
                chunk,
                bam_path,
                fasta_path,
                queue,
                mean_insert_size,
                sd_insert_size,
                mapping_qualities_mean,
                produced_counter,
                skipped_counter,
                failed_counter,
                image_output_dir,
            ),
        )
        producer.start()
        producers.append(producer)

    max_gpu_workers = 2 if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 1
    consumer = ctx.Process(
        target=prediction_consumer,
        args=(checkpoint_path, batch_size, queue, output_path, consumed_counter, max_gpu_workers),
    )
    consumer.start()

    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_stage2_progress,
        args=(
            len(candidates),
            queue,
            produced_counter,
            consumed_counter,
            skipped_counter,
            failed_counter,
            monitor_stop,
            started_at,
        ),
        daemon=True,
    )
    monitor_thread.start()

    try:
        for producer in producers:
            producer.join()
            if producer.exitcode != 0:
                raise RuntimeError(f"Refinement worker exited with code {producer.exitcode}")

        queue.put("DONE")
        consumer.join()
        if consumer.exitcode != 0:
            raise RuntimeError(f"Refinement prediction process exited with code {consumer.exitcode}")
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=1.0)

    produced = _counter_value(produced_counter)
    consumed = _counter_value(consumed_counter)
    skipped = _counter_value(skipped_counter)
    failed = _counter_value(failed_counter)
    _emit_stage2_progress(
        len(candidates),
        produced,
        consumed,
        skipped,
        failed,
        _safe_queue_size(queue),
        started_at,
    )
    emit(
        "stage-2",
        f"refinement completed in {format_duration(time.time() - started_at)}; final CSV saved to {output_path}",
    )

    return output_path


def build_parser():
    parser = argparse.ArgumentParser(description="Run SharpSV stage-2 refinement from the stage-1 candidate CSV")
    parser.add_argument("-fasta_path", "--fasta_path", required=True, help="Reference genome FASTA path")
    parser.add_argument(
        "-bamfilepath",
        "--bamfilepath",
        "--bam_path",
        dest="bamfilepath",
        required=True,
        help="Input sorted and indexed BAM file",
    )
    parser.add_argument(
        "-inputcsv",
        "--inputcsv",
        "--vcf_path",
        dest="inputcsv",
        required=True,
        help="Stage-1 intermediate candidate CSV path",
    )
    parser.add_argument(
        "-checkpointpath",
        "--checkpointpath",
        "--model_ckpt",
        dest="checkpointpath",
        required=True,
        help="Stage-2 pretrained checkpoint path",
    )
    parser.add_argument("-output", "--output", "--output_csv", dest="output", required=True, help="Final CSV path")
    parser.add_argument(
        "--image-output-dir",
        dest="image_output_dir",
        default=None,
        help="Optional directory for per-candidate stage-2 image contact sheets.",
    )
    parser.add_argument(
        "-processes",
        "--processes",
        type=int,
        default=None,
        help="Worker process count. Defaults to all available CPUs.",
    )
    parser.add_argument("--sv_length", default=DEFAULT_SV_LENGTH, help=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    refine_intermediate_csv(
        fasta_path=args.fasta_path,
        bam_path=args.bamfilepath,
        input_csv_path=args.inputcsv,
        checkpoint_path=args.checkpointpath,
        output_path=args.output,
        processes=args.processes,
        batch_size=args.batch_size,
        sv_length=args.sv_length,
        image_output_dir=args.image_output_dir,
    )
    emit("stage-2", f"final refinement CSV saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

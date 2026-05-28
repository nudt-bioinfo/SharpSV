import numpy as np
import pandas as pd


def get_clip_num(sam_file, chr_id, pos_l, pos_r):
    clip_temp = []
    for read in sam_file.fetch(chr_id, pos_l, pos_r):
        if read.cigarstring is None:
            continue

        base_pos = read.get_reference_positions(True)
        read_len = len(base_pos)
        index = 0
        for read_map_pos in range(read_len):
            if base_pos[read_map_pos] is not None:
                break
            index += 1

        read_start = read.reference_start - index
        read_end = read_start + read_len - 1

        for i in range(pos_r - pos_l + 1):
            current_pos = pos_l + i
            if not (read_start <= current_pos <= read_end):
                continue

            index_ptr = read_start
            for cigar in read.cigartuples:
                if index_ptr <= current_pos < index_ptr + cigar[1]:
                    map_type = cigar[0]
                    clip_temp.append((current_pos, -map_type))
                    break
                index_ptr += cigar[1]

    if not clip_temp:
        return []

    clip_record_np = np.array(clip_temp)
    df = pd.DataFrame(clip_record_np)
    clip_record_df = df.groupby(0).sum()
    clip_record_df = clip_record_df // 4
    temp = clip_record_df.reset_index()
    return np.array(temp).tolist()


def get_depth(sam_file, chr_id, pos_l, pos_r):
    read_depth = sam_file.count_coverage(chr_id, pos_l, pos_r)
    depth = np.array(list(read_depth)).sum(axis=0)
    return list(depth)

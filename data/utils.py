import os


def get_tmp_filename(filename):
    tmp_name = f"tmp_{os.path.basename(filename)}"
    path = os.path.split(filename)[:-1]
    return os.path.join(*path, tmp_name)


def gaps_from_fasta(fasta_path, gaps_path):
    """
    Find gaps in fasta file from `fasta_path`
    and write them to `gaps_path`
    """
    cur_chr = None
    cur_N_start = None
    cur_idx = 0
    with open(fasta_path) as f:
        with open(gaps_path, "w") as f2:
            for line in f:
                if line[0] == ">":
                    if cur_N_start is not None:
                        f2.write(
                            "\t".join([cur_chr, str(cur_N_start), str(cur_idx - 1)])
                        )
                        f2.write("\n")
                    cur_chr = line[1:].rstrip()
                    cur_N_start = None
                    cur_idx = 0
                    # print(f'reached {cur_chr}')
                    continue
                for c in line.rstrip():
                    if c == "N" and cur_N_start is None:
                        cur_N_start = cur_idx
                    elif c != "N" and cur_N_start is not None:
                        f2.write("\t".join([cur_chr, str(cur_N_start), str(cur_idx)]))
                        f2.write("\n")
                        cur_N_start = None
                    cur_idx += 1


def elongate_intervals(short_intervals_path, long_intervals_path, padding=500):
    """
    Make intervals from `short_intervals_path` longer
    by `padding` from both sides and write to `long_intervals_path`
    """
    tmp_long_intervals_path = get_tmp_filename(long_intervals_path)
    with open(short_intervals_path) as f1:
        with open(tmp_long_intervals_path, "w") as f2:
            for line in f1:
                chrom, start, end = line.split()
                start = max(0, int(start) - padding)
                end = int(end) + padding
                f2.write(f"{chrom}\t{start}\t{end}\n")

    # merge intervals
    os.system(f"bedtools merge -i {tmp_long_intervals_path} > {long_intervals_path}")
    os.system(f"rm {tmp_long_intervals_path}")


def create_blacklist(fasta_gaps_path, encode_blacklist_path, blacklist_path):
    """
    Merge blacklists from fasta gaps file at `fasta_gaps_path`
    and ENCODE blacklist file `encode_blacklist_path`
    and write to `blacklist_path`
    """
    os.system(f"cat {fasta_gaps_path} {encode_blacklist_path} > {blacklist_path}")
    tmp_blacklist_path = get_tmp_filename(blacklist_path)
    os.system(
        f"sort -k1,1V -k2,2n -k3,3n {blacklist_path} | cut -f1-3 > {tmp_blacklist_path}"
    )
    os.system(f"bedtools merge -i {tmp_blacklist_path} > {blacklist_path}")
    os.system(f"rm {tmp_blacklist_path}")


def create_targets(
    target_features,
    all_targets_path,
    blacklist_intervals_path,
    target_path,
    target_sampling_intervals_path,
    distinct_features_path,
):
    """
    Create files with target feature intervals
    given intervals blacklist and target features

    Parameters
    ----------
    target_features : list(str)
        A list of features to be included in the target.
    all_targets_path : str
        Path to file with all intervals for all features.
    blacklist_intervals_path : str
        Path to file with blacklisted intervals.
    target_path : str
        File to create with only relevant target feature intervals included.
    target_sampling_intervals_path : str
        File to create with non-blacklisted intervals to sample from
        contatining at least one non-zero feature.
    distinct_features_path : str
        File to create with distinct triplets of `Cell|Feature|Cell_info`
        from `target_features`

    """

    distinct_features = set()
    tmp_target_path = get_tmp_filename(target_path)
    with open(all_targets_path) as f:
        with open(tmp_target_path, "w") as f2:
            for line in f:
                feature_name = line.split("|")[1]
                if feature_name in target_features:
                    f2.write(line)
                    distinct_feature = line.split()[-1].rstrip()
                    distinct_features.add(distinct_feature)

    with open(distinct_features_path, "w") as f:
        for distinct_feature in distinct_features:
            f.write(distinct_feature)
            f.write("\n")

    os.system(
        f"bedtools subtract -a {tmp_target_path} -b {blacklist_intervals_path} | \
        sort -k1,1V -k2,2n -k3,3n > {target_path}"
    )
    os.system(
        f"cut -f1-3 {target_path} | bedtools merge > {target_sampling_intervals_path}"
    )
    os.system(f"bgzip -c {target_path} > {target_path}.gz")
    os.system(f"rm {tmp_target_path}")


def full_target_file_pipeline(
    fasta_path,
    encode_blacklist_path,
    all_targets_path,
    target_features=["DNase"],
    target_path="sorted_data.bed",
    target_sampling_intervals_path="target_intervals.bed",
    distinct_features_path="distinct_features.txt",
):
    """
    Full pipeline of target files generation from target features
    """
    bed_files_path = os.path.join(*(os.path.split(target_path)[:-1]))

    # find gaps in fasta
    gaps_path = os.path.join(bed_files_path, "gaps.bed")
    gaps_from_fasta(fasta_path, gaps_path)

    # elongate gaps to avoid using
    # any unknown sites in the dataset
    long_gaps_path = os.path.join(bed_files_path, "long_gaps.bed")
    elongate_intervals(gaps_path, long_gaps_path, padding=500)

    # create blacklist from fasta gaps and ENCODE blacklist
    blacklist_path = os.path.join(bed_files_path, "blacklist.bed")
    create_blacklist(long_gaps_path, encode_blacklist_path, blacklist_path)

    # create target feature files from specific targets and file with all targets
    create_targets(
        target_features,
        all_targets_path,
        blacklist_path,
        target_path,
        target_sampling_intervals_path,
        distinct_features_path,
    )

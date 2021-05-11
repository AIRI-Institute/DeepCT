import os
import tempfile

CHROMOSOMES = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y", "M"]]


def interval_from_line(bed_line, pad_left=0, pad_right=0, chrom_counts=None):
    chrom, start, end = bed_line.rstrip().split("\t")[:3]
    start = max(0, int(start) - pad_left)
    if pad_right:
        end = min(int(end) + pad_right, chrom_counts[chrom])
    else:
        end = int(end)
    return chrom, start, end


def pad_interval_line(line, padding=0, chrom_counts=None):
    chrom, start, end = interval_from_line(line, padding, padding, chrom_counts)
    line_tail = line.rstrip().split("\t")[3:]
    padded_line = "\t".join([chrom, str(start), str(end), *line_tail])
    return f"{padded_line}\n"


def preprocess_fasta(fasta_path, gaps_path, chrom_ends_len=500, chrom_ends_path=None):
    """
    Find gaps in fasta file from `fasta_path` and write them to `gaps_path`.
    If `chrom_ends_len > 0`, write intervals of length `chrom_ends_len`
    at the start and end of each chromosome to `chrom_ends_path`.
    """
    cur_chr = None
    cur_N_start = None
    cur_idx = 0
    chr_counts = {chrom: 0 for chrom in CHROMOSOMES}
    with open(fasta_path) as f1, open(gaps_path, "w") as f2:
        for line in f1:
            if line[0] == ">":
                if cur_N_start is not None:
                    f2.write("\t".join([cur_chr, str(cur_N_start), str(cur_idx - 1)]))
                    f2.write("\n")
                if cur_chr is not None:
                    chr_counts[cur_chr] = cur_idx - 1
                cur_chr = line[1:].rstrip()
                cur_N_start = None
                cur_idx = 0
                continue
            for c in line.rstrip():
                if c == "N" and cur_N_start is None:
                    cur_N_start = cur_idx
                elif c != "N" and cur_N_start is not None:
                    f2.write("\t".join([cur_chr, str(cur_N_start), str(cur_idx)]))
                    f2.write("\n")
                    cur_N_start = None
                cur_idx += 1
    if chrom_ends_len > 0:
        with open(chrom_ends_path, "w") as f3:
            for chrom, chrom_cnt in chr_counts.items():
                if chrom_cnt == 0:
                    continue
                f3.write("\t".join([chrom, "0", str(chrom_ends_len)]))
                f3.write("\n")
                f3.write(
                    "\t".join([chrom, str(chrom_cnt - chrom_ends_len), str(chrom_cnt)])
                )
                f3.write("\n")


def get_chrom_counts(chrom_ends_path):
    """
    Load chromosome sizes from `chrom_ends_path` file with
    intervals at the ends of chromosomes
    """
    chrom_counts = {}
    with open(chrom_ends_path) as f:
        for line in f:
            split_line = line.split("\t")
            if split_line[1] != "0":
                chrom_counts[split_line[0]] = int(split_line[2])
    return chrom_counts


def elongate_intervals(
    short_intervals_path, long_intervals_path, padding=500, chrom_counts=None
):
    """
    Make intervals from `short_intervals_path` longer by `padding`
    from both sides (but not exceeding the chromosome size from `chrom_counts`)
    and write to `long_intervals_path`
    """
    with open(short_intervals_path) as f1, tempfile.NamedTemporaryFile(
        "w", delete=False
    ) as f2:
        for line in f1:
            padded_line = pad_interval_line(line, padding, chrom_counts)
            f2.write(padded_line)
    # merge intervals
    os.system(f"bedtools merge -i {f2.name} > {long_intervals_path}")
    os.unlink(f2.name)


def create_blacklist(
    fasta_gaps_path, encode_blacklist_path, blacklist_path, chrom_ends_path=None
):
    """
    Merge blacklists from fasta gaps file at `fasta_gaps_path`,
    ENCODE blacklist file `encode_blacklist_path`,
    and chromosome ends file at `chrom_ends_path`
    and write to `blacklist_path`
    """
    if chrom_ends_path is not None:
        os.system(
            f"cat {fasta_gaps_path} {encode_blacklist_path} {chrom_ends_path} > \
            {blacklist_path}"
        )
    else:
        os.system(f"cat {fasta_gaps_path} {encode_blacklist_path} > {blacklist_path}")
    f = tempfile.NamedTemporaryFile("w", delete=False)
    os.system(f"sort -k1,1V -k2,2n -k3,3n {blacklist_path} | cut -f1-3 > {f.name}")
    os.system(f"bedtools merge -i {f.name} > {blacklist_path}")
    f.close()
    os.unlink(f.name)


def create_targets(
    target_features,
    all_targets_path,
    blacklist_intervals_path,
    target_path,
    target_sampling_intervals_path,
    distinct_features_path,
    pad_targets=100,
    chrom_counts=None,
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
    pad_targets : int
        How much to pad target intervals with (to sample complete 0's)
    chrom_counts : dict(str: int)
        Size of each chromosome padded intervals are not supposed to go over
    """

    distinct_features = set()
    f = tempfile.NamedTemporaryFile("w", delete=False)
    with open(all_targets_path) as f1, tempfile.NamedTemporaryFile(
        "w", delete=False
    ) as f2, tempfile.NamedTemporaryFile("w", delete=False) as f3:
        for line in f1:
            feature_name = line.split("|")[1]
            if feature_name in target_features:
                padded_line = pad_interval_line(line, pad_targets, chrom_counts)
                if "chrM" in padded_line:
                    continue
                f2.write(line)
                f3.write(padded_line)
                distinct_feature = line.split("\t")[-1].rstrip()
                distinct_features.add(distinct_feature)
    distinct_features = sorted(distinct_features)
    with open(distinct_features_path, "w") as f:
        for distinct_feature in distinct_features:
            f.write(distinct_feature)
            f.write("\n")

    # write target data
    os.system(
        f"bedtools subtract -a {f2.name} -b {blacklist_intervals_path} | \
        sort -k1,1V -k2,2n -k3,3n > {target_path}"
    )
    os.unlink(f2.name)
    os.system(f"bgzip -c {target_path} > {target_path}.gz")
    os.system(f"tabix -s 1 -b 2 -e 3 {target_path}.gz")

    # write padded sampling intervals
    os.system(
        f"bedtools subtract -a {f3.name} -b {blacklist_intervals_path} | \
        sort -k1,1V -k2,2n -k3,3n | cut -f1-3 | bedtools merge > {target_sampling_intervals_path}"
    )
    os.unlink(f3.name)


def full_target_file_pipeline(
    fasta_path,
    encode_blacklist_path,
    all_targets_path,
    target_features_path,
    target_path="sorted_data.bed",
    target_sampling_intervals_path="target_intervals.bed",
    distinct_features_path="distinct_features.txt",
    elongate_encode_blacklist=True,
    blacklist_padding=500,
    target_padding=30,
):
    """
    Full pipeline of target files generation from target features
    """
    bed_files_path = os.path.dirname(target_path)

    # find gaps and chromosome ends in fasta
    gaps_path = os.path.join(bed_files_path, "gaps.bed")
    chrom_ends_path = os.path.join(bed_files_path, "chrom_ends_blacklist.bed")
    preprocess_fasta(
        fasta_path,
        gaps_path,
        chrom_ends_len=blacklist_padding,
        chrom_ends_path=chrom_ends_path,
    )

    # load fasta chromosome counts
    chrom_counts = get_chrom_counts(chrom_ends_path)

    # elongate gaps to avoid using any unknown sites in the dataset
    long_gaps_path = os.path.join(bed_files_path, "long_gaps.bed")
    elongate_intervals(
        gaps_path, long_gaps_path, padding=blacklist_padding, chrom_counts=chrom_counts
    )

    # elongate encode blacklist to avoid using
    # any blacklisted sites in the dataset
    if elongate_encode_blacklist:
        long_encode_blacklist_path = os.path.join(
            bed_files_path, f"long_{os.path.basename(encode_blacklist_path)}"
        )
        elongate_intervals(
            encode_blacklist_path,
            long_encode_blacklist_path,
            padding=blacklist_padding,
            chrom_counts=chrom_counts,
        )
        encode_blacklist_path = long_encode_blacklist_path

    # create blacklist from fasta gaps and ENCODE blacklist
    blacklist_path = os.path.join(bed_files_path, "blacklist.bed")
    create_blacklist(
        long_gaps_path, encode_blacklist_path, blacklist_path, chrom_ends_path
    )

    # read a list of target features
    with open(target_features_path) as f:
        target_features = list(map(lambda x: x.rstrip(), f.readlines()))

    # create target feature files from specific targets and file with all targets
    create_targets(
        target_features,
        all_targets_path,
        blacklist_path,
        target_path,
        target_sampling_intervals_path,
        distinct_features_path,
        pad_targets=target_padding,
        chrom_counts=chrom_counts,
    )

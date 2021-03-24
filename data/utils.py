import os
import tempfile


def gaps_from_fasta(fasta_path, gaps_path, chrom_ends_len=500, chrom_ends_path=None):
    """
    Find gaps in fasta file from `fasta_path` and write them to `gaps_path`.
    If `chrom_ends_len > 0`, write intervals of length `chrom_ends_len`
    at the start and end of each chromosome to `chrom_ends_path`
    """
    cur_chr = None
    cur_N_start = None
    cur_idx = 0
    chr_counts = {}
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
                f3.write("\t".join([chrom, "0", str(chrom_ends_len)]))
                f3.write("\n")
                f3.write(
                    "\t".join([chrom, str(chrom_cnt - chrom_ends_len), str(chrom_cnt)])
                )
                f3.write("\n")


def elongate_intervals(short_intervals_path, long_intervals_path, padding=500):
    """
    Make intervals from `short_intervals_path` longer
    by `padding` from both sides and write to `long_intervals_path`
    """
    with open(short_intervals_path) as f1, tempfile.NamedTemporaryFile(
        "w", delete=False
    ) as f2:
        for line in f1:
            split_line = line.rstrip().split("\t")
            chrom, start, end = split_line[:3]
            line_tail = split_line[4:]
            start = max(0, int(start) - padding)
            end = int(end) + padding
            f2.write("\t".join([chrom, str(start), str(end), *line_tail]))
            f2.write("\n")
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
    f = tempfile.NamedTemporaryFile("w", delete=False)
    with open(all_targets_path) as f1, tempfile.NamedTemporaryFile(
        "w", delete=False
    ) as f2:
        for line in f1:
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
        f"bedtools subtract -a {f2.name} -b {blacklist_intervals_path} | \
        sort -k1,1V -k2,2n -k3,3n > {target_path}"
    )
    os.unlink(f2.name)
    os.system(
        f"cut -f1-3 {target_path} | bedtools merge > {target_sampling_intervals_path}"
    )
    os.system(f"bgzip -c {target_path} > {target_path}.gz")


def full_target_file_pipeline(
    fasta_path,
    encode_blacklist_path,
    all_targets_path,
    target_features=["DNase"],
    target_path="sorted_data.bed",
    target_sampling_intervals_path="target_intervals.bed",
    distinct_features_path="distinct_features.txt",
    elongate_encode_blacklist=True,
    interval_padding=500,
):
    """
    Full pipeline of target files generation from target features
    """
    bed_files_path = os.path.dirname(target_path)

    # find gaps and chromosome ends in fasta
    gaps_path = os.path.join(bed_files_path, "gaps.bed")
    chrom_ends_path = os.path.join(bed_files_path, "chrom_ends.bed")
    gaps_from_fasta(
        fasta_path,
        gaps_path,
        chrom_ends_len=interval_padding,
        chrom_ends_path=chrom_ends_path,
    )

    # elongate gaps to avoid using any unknown sites in the dataset
    long_gaps_path = os.path.join(bed_files_path, "long_gaps.bed")
    elongate_intervals(gaps_path, long_gaps_path, padding=interval_padding)

    # elongate encode blacklist to avoid using
    # any blacklisted sites in the dataset
    if elongate_encode_blacklist:
        long_encode_blacklist_path = os.path.join(
            bed_files_path, f"long_{os.path.basename(encode_blacklist_path)}"
        )
        elongate_intervals(
            encode_blacklist_path, long_encode_blacklist_path, padding=interval_padding
        )
        encode_blacklist_path = long_encode_blacklist_path

    # create blacklist from fasta gaps and ENCODE blacklist
    blacklist_path = os.path.join(bed_files_path, "blacklist.bed")
    create_blacklist(
        long_gaps_path, encode_blacklist_path, blacklist_path, chrom_ends_path
    )

    # create target feature files from specific targets and file with all targets
    create_targets(
        target_features,
        all_targets_path,
        blacklist_path,
        target_path,
        target_sampling_intervals_path,
        distinct_features_path,
    )

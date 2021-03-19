import json

from utils import full_target_file_pipeline


def create_mini_fasta(fasta_path, mini_fasta_path, mini_fasta_counts_path):
    """
    Sample some lines from `fasta_path` to create mini-genome
    and write to `mini_fasta_path`
    """
    chr_counts = {}
    with open(fasta_path) as f1, open(mini_fasta_path, "w") as f2:
        chr_cnt = 0
        chrom = "chr1"
        for i, line in enumerate(f1):
            if line[0] == ">":
                f2.write(line)
                if chr_cnt != 0:
                    chr_counts[chrom] = chr_cnt

                chrom = line.rstrip()[1:]
                chr_cnt = 0
            else:
                if i % 10000 == 0:
                    f2.write(line)
                    chr_cnt += len(line.rstrip())
        chr_counts[chrom] = chr_cnt

    with open(mini_fasta_counts_path, "w") as f:
        json.dump(chr_counts, f)


def create_mini_targets(
    all_targets_path, mini_fasta_counts_path, mini_targets_path, sequence_length=100
):
    """
    Sample some lines from `all_targets_path` to create features
    for mini-genome with chromosome lengths given by `mini_fasta_counts_path`
    and write to `mini_targets_path`
    """
    with open(mini_fasta_counts_path) as f:
        mini_chroms = json.load(f)

    with open(all_targets_path) as f1, open(mini_targets_path, "w") as f2:
        for line in f1:
            chrom, start, end, feature = line.rstrip().split("\t")[:4]
            # move feature intervals
            start = max(0, int(start) - 10000)
            end = max(0, int(end) - 10000)
            # make sure we could still sample
            radius = sequence_length / 2
            if start < radius or end + radius > mini_chroms[chrom]:
                continue
            if (
                start < mini_chroms[chrom]
                and end <= mini_chroms[chrom]
                and start != end
            ):
                f2.write("\t".join([chrom, str(start), str(end), feature]))
                f2.write("\n")


def create_test_data(
    fasta_path,
    all_targets_path,
    target_features=["DNase"],
    mini_fasta_path="test_data/mini_male.hg19.fasta",
    mini_fasta_counts_path="test_data/mini_male.hg19.json",
    mini_targets_path="test_data/mini_all_sorted_data.bed",
    encode_blacklist_path="long_hg19_blacklist_ENCFF001TDO.bed",
    target_path="test_data/mini_sorted_data.bed",
    target_sampling_intervals_path="test_data/target_intervals.bed",
    distinct_features_path="test_data/distinct_features.txt",
):
    """
    Full pipeline to create test data from
    original fasta at `fasta_path` and features at `all_targets_path`
    """
    # create mini fasta
    create_mini_fasta(fasta_path, mini_fasta_path, mini_fasta_counts_path)
    # create mini targets for all features
    create_mini_targets(all_targets_path, mini_fasta_counts_path, mini_targets_path)
    # create test data
    full_target_file_pipeline(
        mini_fasta_path,
        encode_blacklist_path,
        mini_targets_path,
        target_features,
        target_path,
        target_sampling_intervals_path,
        distinct_features_path,
        elongate_encode_blacklist=False,
    )


if __name__ == "__main__":
    create_test_data(
        fasta_path="/mnt/datasets/DeepCT/male.hg19.fasta",
        all_targets_path="/mnt/datasets/DeepCT/all_features/sorted_deepsea_data.bed",
    )

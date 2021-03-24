from utils import full_target_file_pipeline

fasta_path = "/mnt/datasets/DeepCT/male.hg19.fasta"
encode_blacklist_path = "long_hg19_blacklist_ENCFF001TDO.bed"
all_targets_path = "/mnt/datasets/DeepCT/all_features/sorted_deepsea_data.bed"

full_target_file_pipeline(
    fasta_path,
    encode_blacklist_path,
    all_targets_path,
    target_features=["DNase"],
    target_path="sorted_data.bed",
    target_sampling_intervals_path="target_intervals.bed",
    distinct_features_path="distinct_features.txt",
    elongate_encode_blacklist=False,
    interval_padding=500,
)

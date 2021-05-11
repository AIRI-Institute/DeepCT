from utils import full_target_file_pipeline

DATA_DIR = "/mnt/datasets/DeepCT/dataset_data/all_features"

fasta_path = "/mnt/datasets/DeepCT/male.hg19.fasta"
encode_blacklist_path = "long_hg19_blacklist_ENCFF001TDO.bed"
all_targets_path = "/mnt/datasets/DeepCT/all_features/sorted_deepsea_data.bed"
target_features_path = f"{DATA_DIR}/target_features.txt"

full_target_file_pipeline(
    fasta_path,
    encode_blacklist_path,
    all_targets_path,
    target_features_path,
    target_path=f"{DATA_DIR}/sorted_data.bed",
    target_sampling_intervals_path=f"{DATA_DIR}/target_intervals.bed",
    distinct_features_path=f"{DATA_DIR}/distinct_features.txt",
    elongate_encode_blacklist=False,
    blacklist_padding=500,
    target_padding=30,
)

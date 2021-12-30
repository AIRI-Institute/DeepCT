import argparse
import os
import re
import sys

import pandas as pd

parser = argparse.ArgumentParser(description="Process datasets from Boix et al.")
parser.add_argument(
    "input", type=str, help="input directory path. Should contain bigWig files"
)
parser.add_argument("metadata", type=str, help="Path to metadata table fomr Boix et al")
parser.add_argument("out", type=str, help="output directory path")

args = parser.parse_args()
# create directories
os.makedirs(args.out, exist_ok=True)

metadata = pd.read_csv(args.metadata, sep="\t").set_index("id")
target_files = [
    os.path.join(args.input, f)
    for f in os.listdir(args.input)
    if os.path.isfile(os.path.join(args.input, f))
    and re.compile("FINAL.*\.bigWig").fullmatch(f) is not None
]

with open(
    os.path.join(args.out, "distinct_features.txt"), "w"
) as distinct_features, open(
    os.path.join(args.out, "target_features.txt"), "w"
) as target_features, open(
    os.path.join(args.out, "features2files_mapping.txt"), "w"
) as features2files_mapping:
    target_features_list = []
    for file in target_files:
        feature = os.path.basename(file).split("_")[1]
        cell_type = os.path.basename(file).split("_")[2][:-4]
        treatment = metadata.loc[cell_type]["perturb"]
        if pd.isna(treatment):
            treatment = "None"
        cell_type = metadata.loc[cell_type]["ct"]

        if treatment != "None" and treatment in cell_type:
            cell_type = cell_type.split(treatment)[0]
        if cell_type.endswith("_treated_with_"):
            cell_type = cell_type.split("_treated_with_")[0]
        full_cell_type = "|".join([cell_type, feature, str(treatment)])
        distinct_features.write(full_cell_type + "\n")
        target_features_list.append(feature)
        features2files_mapping.write(
            full_cell_type + "\t" + os.path.abspath(file) + "\n"
        )
    for i in set(target_features_list):
        target_features.write(i + "\n")

import argparse
import logging
import os
import re
import sys
import traceback

import pandas as pd
import pyBigWig
from pybedtools import BedTool

success_message = "successfully written to disk"

# process bigWig file and output data in bed format
def process_bigwig(
    bigWig_file, out_fname, cellTypeName, chrms_dict=None, threshold=4.4, peaksize=75
):
    """
    bigWig_file - input file name (str) to read data
    out_fname - output file name
    cellTypeName - string literal which will be written as 4th col of the output bed file
    chrms_dict - dict chrmName --> chrSize to check that current file has correct genome
                 if None chrms_dict from current file will be considered correct w/o check
    threshold - bigWig signal threshold above which the loci are considered as peaks
    peaksize - distance around peaking value considered as peak
    out_dir - directory to store output files

    Returns
    chrms_dict - dict of chrmName --> chrSize for subsequent assembly checks
    """

    def get_valid_chrms(bw):
        return {
            k: v
            for k, v in bw.chroms().items()
            if "rand" not in k and "chrM" not in k and "chrY" not in k
        }

    # open bigWig
    bw = pyBigWig.open(bigWig_file)

    # check chrm names and length
    if chrms_dict is not None:
        for chrm in chrms_dict:
            if chrm not in bw.chroms() or bw.chroms()[chrm] != chrms_dict[chrm]:
                logging.error(
                    "chromsome not found: " + chrm + " in file " + bigWig_file
                )
                raise KeyError
    else:
        chrms_dict = get_valid_chrms(bw)

    chrm_wise_beds = []
    for ind, chrm in enumerate(sorted(chrms_dict.keys())):  # iterate over chrms
        # get intervals with nonzero signal values and select those with value above threshold
        intervals = [i for i in bw.intervals(chrm) if i[2] > threshold]
        if len(intervals) == 0:
            logging.warning("No data on " + chrm + " in file " + bigWig_file)
            continue
        intervals = (
            pd.DataFrame(intervals)
            .drop(columns=2)
            .rename(columns={0: "start", 1: "end"})
        )

        # add chrm record
        intervals["chrm"] = [chrm] * len(intervals)
        # extend interval around peak by peaksize
        intervals["start"] = intervals["start"].apply(lambda x: max(x - peaksize, 0))
        intervals["end"] = intervals["end"].apply(
            lambda x: min(x + peaksize, chrms_dict[chrm])
        )
        intervals["cellType"] = cellTypeName

        # merge extended intervalas and save to temp file
        chrm_wise_beds.append(
            BedTool.from_dataframe(
                intervals[["chrm", "start", "end", "cellType"]]
            ).merge(c=4, o="first")
        )

    # concatenate all bed files
    if len(chrm_wise_beds) == 0:
        logging.error("No data in in file " + bigWig_file)
        sys.exit(2)
    BedTool.cat(*chrm_wise_beds, postmerge=False).moveto(out_fname)
    return chrms_dict


# parse log file to check wherher the output file aready exists
def check_log(logfile, f):
    with open(logfile) as fin:
        for line in fin:
            if (f in line) and (success_message in line):
                return True
    return False


parser = argparse.ArgumentParser(description="Process datasets from Boix et al.")
parser.add_argument(
    "input", type=str, help="input directory path. Should contain bigWig files"
)
parser.add_argument("metadata", type=str, help="Path to metadata table fomr Boix et al")
parser.add_argument(
    "--thr", type=float, help="threshold of bigWig peak's p-values", default=4.4
)
parser.add_argument(
    "--psize",
    type=int,
    help="pad each peak above threshold by psize nucleotides",
    default=75,
)
parser.add_argument("--out", type=str, help="output directory path", default="./out")
parser.add_argument("--log", type=str, help="log file path")
parser.add_argument(
    "--rewrite",
    action="store_true",
    default=False,
    help="reprocess files if output exists",
)
args = parser.parse_args()

# create directories
os.makedirs(args.out, exist_ok=True)

if args.rewrite:
    filemode = "w"
else:
    filemode = "a"

if args.log is None:
    args.log = os.path.join(args.out, "data_processing.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
    filename=args.log,
    filemode=filemode,
)
logging.getLogger("").addHandler(logging.StreamHandler())

target_files = [
    os.path.join(args.input, f)
    for f in os.listdir(args.input)
    if os.path.isfile(os.path.join(args.input, f))
    and re.compile("FINAL.*\.bigWig").fullmatch(f) is not None
]
logger = logging.getLogger("")
logger.info("Found %d target files" % len(target_files))

metadata = pd.read_csv(args.metadata, sep="\t").set_index("id")

with open(os.path.join(args.out, "all_target.txt"), filemode) as all_targets:
    chrms_dict = None
    for file in target_files:
        destination = os.path.abspath(
            os.path.join(args.out, os.path.basename(file)) + ".bed"
        )
        if (
            os.path.isfile(destination)
            and (check_log(args.log, destination))
            and not args.rewrite
        ):
            print("File %s already processed, skipping" % file)
            continue

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

        logging.info("Processing " + file)
        try:
            chrms_dict = process_bigwig(
                file, destination, full_cell_type, chrms_dict, args.thr, args.psize
            )
        except Exception as e:
            logging.error(str(e))
            traceback.print_exc()
            break
        all_targets.write(full_cell_type + "\n")
        logging.info(destination + " " + success_message)

# import cProfile
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyBigWig

TARGET_LENGTH = 1
SAMPLE_FILES = True
N_PROCESSES = 23
N_THREADS = 16
N_CONCURRENT = 30000
DATA_FOLDER = (
    "/mnt/datasets/DeepCT/dataset_data/Biox_et_al/complete_clique1_with0/samples/"
)
LOG_FOLDER = os.path.join(DATA_FOLDER, "logs")
SAMPLES_FILE = "complete_clique1_with0_samples_200.bed"
BIGWIG_FOLDER = "/mnt/10tb/home/vsfishman/TheCellState/DeepCT/data/Boix_data/"

SEQUENCE_LENGTH = 1000
TARGET_BIN_SIZE = 200
CONTEXT_LENGTH = (SEQUENCE_LENGTH - TARGET_LENGTH * TARGET_BIN_SIZE) // 2

CHROM_SIZES = {
    "chr1": 249250621,
    "chr2": 243199373,
    "chr3": 198022430,
    "chr4": 191154276,
    "chr5": 180915260,
    "chr6": 171115067,
    "chr7": 159138663,
    "chr8": 146364022,
    "chr9": 141213431,
    "chr10": 135534747,
    "chr11": 135006516,
    "chr12": 133851895,
    "chr13": 115169878,
    "chr14": 107349540,
    "chr15": 102531392,
    "chr16": 90354753,
    "chr17": 81195210,
    "chr18": 78077248,
    "chr19": 59128983,
    "chr20": 63025520,
    "chr21": 48129895,
    "chr22": 51304566,
    "chrX": 155270560,
    # 'chrY': 59373566,
    # 'chrM': 16571
}


def log_info(fd, info):
    fd.write("\t".join(map(str, info)) + "\n")
    fd.flush()


def read_track_values(chrom, chrom_size, track_file, track_idx):
    t1 = time.time()
    bw_file = pyBigWig.open(track_file)
    arr = bw_file.values(chrom, 0, chrom_size, numpy=True)
    t2 = time.time()
    return chrom, track_idx, arr, t2 - t1


def read_track_stats(
    chrom, start, end, track_file, track_idx, sample_idx, target_length=TARGET_LENGTH
):
    t1 = time.time()
    bw_file = pyBigWig.open(track_file)
    try:
        arr = bw_file.stats(
            chrom, int(start), int(end), type="mean", nBins=target_length, numpy=True
        )
    except RuntimeError:
        print(chrom, start, end, flush=True)
        import pdb

        pdb.set_trace()
    t2 = time.time()
    return chrom, track_idx, sample_idx, arr, t2 - t1


def create_dataset(
    bw_files, chrom_sizes, chrom_folder=".", n_threads=5, n_concurrent=100, log_fd=None
):
    chrom_arrays = {}
    with ThreadPoolExecutor(n_threads) as pool:
        futures_args = []
        for chrom, chrom_size in chrom_sizes.items():
            for track_idx, track_file in enumerate(bw_files):
                futures_args.append((chrom, chrom_size, track_file, track_idx))
            chrom_arrays[chrom] = np.memmap(
                f"{chrom}.arr",
                dtype="float32",
                mode="w+",
                shape=(len(bw_files), chrom_size),
            )

        log_fd.write(f"Ready to execute {len(futures_args)} tasks\n")
        log_fd.flush()
        for i in range(0, len(futures_args), n_concurrent):
            futures = []
            to_exec = futures_args[i : min(len(futures_args), i + n_concurrent)]
            for exec_args in to_exec:
                futures.append(pool.submit(read_track_values, *exec_args))
            for res in as_completed(futures):
                chrom, track_idx, arr, t = res.result()
                log_info(log_fd, ("r", chrom, track_idx, t))
                # log_fd.write(f'chrom {chrom} track {track_idx} retrieved in {t:.3f} seconds\n')#, flush=True)
                # log_fd.flush()
                t1 = time.time()
                chrom_arrays[chrom][track_idx, :] = arr
                t2 = time.time()
                del arr
                log_info(log_fd, ("w", chrom, track_idx, t2 - t1))
                # log_fd.write(f'wrote chrom {chrom} track {track_idx} in {t2 - t1:.3f} seconds\n')#, flush=True)
                # log_fd.flush()


def create_chrom_file(
    bw_files,
    chrom,
    chrom_size,
    chrom_folder,
    n_threads=5,
    n_concurrent=100,
    log_fd=None,
):
    with ThreadPoolExecutor(n_threads) as pool:
        futures_args = []
        for track_idx, track_file in enumerate(bw_files):
            futures_args.append((chrom, chrom_size, track_file, track_idx))
        chrom_array = np.memmap(
            os.path.join(chrom_folder, f"{chrom}.arr"),
            dtype="float32",
            mode="w+",
            shape=(len(bw_files), chrom_size),
        )

        log_fd.write(f"Ready to execute {len(futures_args)} tasks\n")
        log_fd.flush()
        for i in range(0, len(futures_args), n_concurrent):
            futures = []
            to_exec = futures_args[i : min(len(futures_args), i + n_concurrent)]
            for exec_args in to_exec:
                futures.append(pool.submit(read_track_values, *exec_args))
            for res in as_completed(futures):
                chrom, track_idx, arr, t = res.result()
                log_info(log_fd, ("r", chrom, track_idx, t))
                # log_fd.write(f'chrom {chrom} track {track_idx} retrieved in {t:.3f} seconds\n')#, flush=True)
                # log_fd.flush()
                t1 = time.time()
                chrom_array[track_idx, :] = arr
                t2 = time.time()
                log_info(log_fd, ("w", chrom, track_idx, t2 - t1))
                # log_fd.write(f'wrote chrom {chrom} track {track_idx} in {t2 - t1:.3f} seconds\n')#, flush=True)
                # log_fd.flush()


def futures_args_generator(samples, bw_files, n_concurrent=100):
    n_samples = len(samples)
    chrom_counts = samples.groupby("chrom")["start"].count().to_dict()
    chrom_sample_idxs = {chrom: -1 for chrom in chrom_counts}

    futures_args = []
    # for sample_idx in tqdm(range(n_samples)):
    for sample_idx in range(n_samples):
        sample = samples.iloc[sample_idx]
        chrom = sample["chrom"]
        chrom_sample_idxs[chrom] += 1
        start = sample["start"] + CONTEXT_LENGTH
        end = sample["end"] - CONTEXT_LENGTH
        for track_idx, track_file in enumerate(bw_files):
            futures_args.append(
                (chrom, start, end, track_file, track_idx, chrom_sample_idxs[chrom])
            )
            if len(futures_args) == n_concurrent:
                yield futures_args
                futures_args = []
    if futures_args:
        yield futures_args


def create_samples_dataset(
    bw_files, samples, chrom_folder=".", n_threads=5, n_concurrent=100, log_fd=None
):
    chrom_arrays = {}
    with ThreadPoolExecutor(n_threads) as pool:
        chrom_counts = samples.groupby("chrom")["start"].count().to_dict()
        for chrom in chrom_counts:
            chrom_arrays[chrom] = np.memmap(
                os.path.join(chrom_folder, f"{chrom}.arr"),
                dtype="float32",
                mode="w+",
                shape=(len(bw_files), chrom_counts[chrom], TARGET_LENGTH),
            )
        """
        # COMMENTED SO THAT WE DON'T generate all arguments and keep them in-memory
        # when we can yield from a generator
        n_samples = len(samples)
        futures_args = []
        chrom_sample_idxs = {chrom: -1 for chrom in chrom_counts}
        # for sample_idx in tqdm(range(n_samples)):
        for sample_idx in tqdm(range(n_samples)):
            sample = samples.iloc[sample_idx]
            chrom = sample["chrom"]
            chrom_sample_idxs[chrom] += 1
            start = sample["start"] + CONTEXT_LENGTH
            end = sample["end"] - CONTEXT_LENGTH
            for track_idx, track_file in enumerate(bw_files):
                futures_args.append(
                    (chrom, start, end, track_file, track_idx, chrom_sample_idxs[chrom])
                )
                # futures.append(pool.submit(
                    read_track_stats,
                    chrom,
                    start,
                    end,
                    track_file,
                    track_idx,
                    chrom_sample_idxs[chrom]
                ))
        """
        # print(f"Created {len(futures)} futures")
        # log_fd.write(f"Ready to execute {len(futures_args)} tasks\n")
        # log_fd.flush()
        for args_to_exec in futures_args_generator(
            samples, bw_files, n_concurrent=n_concurrent
        ):
            #        for i in range(0, len(futures_args), n_concurrent):
            # print('new concurrent cycle')
            futures = []
            # to_exec = futures_args[i:min(len(futures_args), i + n_concurrent)]
            for exec_args in args_to_exec:
                futures.append(pool.submit(read_track_stats, *exec_args))
            for res in as_completed(futures):
                chrom, track_idx, sample_idx, arr, t = res.result()
                log_info(log_fd, ("r", chrom, sample_idx, track_idx, t))
                # log_str = f'chrom {chrom} sample {sample_idx} track {track_idx} retrieved in {t:.3f} seconds\n')
                # log_fd.write(log_str, flush=True)
                # log_fd.flush()
                t1 = time.time()
                chrom_arrays[chrom][track_idx, sample_idx, :] = arr
                t2 = time.time()
                log_info(log_fd, ("w", chrom, sample_idx, track_idx, t2 - t1))
                # log_str = f'wrote chrom {chrom} sample {sample_idx} track {track_idx} in {t2 - t1:.3f} seconds\n')
                # log_fd.write(log_str, flush=True)
                # log_fd.flush()


def main(chrom):
    # chrom_sizes = CHROM_SIZES#{'chr10': 135534747}

    data_dir = BIGWIG_FOLDER
    bw_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".bigWig")
    ]  # [::100]

    chrom_folder = DATA_FOLDER
    samples = pd.read_csv(SAMPLES_FILE, sep="\t", header=None)
    samples.columns = ["chrom", "start", "end", "index"]
    samples = samples[samples["chrom"] == chrom]
    # print(len(samples))
    # samples = samples[::100]

    # create_dataset(bw_files, chrom_sizes, n_threads=8)
    # print('about to create')
    logfile = os.path.join(LOG_FOLDER, f"{chrom}_writer_log.txt")
    with open(logfile, "w") as log_fd:
        if SAMPLE_FILES:
            create_samples_dataset(
                bw_files,
                samples,
                chrom_folder=chrom_folder,
                n_threads=N_THREADS,
                log_fd=log_fd,
                n_concurrent=N_CONCURRENT,
            )
        else:
            create_chrom_file(
                bw_files,
                chrom,
                CHROM_SIZES[chrom],
                chrom_folder=chrom_folder,
                n_threads=N_THREADS,
                log_fd=log_fd,
                n_concurrent=N_CONCURRENT,
            )
    # chrom, track_idx, arr, t = read_track_values('chr10', 135534747, bw_files[0], 0)
    # print(t)


if __name__ == "__main__":
    # cProfile.run('main()')
    # main('chr10')
    with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        executor.map(main, CHROM_SIZES.keys())

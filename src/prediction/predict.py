"""
This module provides the `AnalyzeSequences` class and supporting
methods.
Based on selene's predict/AnalyzeSequence.py with minor modifications
"""
import math
import os
import warnings
from time import time

import numpy as np
import pandas as pd
import pyfaidx
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu
from selene_sdk.predict._common import (
    _pad_sequence,
    _truncate_sequence,
    get_reverse_complement,
    get_reverse_complement_encoding,
)

# from ._in_silico_mutagenesis import _ism_sample_id
# from ._in_silico_mutagenesis import in_silico_mutagenesis_sequences
# from ._in_silico_mutagenesis import mutate_sequence
# from ._variant_effect_prediction import _handle_long_ref
# from ._variant_effect_prediction import _handle_standard_ref
# from ._variant_effect_prediction import _handle_ref_alt_predictions
# from ._variant_effect_prediction import _process_alt
# from ._variant_effect_prediction import read_vcf_file
# from .predict_handlers import AbsDiffScoreHandler
# from .predict_handlers import DiffScoreHandler
# from .predict_handlers import LogitScoreHandler
from selene_sdk.predict.predict_handlers import (  # , WritePredictionsHandler_
    WritePredictionsHandler,
    WritePredictionsMultiCtBigWigHandler,
)
from selene_sdk.utils import initialize_logger, load_model_from_state_dict
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from src.dataset import _FEATURE_NOT_PRESENT

# from .predict_handlers import WriteRefAltHandler


def bh(pvals, fdr=0.05):
    ranks = np.argsort(pvals) + 1
    n_tests = len(pvals)
    corrected_pvals = []
    for i in range(len(pvals)):
        pval = pvals[i]
        rank = ranks[i]
        corrected_pval = rank / n_tests * fdr
        corrected_pvals.append(corrected_pval)
    corrected_pvals = np.array(corrected_pvals)
    bh_lower_idx = np.where((pvals < corrected_pvals) == True)
    fdr_pval_cutoff = pvals[bh_lower_idx].max()
    return corrected_pvals, fdr_pval_cutoff


class AnalyzeSequences(object):
    """
    Score sequences and their variants using the predictions made
    by a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        A sequence-based model architecture.
    trained_model_path : str or list(str)
        The path(s) to the weights file for a trained sequence-based model.
        For a single path, the model architecture must match `model`. For
        a list of paths, assumes that the `model` passed in is of type
        `selene_sdk.utils.MultiModelWrapper`, which takes in a list of
        models. The paths must be ordered the same way the models
        are ordered in that list. `list(str)` input is an API-only function--
        Selene's config file CLI does not support the `MultiModelWrapper`
        functionality at this time.
    sequence_length : int
        The length of sequences that the model is expecting.
    features : list(str)
        The names of the features that the model is predicting.
    batch_size : int, optional
        Default is 64. The size of the mini-batches to use.
    device : str
        Specifies device to use (i.e. 'cpu' or 'cuda:0').
    data_parallel : bool, optional
        Default is `False`. Specify whether multiple GPUs are available for
        torch to use during training.
    reference_sequence : class, optional
        Default is `selene_sdk.sequences.Genome`. The type of sequence on
        which this analysis will be performed. Please note that if you need
        to use variant effect prediction, you cannot only pass in the
        class--you must pass in the constructed `selene_sdk.sequences.Sequence`
        object with a particular sequence version (e.g. `Genome("hg19.fa")`).
        This version does NOT have to be the same sequence version that the
        model was trained on. That is, if the sequences in your variants file
        are hg19 but your model was trained on hg38 sequences, you should pass
        in hg19.
    write_mem_limit : int, optional
        Default is 5000. Specify, in MB, the amount of memory you want to
        allocate to storing model predictions/scores. When running one of
        _in silico_ mutagenesis, variant effect prediction, or prediction,
        prediction/score handlers will accumulate data in memory and only
        write this data to files periodically. By default, Selene will write
        to files when the total amount of data (across all handlers) takes up
        5000MB of space. Please keep in mind that Selene will not monitor the
        memory needed to actually carry out the operations (e.g. variant effect
        prediction) or load the model, so `write_mem_limit` should always be
        less than the total amount of CPU memory you have available on your
        machine. For example, for variant effect prediction, we load all
        the variants in 1 file into memory before getting the predictions, so
        your machine must have enough memory to accommodate that. Another
        possible consideration is your model size and whether you are
        using it on the CPU or a CUDA-enabled GPU (i.e. setting
        `use_cuda` to True).

    Attributes
    ----------
    model : torch.nn.Module
        A sequence-based model that has already been trained.
    trained_model_path: str
        A path to load trained model
    sequence_length : int
        The length of sequences that the model is expecting.
    n_cell_types: int
        The number of cell types used in model training
    batch_size : int
        The size of the mini-batches to use.
    features : list(str)
        The names of the features that the model is predicting.
    device : str
        Specifies device to use (i.e. 'cpu' or 'cuda:0').
    data_parallel : bool
        Whether to use multiple GPUs or not.
    reference_sequence : class
        The type of sequence on which this analysis will be performed.
    distinct_features: list(str) or None
        names of distinct features to infer cell type labels
        if None cells will be labaled by numbers in numerical order
    """

    def __init__(
        self,
        model,
        trained_model_path,
        sequence_length,
        features,
        n_cell_types,
        reference_sequence,
        center_bin=200,
        batch_size=64,
        device="cpu",
        data_parallel=False,
        write_mem_limit=500,
        distinct_features=None,
    ):
        """
        Constructs a new `AnalyzeSequences` object.
        """
        self.model = model
        self.device = torch.device(device)

        trained_model = torch.load(
            trained_model_path,
            map_location=self.device,  # lambda storage, location: storage
        )
        if "state_dict" in trained_model:
            self.model = load_model_from_state_dict(trained_model["state_dict"], model)
        else:
            self.model = load_model_from_state_dict(trained_model, model)

        self.model.eval()

        self.data_parallel = data_parallel

        if self.data_parallel:
            self.model = nn.DataParallel(model)
        else:
            self.model.to(self.device)

        self.sequence_length = sequence_length

        self._start_radius = sequence_length // 2
        self._end_radius = self._start_radius
        if sequence_length % 2 != 0:
            self._start_radius += 1
        self.center_bin = center_bin

        self.batch_size = batch_size
        self.features = features
        self.reference_sequence = reference_sequence
        self.n_cell_types = n_cell_types

        if distinct_features is not None:
            self._cell_types = []
            for distinct_feature_index, distinct_feature in enumerate(
                distinct_features
            ):
                feature_name, cell_type = self._parse_distinct_feature(distinct_feature)
                if feature_name not in self.features:
                    continue
                if cell_type not in self._cell_types:
                    self._cell_types.append(cell_type)
            self.n_target_features = len(self.features)
            self._feature_indices_by_cell_type_index = np.full(
                (self.n_cell_types, self.n_target_features), _FEATURE_NOT_PRESENT
            )
            for distinct_feature_index, distinct_feature in enumerate(
                distinct_features
            ):
                feature_name, cell_type = self._parse_distinct_feature(distinct_feature)
                if feature_name not in self.features:
                    continue
                feature_index = self.features.index(feature_name)
                cell_type_index = self._cell_types.index(cell_type)
                self._feature_indices_by_cell_type_index[cell_type_index][
                    feature_index
                ] = distinct_feature_index
        else:
            self._cell_types = list(map(str, range(self.n_cell_types)))

        assert self.n_cell_types == self.model._n_cell_types == len(self._cell_types)
        # self.cell_types_one_hot = np.zeros(shape=(self.n_cell_types, self.n_cell_types),
        #                                    dtype=np.uint8)   # note that we use float64 here
        #                                                      # to make it compatible with model's weights dtype later
        # self.cell_types_one_hot[np.diag_indices(self.n_cell_types)] = 1
        self._write_mem_limit = write_mem_limit
        print("Successfully created analyzer object")

    def _parse_distinct_feature(self, distinct_feature):
        """
        Parse a combination of `cell_type|feature_name|info` into
        `(feature_name, cell_type)`
        """
        feature_description = distinct_feature.split("|")
        feature_name = feature_description[1]
        cell_type = feature_description[0]
        addon = feature_description[2]
        if addon != "None":
            cell_type = cell_type + "_" + addon
        return feature_name, cell_type

    def _initialize_reporters(
        self,
        save_data,
        output_path_prefix,
        output_format,
        colnames_for_ids,
        output_size=None,
    ):
        """
        Initialize the handlers to which Selene reports model predictions

        Parameters
        ----------
        save_data : list(str)
            A list of the data files to output. Currently only: ["predictions"] supported.
        output_path_prefix : str
            Path to which the reporters will output data files. Selene will
            add a prefix to the resulting filename, where the prefix is based
            on the name of the user-specified input file. This allows a user
            to distinguish between output files from different inputs when
            a user specifies the same output directory for multiple inputs.
        output_format : {'tsv', 'hdf5','bigWig'}
            The desired output format. Currently Selene supports TSV and HDF5
            formats.
        colnames_for_ids : list(str)
            Specify the names of columns that will be used to identify the
            sequence for which Selene has made predictions (e.g. (chrom,
            pos, id, ref, alt) will be the column names for variant effect
            prediction outputs).
        output_size : int, optional
            The total number of rows in the output. Must be specified when
            the output_format is hdf5.

        Returns
        -------
        list(selene_sdk.predict.predict_handlers.PredictionsHandler)
            List of reporters to update as Selene receives model predictions.

        """
        assert save_data == ["predictions"]
        reporters = []

        if output_format in ["tsv", "hdf5"]:
            constructor_args = [
                self.features,
                colnames_for_ids,
                output_path_prefix,
                output_format,
                output_size,
                self._write_mem_limit // len(save_data),
            ]
            reporters.append(
                WritePredictionsHandler(*constructor_args, write_labels=True)
            )
        else:
            raise NotImplementedError
        return reporters

    def _get_sequences_from_bed_file(
        self,
        input_path,
        strand_index=None,
        sample_continuous=False,
        output_NAs_to_file=None,
        reference_sequence=None,
    ):
        """
        Get the adjusted sequence coordinates and labels corresponding
        to each row of coordinates in an input BED file. The coordinates
        specified in each row are only used to find the center position
        for the resulting sequence--all regions returned will have the
        length expected by the model.

        Parameters
        ----------
        input_path : str
            Input filepath to BED file.
        strand_index : int or None, optional
            Default is None. If sequences must be strand-specific,
            the input BED file may include a column specifying the
            strand ({'+', '-', '.'}).
        output_NAs_to_file : str or None, optional
            Default is None. Only used if `reference_sequence` is also not None.
            Specify a filepath to which invalid variants are written.
            Invalid = sequences that cannot be fetched, either because
            the exact chromosome cannot be found in the `reference_sequence` FASTA
            file or because the sequence retrieved is out of bounds or overlapping
            with any of the blacklist regions.
        reference_sequence : selene_sdk.sequences.Genome or None, optional
            Default is None. The reference genome.

        Returns
        -------
        list(tup), list(tup)
            The sequence query information (chrom, start, end, strand)
            and the labels (the index, genome coordinates, and sequence
            specified in the BED file).

        """
        sequences = []
        labels = []
        na_rows = []
        check_chr = True
        for chrom in reference_sequence.get_chrs():
            if not chrom.startswith("chr"):
                check_chr = False
                break
        with open(input_path, "r") as read_handle:
            for i, line in enumerate(read_handle):
                cols = line.strip().split()
                if len(cols) < 3:
                    na_rows.append(line)
                    continue
                chrom = cols[0]
                start = cols[1]
                end = cols[2]
                strand = "."
                if isinstance(strand_index, int) and len(cols) > strand_index:
                    strand = cols[strand_index]
                if "chr" not in chrom and check_chr is True:
                    chrom = "chr{0}".format(chrom)
                if (
                    not str.isdigit(start)
                    or not str.isdigit(end)
                    or chrom not in self.reference_sequence.genome
                ):
                    na_rows.append(line)
                    continue
                start, end = int(start), int(end)
                if sample_continuous:
                    for mid_pos in range(start, end + 1):
                        seq_start = mid_pos - self._start_radius
                        seq_end = mid_pos + self._end_radius
                        if reference_sequence:
                            if not reference_sequence.coords_in_bounds(
                                chrom, seq_start, seq_end
                            ):
                                na_rows.append(line)
                                continue
                        sequences.append((chrom, seq_start, seq_end, strand))
                        labels.append((i, chrom, seq_start, seq_end, strand))
                else:
                    mid_pos = start + ((end - start) // 2)
                    seq_start = mid_pos - self._start_radius
                    seq_end = mid_pos + self._end_radius
                    if reference_sequence:
                        if not reference_sequence.coords_in_bounds(
                            chrom, seq_start, seq_end
                        ):
                            na_rows.append(line)
                            continue
                    sequences.append((chrom, seq_start, seq_end, strand))
                    labels.append((i, chrom, start, end, strand))

        if reference_sequence and output_NAs_to_file:
            with open(output_NAs_to_file, "w") as file_handle:
                for na_row in na_rows:
                    file_handle.write(na_row)
        return sequences, labels

    def _predict(self, batch_sequences):
        """
        Return model predictions for a batch of sequences.
        This function extends predict function from selene_sdk/predict/_common.py
        to handle cell-wise models

        Parameters
        ----------
        batch_sequences : numpy.ndarray
            `batch_sequences` has the shape :math:`B \\times L \\times N`,
            where :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet.

        Returns
        -------
        numpy.ndarray
            The model predictions of shape :math:`B \\times F`, where :math:`F`
            is the number of features (classes) the model predicts.

        """

        # note that we need transpose to convert from
        # seqlen x alphabet to seqlen x alphabet, which is expected by model
        batch_sequences = (
            torch.from_numpy(batch_sequences).float().transpose(1, 2).to(self.device)
        )

        batch_cell_types = "dummy"  # note that to allow compatibility with older models
        # cell type batch is generated as diagona matrix N_cell_types x N_cell_types
        # inside the forward pass function of the model
        with torch.no_grad():
            outputs = self.model(batch_sequences, batch_cell_types)

        means = outputs[:, -1:, :]
        deviations = outputs[:, :-1, :]
        predictions = means + deviations
        # predictions = torch.sigmoid(outputs)
        return predictions

    def _get_predictions(self, sequences, batch_ids):
        # as soon as we got batch_size of different sequences
        # we pass them to the model as tensor batch_size x alphabet x seq_len and obtain
        # predictions tensor batch_size x n_cell_types x n_features
        # next we update batch ids to add info about features and cell types

        preds = self._predict(sequences)
        batch_size = sequences.shape[
            0
        ]  # note that for last batch it could be smaller than self.batch_size
        preds = (
            preds.cpu()
            .numpy()
            .reshape((batch_size * self.n_cell_types, len(self.features)))
        )

        # update prediction information (batch_ids) to add cell type and feature info

        cell_type_updated_batch_ids = [
            bi + (cell_type_id,)
            for bi in batch_ids
            for cell_type_id in range(self.n_cell_types)
        ]
        return preds, cell_type_updated_batch_ids

    def _get_bigWig_header(self, seq_coords):
        chrms = []
        sizes = []
        for coord in seq_coords:
            chrm, start, end = coord[:3]
            if not chrm in chrms:
                chrms.append(chrm)
                sizes.append(end)
                continue
            if chrm != chrms[-1] or end <= sizes[-1]:
                raise Exception("Unsorted sequences cann't be passed to bigWig writer")
            sizes[-1] = end
        return list(zip(chrms, sizes))

    def get_predictions_for_bed_file(
        self,
        input_path,
        output_dir,
        output_format="tsv",
        strand_index=None,
        sample_continuous=False,
    ):
        """
        Get model predictions for sequences specified as genome coordinates
        in a BED file. Coordinates do not need to be the same length as the
        model expected sequence input--predictions will be centered at the
        midpoint of the specified start and end coordinates if
        sample_continuous == False. If sample_continuous == True model will
        retrieve a sequence of required length for each point in range from
        start to end coordinate.

        Parameters
        ----------
        input_path : str
            Input path to the BED file.
        output_dir : str
            Output directory to write the model predictions.
        output_format : {'tsv', 'hdf5', 'bigWig'}, optional
            Default is 'tsv'. Choose whether to save TSV or HDF5 output files.
            TSV is easier to access (i.e. open with text editor/Excel) and
            quickly peruse, whereas HDF5 files must be accessed through
            specific packages/viewers that support this format (e.g. h5py
            Python package). Choose

                * 'tsv' if your list of sequences is relatively small
                  (:math:`10^4` or less in order of magnitude) and/or your
                  model has a small number of features (<1000).
                * 'hdf5' for anything larger and/or if you would like to
                  access the predictions/scores as a matrix that you can
                  easily filter, apply computations, or use in a subsequent
                  classifier/model. In this case, you may access the matrix
                  using `mat["data"]` after opening the HDF5 file using
                  `mat = h5py.File("<output.h5>", 'r')`. The matrix columns
                  are the features and will match the same ordering as your
                  features .txt file (same as the order your model outputs
                  its predictions) and the matrix rows are the sequences.
                  Note that the row labels (FASTA description/IDs) will be
                  output as a separate .txt file (should match the ordering
                  of the sequences in the input FASTA).

        strand_index : int or None, optional
            Default is None. If the trained model makes strand-specific
            predictions, your input file may include a column with strand
            information (strand must be one of {'+', '-', '.'}). Specify
            the index (0-based) to use it. Otherwise, by default '+' is used.

        sample_continuous : bool
            if True, each record in .bed file will be treated as continuous target
            interval for prediction, i.e. prediction will be generated for each
            point between start and end. If False, each record will be treated as
            single target region, and predicion will be generated for midpoint (between
            start and end), which was default selene behaviour.
        Returns
        -------
        None
            Writes the output to file(s) in `output_dir`. Filename will
            match that specified in the filepath.

        """
        _, filename = os.path.split(input_path)
        output_prefix = os.path.splitext(os.path.basename(input_path))[0]

        seq_coords, labels = self._get_sequences_from_bed_file(
            input_path,
            strand_index=strand_index,
            sample_continuous=sample_continuous,
            output_NAs_to_file="{0}.NA".format(os.path.join(output_dir, output_prefix)),
            reference_sequence=self.reference_sequence,
        )

        if output_format == "bigWig":
            constructor_args = []
            header = self._get_bigWig_header(seq_coords)
            reporter = WritePredictionsMultiCtBigWigHandler(
                self.features,
                [1, 2, 3],
                6,
                self._cell_types,
                self._get_bigWig_header(seq_coords),
                os.path.join(output_dir, output_prefix),
                self._write_mem_limit,
            )
        else:
            reporter = self._initialize_reporters(
                ["predictions"],
                os.path.join(output_dir, output_prefix),
                output_format,
                [
                    "index",
                    "chrom",
                    "start",
                    "end",
                    "strand",
                    "contains_unk",
                    "cell_type",
                ],
                output_size=len(labels),
            )[
                0
            ]  # ,
            # mode="prediction")[0]
        sequences = None
        batch_ids = []
        for i, (label, coords) in enumerate(zip(tqdm(labels), seq_coords)):
            (
                encoding,
                contains_unk,
            ) = self.reference_sequence.get_encoding_from_coords_check_unk(
                *coords, pad=True
            )
            if sequences is None:
                sequences = np.zeros((self.batch_size, *encoding.shape))
            if i and i % self.batch_size == 0:
                preds, cell_type_updated_batch_ids = self._get_predictions(
                    sequences, batch_ids
                )
                sequences = np.zeros((self.batch_size, *encoding.shape))
                # print ([i[2] for i in cell_type_updated_batch_ids])
                reporter.handle_batch_predictions(preds, cell_type_updated_batch_ids)
                batch_ids = []
            batch_ids.append(label + (contains_unk,))
            sequences[i % self.batch_size, :, :] = encoding
            if contains_unk:
                pass
                # warnings.warn(("For region {0}, "
                #               "reference sequence contains unknown "
                #               "base(s). --will be marked `True` in the "
                #               "`contains_unk` column of the .tsv or "
                #               "row_labels .txt file.").format(label))

        if (batch_ids and i == 0) or i % self.batch_size != 0:
            sequences = sequences[: i % self.batch_size + 1, :, :]
            preds, cell_type_updated_batch_ids = self._get_predictions(
                sequences, batch_ids
            )
            reporter.handle_batch_predictions(preds, cell_type_updated_batch_ids)

        reporter.write_to_file()
        reporter.close_handlers()

    def variant_effect_prediction(
        self,
        input_path,
        output_dir,
        fdr=0.1,
    ):
        """
        Get model score predictions for variants specified in a VCF file.
        Two sets of scores for sequences containing a reference allele and
        a variant allele. Final scores are averaged model outputs for
        `self.center_bin` sequences containing the specified allele in different locations
        of the sequence.
        Note: This procedure will ignore `self.batch_size` in favor of gathering
        `self.center_bin` sequences into a batch.

        Parameters
        ----------
        input_path : str
            Input path to the VCF file.
        output_dir : str
            Output directory to write the model predictions.
        fdr : float
            FDR for Benjamini-Hochberg p-value correction procedure.

        Returns
        -------
        None
            Writes the output to file(s) in `output_dir`. Filename will
            match that specified in the filepath.

        """
        output_dir = os.path.join(
            output_dir, os.path.splitext(os.path.basename(input_path))[0]
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(
            os.path.join(output_dir, "measured_tracks_mask.npy"),
            self._feature_indices_by_cell_type_index.T,
        )

        variants = pd.read_csv(input_path, sep="\t", header=None)
        variants = variants.drop(columns=[2, 5, 6])
        variants.columns = ["CHROM", "POS", "REF", "ALT", "NOTE"]
        variants[["AUSTATUS", "FIDSID"]] = variants["NOTE"].str.split(";", expand=True)
        variants["AUSTATUS"] = variants["AUSTATUS"].str.slice(start=9)
        variants["FIDSID"] = variants["FIDSID"].str.slice(start=7)
        variants[["FID", "SID"]] = variants["FIDSID"].str.split(".", expand=True)
        variants = variants.drop(columns=["FIDSID", "NOTE"])

        # set up for mean effect prediction calculations
        case_effect_sum = 0
        control_effect_sum = 0
        n_bad_cases = 0
        n_bad_controls = 0
        case_effects = []
        control_effects = []

        # set up for family-specific effect prediction difference distribution calculation
        family_specific_effect_diff = []

        fids = variants["FID"].unique()
        for fid in tqdm(fids):
            family_variants = variants[variants["FID"] == fid]

            # set up for matched proband - sibling distribution
            family_case_effect = 0
            family_control_effect = 0
            n_bad_family_cases = 0
            n_bad_family_controls = 0
            for _, variant in family_variants.iterrows():
                pos = int(variant["POS"]) - 1
                chrom = variant["CHROM"]
                ref = variant["REF"]
                alt = variant["ALT"]
                austatus = variant["AUSTATUS"]

                if ref not in self.reference_sequence.BASES_ARR:
                    raise ValueError(
                        f"Ref allele from VCF is not one of "
                        f"{self.reference_sequence.BASES_ARR}"
                    )

                # query a sequence of length 1200 to cut into batch samples later
                batch_start = pos - self.center_bin // 2 - self._start_radius + 1
                batch_end = pos + self.center_bin // 2 + self._end_radius
                batch_sequence = self.reference_sequence.get_encoding_from_coords(
                    chrom, batch_start, batch_end, strand="+", pad=False
                )
                variant_idx = pos - batch_start

                # compare retrieved sequence encoding with expected ref encoding
                seq_ref_letter = self.reference_sequence.INDEX_TO_BASE[
                    np.where(batch_sequence[variant_idx] == 1)[0][0]
                ]
                if seq_ref_letter != ref:
                    if austatus == "case":
                        n_bad_family_cases += 1
                        n_bad_cases += 1
                    else:
                        n_bad_family_controls += 1
                        n_bad_controls += 1
                    with open(os.path.join(output_dir, "wrong_refs.txt"), "a") as f:
                        for key in ["CHROM", "POS", "REF", "ALT", "FID", "SID"]:
                            f.write(f"{variant[key]}\t")
                        f.write("\n")
                    continue
                    raise ValueError(
                        f"Expected ref allele {ref} from VCF, retrieved "
                        f"{seq_ref_letter} from fasta in {chrom} position {pos}."
                    )

                # create batch
                batch = []
                for seq_start in range(self.center_bin):
                    seq_end = seq_start + self.sequence_length

                    sample = batch_sequence[seq_start:seq_end]
                    batch.append(sample)
                batch = np.stack(batch)

                # get scores for reference sequences
                preds = self._predict(batch)
                preds = preds.cpu().numpy()
                mean_preds = preds.mean(axis=0)

                # get scores for sequences with variant
                if ref != alt:
                    batch_alt_sequence = batch_sequence.copy()
                    alt_encoding = self.reference_sequence.sequence_to_encoding(alt)
                    batch_alt_sequence[variant_idx] = alt_encoding

                    # create batch
                    batch = []
                    for seq_start in range(self.center_bin):
                        seq_end = seq_start + self.sequence_length

                        sample = batch_alt_sequence[seq_start:seq_end]
                        batch.append(sample)
                    batch = np.stack(batch)

                    alt_preds = self._predict(batch)
                    alt_preds = alt_preds.cpu().numpy()
                    mean_alt_preds = alt_preds.mean(axis=0)
                else:
                    raise ValueError(
                        f"Ref allele ({ref}) cannot be the same as alt"
                        f" ({alt}) in {chrom} position {pos}."
                    )

                # variant score
                score = np.abs(mean_alt_preds - mean_preds)

                # update family stats
                if austatus == "case":
                    family_case_effect = family_case_effect + score
                    case_effects.append(score)
                else:
                    family_control_effect = family_control_effect + score
                    control_effects.append(score)
            family_counts = family_variants["AUSTATUS"].value_counts()
            if "case" in family_counts:
                n_family_cases = family_counts["case"] - n_bad_family_cases

                # update total case/control effects
                case_effect_sum = case_effect_sum + family_case_effect

                # compute family effect mean
                family_case_effect = family_case_effect / n_family_cases

                # write per-family score differences
                np.save(
                    os.path.join(output_dir, f"mean_abs_case_effect_{fid}.npy"),
                    family_case_effect,
                )
            if "control" in family_counts:
                n_family_controls = family_counts["control"] - n_bad_family_controls

                # update total case/control effects
                control_effect_sum = control_effect_sum + family_control_effect

                # compute family effect mean
                family_control_effect = family_control_effect / n_family_controls

                # write per-family score differences
                np.save(
                    os.path.join(output_dir, f"mean_abs_control_effect_{fid}.npy"),
                    family_control_effect,
                )

        variant_counts = variants["AUSTATUS"].value_counts()
        n_total_cases = variant_counts["case"] - n_bad_cases
        n_total_controls = variant_counts["control"] - n_bad_controls

        mean_case_effect = case_effect_sum / n_total_cases
        mean_control_effect = control_effect_sum / n_total_controls

        # write mean case and control effects
        np.save(
            os.path.join(output_dir, "total_mean_abs_case_effect.npy"), mean_case_effect
        )
        np.save(
            os.path.join(output_dir, "total_mean_abs_control_effect.npy"),
            mean_control_effect,
        )

        case_effects = np.stack(case_effects)
        control_effects = np.stack(control_effects)
        np.save(os.path.join(output_dir, "all_case_ref_outputs.npy"), case_effects)
        np.save(
            os.path.join(output_dir, "all_control_ref_outputs.npy"), control_effects
        )

        # run Mann-Whitney U-Test
        pvals = np.zeros(case_effects.shape[1:])
        for ct in range(pvals.shape[0]):
            for feat in range(pvals.shape[1]):
                _, pval = mannwhitneyu(
                    case_effects[:, ct, feat], control_effects[:, ct, feat]
                )
                pvals[ct, feat] = pval
        np.save(os.path.join(output_dir, "mwu_abs_pvals.npy"), pvals)

        # get corrected pvals via Benjamini-Hochberg procedure
        bh_pvals, fdr_pval_cutoff = bh(pvals.flatten(), fdr=fdr)
        bh_pvals = bh_pvals.reshape(pvals.shape)

        np.save(os.path.join(output_dir, "mwu_abs_bh_corrected_pvals.npy"), bh_pvals)
        np.save(
            os.path.join(output_dir, f"mwu_abs_bh_fdr{fdr}_cutoff.npy"), fdr_pval_cutoff
        )

    def get_predictions(
        self,
        input_path,
        output_dir=None,
        output_format="tsv",
        strand_index=None,
        sample_continuous=False,
    ):
        """
        Get model predictions for sequences specified as BED file.

        Parameters
        ----------
        input_path : str
            A BED file input.
        output_dir : str, optional
            Default is None. Output directory to write the model predictions.
            If this is left blank a raw sequence input will be assumed, though
            an output directory is required for FASTA and BED inputs.
        output_format : {'tsv', 'hdf5','bigWig'}, optional
            Default is 'tsv'. Choose whether to save TSV or HDF5 output files.
            TSV is easier to access (i.e. open with text editor/Excel) and
            quickly peruse, whereas HDF5 files must be accessed through
            specific packages/viewers that support this format (e.g. h5py
            Python package). Choose

                * 'tsv' if your list of sequences is relatively small
                  (:math:`10^4` or less in order of magnitude) and/or your
                  model has a small number of features (<1000).
                * 'hdf5' for anything larger and/or if you would like to
                  access the predictions/scores as a matrix that you can
                  easily filter, apply computations, or use in a subsequent
                  classifier/model. In this case, you may access the matrix
                  using `mat["data"]` after opening the HDF5 file using
                  `mat = h5py.File("<output.h5>", 'r')`. The matrix columns
                  are the features and will match the same ordering as your
                  features .txt file (same as the order your model outputs
                  its predictions) and the matrix rows are the sequences.
                  Note that the row labels (FASTA description/IDs) will be
                  output as a separate .txt file (should match the ordering
                  of the sequences in the input FASTA).

        strand_index : int or None, optional
            Default is None. If the trained model makes strand-specific
            predictions, your input BED file may include a column with strand
            information (strand must be one of {'+', '-', '.'}). Specify
            the index (0-based) to use it. Otherwise, by default '+' is used.
            (This parameter is ignored if FASTA file is used as input.)

        Returns
        -------
        None
            Writes the output to file(s) in `output_dir`. Filename will
            match that specified in the filepath. In addition, if any base
            in the given or retrieved sequence is unknown, the row labels .txt file
            or .tsv file will mark this sequence or region as `contains_unk = True`.

        """
        os.makedirs(output_dir, exist_ok=True)
        self.get_predictions_for_bed_file(
            input_path,
            output_dir,
            output_format=output_format,
            strand_index=strand_index,
            sample_continuous=sample_continuous,
        )
        return None

    def _pad_or_truncate_sequence(self, sequence):
        if len(sequence) < self.sequence_length:
            sequence = _pad_sequence(
                sequence,
                self.sequence_length,
                self.reference_sequence.UNK_BASE,
            )
        elif len(sequence) > self.sequence_length:
            sequence = _truncate_sequence(sequence, self.sequence_length)

        return sequence

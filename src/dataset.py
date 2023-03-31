import bisect

import numpy as np
import torch
from selene_sdk.sequences import Genome
from selene_sdk.targets import GenomicFeatures, qGenomicFeatures

from src.transforms import PermuteSequenceChannels
from torch.utils.data import RandomSampler
import gc
import random
from tqdm import trange

_FEATURE_NOT_PRESENT = -1
import h5py

class EncodeDataset(torch.utils.data.Dataset):
    """
    Dataset of ENCODE epigenetic features of either
    `(sequence, cell_type, feature_values, feature_mask)`
    or `(sequence, feature_cell_values)`

    Parameters
    ----------
    reference_class : selene.sequence.genome
        class representing the reference sequence
    reference_init_kwargs: dict
        any kwargs to pass to reference_class init function
    distinct_features : list(str)
        List of distinct `cell_type|feature_name|info` combinations available,
        e.g. `["K562|ZBTB33|None", "HCF|DNase|None", "HUVEC|DNase|None"]`.
    target_features : list(str)
        List of names of features we aim to predict, e.g. ["CTCF", "DNase"].
    target_class : selene.sdk.Target,
        selene.sdk target class      
    target_init_kwargs : dict,
        any kwargs to pass to target_class init function
    intervals : list(tuple)
        Intervals to sample from in the format `(chrom, start, end)`,
        e.g. [("chr1", 550, 590), ("chr2", 6100, 6315)].
    transform : callable, optional
        A callback function that takes `sequence, cell_type,
        feature_values, feature_mask` as arguments and returns
        their transformed version.
    sequence_length : int, optional
        Default is 1000. Dataset contains sequences of `sequence_length`
        where genomic features are annotated to the center regions of
        these sequences.
    center_bin_to_predict : int, optional
        Default is 200. Query the tabix-indexed targets file for a region of
        length `center_bin_to_predict`.
    feature_thresholds : float [0.0, 1.0] or None, optional
         Default is 0.5. The `feature_threshold` to pass to the
        `GenomicFeatures` object.
    strand : str
        Default is '+'. Strand to sample from.
    multi_ct_target : bool, optional
        Default is False. Make samples positional i.e.
        a sample would look like `(sequence, 0.0, target, target_mask)`,
        where `target` and `target_mask` have shape `(n_cell_types, n_target_features)`.
    position_skip : int, optional
        Default is 1. Use sequences centered at points that are `position_skip`
        positions apart to avoid samples with big sequence overlaps.
    masked_tracks_path : str or None, optional (default is None)
        path to file containing track names which should be masked
        in addition to the tracks which are not measured (i.e. available
        in distinct_features list). Target_mask will be set to False for 
        these tracks. If set to None no tracks will be masked except 
        unmeasured tracks 
    cash_file_path : str or None, optional (default is None)
        path to hdf5 cash file containig pre-computed inputs and targets

    Attributes
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        The reference sequence that examples are created from.
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features.
    target_class : selene.sdk.Target
        The `selene_sdk.targets.Target` class
    target_init_kwargs : dict
        any kwargs to pass to target_class init function
    target_features : list(str)
        List of names of features we aim to predict, e.g. ["CTCF", "DNase"].
    intervals : list(int)
        A list of intervals that we can draw samples from.
    intervals_length_sums : list(int)
        A list of the cumulative sums of lengths of the intervals
        that we can draw samples from. Used to convert dataset
        index into a position in a specific interval.
    sequence_length : int
        The length of the sequences to  train the model on.
    center_bin_to_predict: int
        Length of the center sequence piece in which to detect
        a feature annotation.
    surrounding_sequence_radius : int
        The length of sequence falling outside of the feature detection
        bin (i.e. `bin_radius`) center, but still within the
        `sequence_length`.
    strand : str
        Strand to sample from.
    multi_ct_target : bool
        Whether data is meant for multiple cell type model input or not.
    n_cell_types : int
        Total number of cell types present in the dataset.
    position_skip : int
        Number of sequence positions to skip between samples.
    masked_measured_tracks : numpy 2D array of int
        Indices of elements indicating id's of tracks
        which are measured (i.e. present in distinct_features file)
        but masked masked due to masked_tracks_path file
    unmasked_measured_tracks : numpy 2D array of int
        Indices of measured and unmasked tracks
    """

    def __init__(
        self,
        reference_class,
        reference_init_kwargs,
        distinct_features,
        target_features,
        intervals,
        samples_mode=False,
        transform=PermuteSequenceChannels(),
        sequence_length=1000,
        center_bin_to_predict=200,
        feature_thresholds=0.5,
        strand="+",
        multi_ct_target=False,
        position_skip=1,
        masked_tracks_path=None,
        target_class=GenomicFeatures,
        target_init_kwargs=None,
        cash_file_path=None
    ):
        self.reference_class = reference_class
        self.reference_init_kwargs = reference_init_kwargs
        self.reference_sequence = self._construct_ref_genome()

        self.distinct_features = distinct_features
        self.target_features = target_features
        self.feature_thresholds = feature_thresholds
        
        # we won't keep those tracks (i.e celltype-feature combintations)
        # where feature is not in target_features
        self.distinct_features = [
                i
                for i in self.distinct_features
                if self._parse_distinct_feature(i)[0] in self.target_features
            ]
        
        self.target_class = target_class
        target_init_kwargs["features"] = self.distinct_features
        self.target_init_kwargs = target_init_kwargs
        self.target = self._construct_target()

        self.multi_ct_target = multi_ct_target
        assert self.multi_ct_target , "From 2023, new releases accept multi_ct_target only"
        self.transform = transform

        self.sequence_length = sequence_length

        self.strand = strand

        self._cell_types = []
        cell_type_indices_by_feature_index = [
            [] for i in range(len(self.target_features))
        ]
        for distinct_feature_index, distinct_feature in enumerate(
            self.distinct_features
        ):
            feature_name, cell_type = self._parse_distinct_feature(distinct_feature)
            assert feature_name in self.target_features
            if cell_type not in self._cell_types:
                self._cell_types.append(cell_type)

        self.n_cell_types = len(self._cell_types)
        self.n_target_features = len(self.target_features)
        self._feature_indices_by_cell_type_index = np.full(
            (self.n_cell_types, self.n_target_features), _FEATURE_NOT_PRESENT
        )

        for distinct_feature_index, distinct_feature in enumerate(
            self.distinct_features
        ):
            feature_name, cell_type = self._parse_distinct_feature(distinct_feature)
            if feature_name not in self.target_features:
                continue
            feature_index = self.target_features.index(feature_name)
            cell_type_index = self._cell_types.index(cell_type)
            self._feature_indices_by_cell_type_index[cell_type_index][
                feature_index
            ] = distinct_feature_index

        if self.multi_ct_target:
            self.target_mask = (
                self._feature_indices_by_cell_type_index != _FEATURE_NOT_PRESENT
            )
            measures_tracks = np.array(self.target_mask)
            masked_tracks = []
            if masked_tracks_path is not None:
                with open(masked_tracks_path) as fin:
                    for line in fin:
                        masked_tracks.append(line.strip())

            for track in masked_tracks:
                feature_name, cell_type = self._parse_distinct_feature(track)
                feature_index = self.target_features.index(feature_name)
                cell_type_index = self._cell_types.index(cell_type)

                # sanity check: we assume we are masking here
                # only those tracks which are measured
                assert self.target_mask[cell_type_index][feature_index]

                self.target_mask[cell_type_index][feature_index] = False
            
            # now we save indices of masked and unmasked measured tracks
            # this will be used later in transform if we want to invert
            # mask for these tracks
            self.masked_measured_tracks = np.nonzero(
                    np.logical_and(measures_tracks,~self.target_mask)
                    )
            self.unmasked_measured_tracks = np.nonzero(
                    np.logical_and(measures_tracks,self.target_mask)
                    )
            
            # sanity check: we assume number of masked_measured_tracks
            # is equal to number of tracks which we asked to be masked
            assert len(self.masked_measured_tracks[0]) == len(masked_tracks)
            assert len(self.masked_measured_tracks[0]) + \
                    len(self.unmasked_measured_tracks[0]) == \
                    len(self.distinct_features)

        self.position_skip = position_skip

        self.samples_mode = samples_mode
        if self.samples_mode:
            self.samples = intervals
        else:
            self.center_bin_to_predict = center_bin_to_predict
            bin_radius = int(self.center_bin_to_predict / 2)
            self._start_radius = bin_radius
            self._end_radius = bin_radius + self.center_bin_to_predict % 2
            self._surrounding_sequence_radius = (
                        self.sequence_length - self.center_bin_to_predict
            ) // 2


            self.intervals = intervals
            self.intervals_length_sums = [0]
            for chrom, pos_start, pos_end in self.intervals:
                interval_length = (pos_end - pos_start) // self.position_skip + 1
                self.intervals_length_sums.append(
                    self.intervals_length_sums[-1] + interval_length
                )

        if self.multi_ct_target:
            self.target_size = self.target_mask.size
        else:
            self.target_size = self.n_target_features
        
        # update transforms: some of transforms may need to get 
        # dataset object 

        try: # transform might be an object with method set_masks
                self.transform.set_tracks_thresholds(self)
        except AttributeError:
            for tr in self.transform.transforms:
                try:
                    tr.set_tracks_thresholds(self)
                    print ("Info: set_tracks_thresholds set for transform ",str(tr))
                except AttributeError:
                    pass
        
        if cash_file_path != None:
            self.cash_file = h5py.open(cash_file_path,"r")
        else:
            self.cash_file = None
        
    def __len__(self):
        if self.cash_file is not None:
            return len(self.cash_file["cell_type"])
        
        if self.samples_mode:
            n_sequences = len(self.samples)
        else:
            n_sequences = self.intervals_length_sums[-1]
        if self.multi_ct_target:
            return n_sequences
        else:
            print ("Error: from 2023 only multi_ct_target supported!")
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.cash_file is not None:
            return self._retrieve_sample_from_cash(idx)
        if self.samples_mode:
            sample_idx, cell_type_idx = self._get_sample_cell_by_idx(idx)
            retrieved_sample = self._retrieve_sample_by_idx(sample_idx, cell_type_idx)
        else:
            chrom, pos, cell_type_idx = self._get_chrom_pos_cell_by_idx(idx)
            retrieved_sample = self._retrieve(chrom, pos, cell_type_idx)
            if retrieved_sample is None: # for some ids we fail to retrieve sequence
                                         # i.e. if interval center - radius_bin < 0
                                         # in this case we try to resample
                random.seed(idx)
                while retrieved_sample is None:
                    current_idx = random.randint(0,len(self)-1)
                    chrom, pos, cell_type_idx = self._get_chrom_pos_cell_by_idx(current_idx)
                    retrieved_sample = self._retrieve(chrom, pos, cell_type_idx)

        if self.transform is not None:
            retrieved_sample = self.transform(retrieved_sample)
            
        return retrieved_sample
    
    def _get_sample_cell_by_idx(self, idx):
        if not self.multi_ct_target:
            cell_type_idx = idx % self.n_cell_types
            sample_idx = idx // self.n_cell_types
        else:
            cell_type_idx = 0
            sample_idx = idx
        return sample_idx, cell_type_idx
    
    def _retrieve_sample_from_cash(self, sample_idx):
        cell_type = self.cash_file["cell_type"][idx]
        target = self.cash_file["target"][idx]
        g_seq = self.cah_file["sequence"]
        retrieved_seq = {}
        for key in g_seq.keys():
            retrieved_seq[key] = g_seq[key][index]

        return retrieved_seq, cell_type, target, self.target_mask


    def _retrieve_sample_by_idx(self, sample_idx, cell_type_idx):
        chrom, start, end, chrom_sample_idx = self.samples[sample_idx]
        context = self.sequence_length - (end - start)
        if context != 0:
            start -= context // 2
            end += context // 2 + context % 2
        track_vector = self.target.get_feature_data(chrom, chrom_sample_idx)
        target, target_mask, cell_type = self._track_vector_to_target(track_vector, cell_type_idx)

        retrieved_seq = self.reference_sequence.get_encoding_from_coords(
            chrom, start, end, self.strand
        )

        return retrieved_seq, cell_type, target, target_mask

    def _get_chrom_pos_cell_by_idx(self, idx):
        """
        Translates dataset index into genomic coordinates and
        cell type `(chrom, pos, cell_type_idx)`

        Parameters
        ----------
        idx : int
            Index of item in the dataset

        Returns
        -------
        chrom, pos, cell type:\
        tuple(str, int, int)
            Chromosome identifier, position in the chromosome, cell type
        """

        if not self.multi_ct_target:
            cell_type_idx = idx % self.n_cell_types
            position_idx = idx // self.n_cell_types
        else:
            cell_type_idx = 0
            position_idx = idx
        interval_idx = bisect.bisect(self.intervals_length_sums, position_idx) - 1
        interval_pos = (
            position_idx - self.intervals_length_sums[interval_idx]
        ) * self.position_skip + self.position_skip // 2

        # handle the edge case when interval_pos is out of interval boundaries
        interval_pos = min(
            interval_pos,
            self.intervals[interval_idx][2] - self.intervals[interval_idx][1],
        )

        chrom, pos_start, _ = self.intervals[interval_idx]
        return chrom, pos_start + interval_pos, cell_type_idx

    def _retrieve(self, chrom, position, cell_type_idx):
        """
        Retrieves a sample for a position for a given cell type
        from `reference_sequence`.

        Parameters
        ----------
        chrom : str
            The name of the region (e.g. "chrX", "YFP")
        position : int
            The position in the query region that we will search around
            for samples.
        cell_type_idx : int
            Cell type index

        Returns
        -------
        retrieved_seq, cell_type, target, target_mask :\
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
            Retrieved encoded sequence, one-hot encoded cell type,
            target features values and target feature mask.
            Target features values (`target`) is a vector of values of target
            features for a given position and cell type. If feature `i` doesn't
            exist for a given cell type, `target[i]` value is meaningless
            but not None. Target feature mask is a binary vector
            corresponding to whether or not a specific feature exists
            in the given dataset for a given cell type.

        """
        bin_start = position - self._start_radius
        bin_end = position + self._end_radius
        track_vector = self.target.get_feature_data(chrom, bin_start, bin_end)

        target, target_mask, cell_type = self._track_vector_to_target(track_vector, cell_type_idx)

        window_start = bin_start - self._surrounding_sequence_radius
        window_end = bin_end + self._surrounding_sequence_radius

        retrieved_seq = self.reference_sequence.get_encoding_from_coords(
            chrom, window_start, window_end, self.strand
        )

        if not self._check_retrieved_sequence(retrieved_seq, chrom, position):
            return None
        
        result = (retrieved_seq, cell_type, target, target_mask)
        return result

        # try: 
        #     retrieved_seq["cell_type"] = cell_type
        #     retrieved_seq["target"] = target
        #     retrieved_seq["target_mask"] = target_mask
        #     return retrieved_seq
        # except IndexError:
        #     return {
        #         "seq" : retrieved_seq,
        #         "cell_type" : cell_type,
        #         "target" : target,
        #         "target_mask" :  target_mask,
        #     }

    def _track_vector_to_target(self, track_vector, cell_type_idx=None):
        if self.multi_ct_target:
            target = []
            for cell_type_idx in range(self.n_cell_types):
                ct_target_idx = self._feature_indices_by_cell_type_index[
                    cell_type_idx
                ]
                ct_target = track_vector[..., ct_target_idx].astype(np.float32)
                target.append(ct_target)
            target = np.array(target).astype(np.float32)
            target_mask = self.target_mask
            cell_type = 0.0
        else:
            target_idx = self._feature_indices_by_cell_type_index[cell_type_idx]
            target = track_vector[..., target_idx].astype(np.float32)
            target_mask = target_idx != _FEATURE_NOT_PRESENT
            cell_type = np.zeros(self.n_cell_types, dtype=np.float32)
            cell_type[cell_type_idx] = 1
        if target.shape != target_mask.shape:
            target_mask = np.repeat(np.expand_dims(target_mask, axis=1), target.shape[1], axis=1)
        return target, target_mask, cell_type

    def _target_to_track_vector(self, target):
        if not self.multi_ct_target:
            raise ValueError('Impossible to recover a vector of tracks \
                    from a single cell type sample')
        track_vector = np.full(len(self.distinct_features), _FEATURE_NOT_PRESENT)
        for cell_type_idx in range(target.shape[0]):
            for feature_idx in range(target.shape[1]):
                track_vector_idx = self._feature_indices_by_cell_type_index[cell_type_idx, feature_idx]
                if track_vector_idx != _FEATURE_NOT_PRESENT:
                    track_vector[track_vector_idx] = target[cell_type_idx, feature_idx]
        return track_vector

    def _check_retrieved_sequence(self, sequence, chrom, position) -> bool:
        """Checks whether retrieved sequence is acceptable.

        Parameters
        ----------
            sequence : numpy.ndarray
                An array of shape [sequence_length, alphabet_size], defines a sequence.

        """
        try:
            sequence.shape # will work if sequence is 1-hot encoded but fails for tokens
        except:
            # TODO maybe implement some other checks for tokens
            return sequence is not None

        if sequence.shape[0] == 0:
            # logger.info(
            print(
                'Full sequence centered at region "{0}" position '
                "{1} could not be retrieved.".format(chrom, position)
            )
            return False
        elif np.sum(sequence) / float(sequence.shape[0]) < 0.60:
            # logger.info(
            print(
                "Over 30% of the bases in the sequence centered "
                "at region \"{0}\" position {1} are ambiguous ('N'). ".format(
                    chrom, position
                )
            )
            return False
        elif sequence.shape[0] != self.sequence_length:
            print(
                f"Sequence retrieved at {chrom} position {position}\
                length {sequence.shape[0]} does not match \
                specified sequence length {self.sequence_length}"
            )
            return False
        return True

    def _construct_ref_genome(self):
        return self.reference_class(**self.reference_init_kwargs)

    def _construct_target(self):
        return self.target_class(**self.target_init_kwargs)

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
    
    def export(self, fname, fmode="a"):
        """
        export current dataset as hdf5 file
        fname - name of the output file
        fmode - file mode for file open. If fmode == 'a' will append to current file
        hdf5key - name of the hdf5 file key where dataset should be saved
        """

        print (f"Export dataset {hdf5key} with length {len(self)} to file {fname}")
        retrieved_seq, cell_type, target, target_mask = self.__getitem__(0)
        
        try:
            seq_keys = retrieved_seq.keys()
            tokenized_data = True
        except AttributeError:
            tokenized_data = False

        with h5py.File(fname, fmode) as g:
            cell_type = g.create_dataset("cell_type", 
                                    shape=(len(self),), 
                                    )
            target = g.create_dataset("target",
                                      shape=[len(self)] + list(target.shape),
                                      dtype=target.dtype
                                      )
            # target_mask = g.create_dataset("target_mask",
            #                           shape=[len(self)] + list(target_mask.shape),
            #                           dtype=target_mask.dtype
            #                           )
            if tokenized_data:
                g_seq = g.create_group("sequence")
                for key,val in retrieved_seq.items():
                    g_seq.create_dataset(key,
                                      shape=[len(self)] + list(val.shape),
                                      dtype=val.dtype
                                    )
            else:
                print ("Export of non-tokenized dataset is not supported")
                raise NotImplementedError
            
            for index in trange(len(self)):
                retrieved_seq, cell_type, target, target_mask = self.__getitem__(index)
                g["cell_type"][index] = cell_type
                g["target"][index,:] = target
                if tokenized_data:
                    for key,val in retrieved_seq.items():
                        g_seq[key][index,:] = val
                else:
                    print ("Export of non-tokenized dataset is not supported")
                    raise NotImplementedError
                    
class LargeRandomSampler(torch.utils.data.RandomSampler):
    """
    Samples elements randomly by splitting the dataset into chunks and permuting
    indices within these chunks. If without replacement, then sample from a
    dataset shuffled in chunks.
    If with replacement, then user can specify `num_samples` to draw.

    Parameters
    ----------
    reference_sequence_path : str
        Path to reference sequence `fasta` file from which to create examples.
    data_source : torch.utils.data.Dataset
        Dataset to sample from.
    replacement : bool
        Samples are drawn on-demand with replacement if ``True``,
        default=``False``.
    num_samples : int
        Number of samples to draw, default=`len(dataset)`. This argument
        is supposed to be specified only when `replacement` is ``True``.
    generator : torch.Generator
        Generator used in sampling.
    chunk_size : int
        Size of chunks that dataset is divided into for shuffling.
    """

    def __init__(
        self,
        data_source,
        replacement=False,
        num_samples=None,
        generator=None,
        chunk_size=10000000,
    ):
        super().__init__(
            data_source,
            replacement=replacement,
            num_samples=num_samples,
            generator=generator,
        )

        self.chunk_size = chunk_size
        self.m_chunks = (len(self.data_source) - 1) // self.chunk_size + 1

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=generator
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()
        else:
            self.chunks_order = self._generate_chunks_order()
            for chunk_idx in self.chunks_order:
                self.cur_chunk = chunk_idx
                chunk_offset = self.chunk_size * self.cur_chunk
                chunk_perm = self._permute_chunk(self.cur_chunk)
                for idx in chunk_perm:
                    yield chunk_offset + idx.item()

    def _generate_chunks_order(self):
        return torch.randperm(self.m_chunks, generator=self.generator).tolist()

    def _permute_chunk(self, chunk_idx):
        n = len(self.data_source)
        if chunk_idx == self.m_chunks - 1:
            chunk_size = n % self.chunk_size
        else:
            chunk_size = self.chunk_size
        return torch.randperm(chunk_size, generator=self.generator)


class SubsetRandomSampler(torch.utils.data.SubsetRandomSampler):
    """
    Samples subset of dataset. Subset size could be defined as
    fraction of dataset size of exact number of samples

    Parameters
    ----------
    data_source : torch.utils.data.Dataset
        Dataset to sample from.
    num_samples : int or float
        Number of samples to draw, default=`len(dataset)`.
        when set to -1, will use all samples in the dataset
        when set to float 0<num_samples<1 will be interpreted as
                                a fraction of dataset to use
        when set to int 1<=num_samples<=len(dataset) will use
                             exactly num_samples samples
        when num_samples>=len(dataset) will reduce num_samples to len(dataset)
    generator : torch.Generator
        Generator used in sampling.
    """

    def __init__(self, data_source, num_samples=-1, generator=None):
        self.data_source = data_source
        if generator == None:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )

        if len(self.data_source) == 0:
            indices = []
        else:
            if num_samples == -1:
                indices = range(len(data_source))
            elif 0 < num_samples < 1:
                num_samples = max(1, len(data_source) * num_samples)
                indices = self._gen_random_index(num_samples, generator)
            elif 1 <= num_samples <= len(data_source):
                indices = self._gen_random_index(int(num_samples), generator)
            elif num_samples > len(data_source):
                indices = range(len(data_source))
            else:
                raise ValueError

        super(SubsetRandomSampler, self).__init__(indices, generator)

    def _gen_random_index(self, N, generator):
        return torch.randint(
            high=len(self.data_source),
            size=(N,),
            dtype=torch.int64,
            generator=generator,
        ).tolist()
    
def encode_worker_init_fn(worker_id):
    """Initialization function for multi-processing DataLoader worker"""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # reconstruct reference genome object
    # because it reads from fasta `pyfaidx`
    # which is not multiprocessing-safe, see:
    # https://github.com/mdshw5/pyfaidx/issues/167#issuecomment-667591513
    dataset.reference_sequence = dataset._construct_ref_genome()
    # and similarly for targets (as they use bigWig file handlers
    # for quantitative features, which are not multiprocessing-safe,
    # see https://github.com/deeptools/pyBigWig/issues/74#issuecomment-439520821 )
    dataset.target = dataset._construct_target()
    
    # some tests indicate that after re-initialization of the dataset unused data loader are not
    # cleared from memory. I hope this will fix this problem
    gc.collect()

# def collate_fn():

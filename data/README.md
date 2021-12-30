# Data

This folder contains files and scripts used for generation of ENCODE dataset and its test data files.

## Data generation

To generate target files for training configuration (to be used for dataset instantiation) use script [generate_target_files.py](generate_target_files.py). 
As an input you can specify paths to:
* reference genome `fasta` file,
* `.bed`-file with ENCODE-blacklisted regions, 
* `.bed`-file with data for all tracks
* file with a list of target features (e.g. DNase, CTCF) to be predicted (this can be a subset of all available features).

The script will generate the following files:
* `.bed`-file with all the intervals where at least one of the target features is non-zero (extended by `target_padding`). These intervals are used to sample sequences for training;
* `.bed`-file with data for target features only;
* `.txt`-file with a list of distinct tracks with target feature names present in the data.

These files can then be used to instantiate the dataset.

## Test data generation
To generate mini-data for tests located in [test_data/](test_data) [create_test_data.py](create_test_data.py) is used.
It creates mini `fasta` file and generates arbitrary targets for it using real target data file, but without preserving any logic.

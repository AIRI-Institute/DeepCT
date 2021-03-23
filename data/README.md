# Data

This folder contains files and scripts used for generation of ENCODE dataset and its test data files.

## Data generation

To generate target files for `EncodeDataset` instantiation use script [generate_target_files.py](generate_target_files.py). 
As an input you can specify paths to:
* the reference genome `fasta` file,
* `bed`-file with ENCODE-blacklisted regions, 
* `bed`-file with data for all features

You can also specify a list of target feature names you want in your dataset, e.g. `["DNase", "CTCF"]`

The script will generate the following files:
* `bed`-file with all the intervals where at least one of the target features is non-zero,
* `bed`-file with data for target features only,
* `txt`-file with a list of distinct features with target feature names present in the data

These files can then be used to instantiate `EncodeDataset`.

## Test data generation
To generate mini-data for tests located in [test_data/](test_data) [create_test_data.py](create_test_data.py) is used.
It creates mini `fasta` file and generates arbitrary targets for it using real target data file, but without preserving any logic.

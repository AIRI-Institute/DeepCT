# DeepCT

### Disclaimer:
This code is provided under [Apache License, Version 2.0](https://opensource.org/licenses/Apache-2.0)

## How to run:

1. Install requirements
```zsh
pip install -r requirements.txt
```
If you run into any problems with installation of `selene`, you can also consult `selene`'s [official instruction](https://github.com/FunctionLab/selene/blob/master/README.md#installing-selene-from-source)

2. Run scripts specified by config files, e.g.:
```zsh
python -m selene_sdk model_configs/config_file.yml
```

## Data 

There are several scripts to download and process the data.
* `data/download_boix_et_al_data.sh` to download raw `*.bigWig` files
* `data/process_boix_et_al_data.py` to preprocess the `*.bigWig` files
* `data/annotate_biox_data.py` to produce `selene`-style `.bed` file containing intervals of present track peaks.
* `data/generate_target_files.py` to produce dataset files to specify in configuration files to run training (see `data/README.md` for more info).

## Training the models

1. Make sure you have prepared your data.
2. Run training using your configuration file, for example `model_configs/train_example.yml`.

Config files for benchmarks mentioned in the paper (most configs assume necessary data files have already been generated and placed into a folder `dataset_data`):
* cell type specificity: `model_configs/boix_train_masked_ct.yml` (classification) and `boix_qDeepCT_train_masked_ct.yml` (regression). 

    *Note:* Before running this config you should also generate all necessary data files for tracks forming a maximal clique and place them into a folder `dataset_data/maximal_clique_with0` and additionally run `src/split_intervals.py` and `src/split_cell_types.npy` and place their results in the same folder;
* sequence specificity: `model_configs/boix_all_multi_ct_train.yml` (classification) and `model_configs/boix_allNonTreated_highqual_multi_ct_qDeepCT_train_2021-11-15-03-30-59.yml` (regression);
* masked tracks prediction: `model_configs/boix_allNonTreated_highqual_masked_multi_ct_qDeepCT_train.yml`.

## Inference

1. Prepare pre-trained model and input files.
2. Run inference using your configuration file, for example `model_configs/inference_example.yml`.

Config file for variant effect predictions mentioned in the paper: `model_configs/boix_all_multi_ct_predict.yml`

# DeepCT

_DeepCT_ can learn complex interconnections of epigenetic features 
and infer unmeasured data from any available input. Furthermore, 
it can learn cell type-specific properties, 
build biologically meaningful vector representations of cell types, 
and utilize these representations to generate cell type-specific predictions 
of the effects of non-coding variations in the human genome.

Our preprint: Sindeeva et al.
[Cell type-specific interpretation of noncoding variants using deep learning-based methods](https://doi.org/10.1101/2021.12.31.474623).

## Tutorial
You can now refer to our [Google Colab tutorial](https://colab.research.google.com/drive/1F4k-ee7MghWdOM-vX_4nusLpEq62pi18?usp=sharing) to play with our model. For inference on a list of variants please refer to our [inference helper repository]()https://github.com/AIRI-Institute/DeepCT-inference-helper.

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
* cell type specificity: `model_configs/boix_train_masked_ct.yml` (classification). 

    *Note:* Before running this config you should also generate all necessary data files for tracks forming a maximal clique and place them into a folder `dataset_data/maximal_clique_with0` and additionally run `src/split_intervals.py` and `src/split_cell_types.npy` and place their results in the same folder;
* sequence specificity: `model_configs/boix_all_multi_ct_train.yml` (classification) and `model_configs/boix_allNonTreated_highqual_multi_ct_qDeepCT_train_2021-11-15-03-30-59.yml` (regression);
* masked tracks prediction: `model_configs/boix_allNonTreated_highqual_masked_multi_ct_qDeepCT_train.yml`.

## Inference

1. Prepare pre-trained model and input files.
2. Run inference using your configuration file, for example `model_configs/inference_example.yml`.

Alternatively, you might want to use [our helper pipeline](https://github.com/AIRI-Institute/DeepCT-inference-helper). 
It comes with the trained model and scripts to yield simple outputs that can be used to annotate VCFs.

Config file for variant effect predictions mentioned in the paper: `model_configs/boix_all_multi_ct_predict.yml`

## Copyright

Provided under Apache License 2.0.

© 2022 Autonomous Non-Profit Organization
"Artificial Intelligence Research Institute" (AIRI).
All rights reserved.

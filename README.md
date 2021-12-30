# DeepCT

### Disclaimer:
This code is provided under [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt)

## How to run:

1. Clone and checkout upgraded `selene`:
```zsh
cd ~
git clone https://github.com/sberbank-ai-lab/selene.git
cd selene
git checkout arlapin/dev
```
2. Install `selene` as per [official instruction](https://github.com/FunctionLab/selene/blob/master/README.md#installing-selene-from-source)

3. Add `selene` to your `$PYTHONPATH`:
```zsh
$PYTHONPATH=$PYTHONPATH:/home/$USER/selene
```

4. Clone this repo

5. Run DeepCT according to config file:
```zsh
cd ~/DeepCT
python -u ~/selene/selene_sdk/cli.py model_configs/biox_all_multi_ct_train.yml
```


## How to evaluate:

### DeeperDeepSEA (trained on DNase-only)
```zsh
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u ~/selene/selene_sdk/cli.py model_configs/deeper_deep_sea_benchmark.yml
```

### DeepCT
1. Update [model_configs/biox_all_multi_ct_eval.yml](model_configs/biox_all_multi_ct_eval.yml) with model's path and config;
2. Run
```zsh
python -u ~/selene/selene_sdk/cli.py model_configs/biox_all_multi_ct_eval.yml
```

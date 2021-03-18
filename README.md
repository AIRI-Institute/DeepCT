# DeepCT

## How to run:

1. Clone and checkout upgraded `selene`:
```zsh
cd ~
git clone https://github.com/arlapin/selene.git
cd selene
git checkout arlapin/dev
```
2. Install `selene` as per [official instruction](https://github.com/FunctionLab/selene/blob/master/README.md#installing-selene-from-source)

3. Add `selene` to your `$PYTHONPATH`:
```zsh
$PYTHONPATH=$PYTHONPATH:/home/$USER/selene
```

4. Clone this repo

5. Run DeepCT on selected GPU:
```zsh
cd ~/DeepCT
CUDA_VISIBLE_DEVICES=0 python -u ~/selene/selene_sdk/cli.py model_configs/single_cell_type.yml
```

# DeepCT

## How to run:

1. Clone and checkout upgraded `selene`:
```zsh
cd ~
git clone https://github.com/arlapin/selene.git
cd selene
git checkout arlapin/wrap_to_tensor
```

2. Add `selene` to your `$PYTHONPATH`:
```zsh
$PYTHONPATH=$PYTHONPATH:/home/$USER/selene
```

3. Clone this repo

5. Run DeepCT:
```zsh
cd ~/DeepCT
python -u ~/selene/selene_sdk/cli.py model_configs/single_cell_type.yml --lr=0.08
```

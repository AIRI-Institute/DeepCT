import numpy as np

# maximum size of targets stored in-memory for validation metrics computation,
# value derived experimentally using data loader of size 2000, batch size 64,
# 194 cell types, and 201 target features
MAX_TOTAL_VAL_TARGET_SIZE = 2000 * 64 * 194 * 201


# TODO: move this to utils
def expand_dims(x):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    return x

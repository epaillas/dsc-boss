import numpy as np
from pathlib import Path

def get_box_randoms(nrandoms, seed=42):
    """
    Get random points inside the box.

    Parameters
    ----------
    nrandoms : int
        Number of random points to generate.
    seed : int, optional
        Random seed.
    Returns
    -------
    randoms : array_like
        Random points inside the box.
    """
    np.random.seed(seed)
    randoms = np.random.rand(nrandoms, 3) * boxsize
    return randoms

# nden = 0.00037723287321312076
nden = 0.00035
boxsize = 2000
nrandoms = 50 * int(nden * boxsize ** 3)

randoms = get_box_randoms(nrandoms)
print(len(randoms))


output_dir = Path(f'/global/homes/e/epaillas/carolscratch/ds_boss/HOD/bossprior/randoms/')
Path(output_dir).mkdir(parents=True, exist_ok=True)
output_fn = output_dir / 'randoms.txt'

np.savetxt(output_fn, randoms)

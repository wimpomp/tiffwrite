from itertools import product

import numpy as np
from tiffwrite import IJTiffFile


def test_single(tmp_path):
    path = tmp_path / 'test.tif'
    with IJTiffFile(path, (3, 4, 5)) as tif:
        for c, z, t in product(range(3), range(4), range(5)):
            tif.save(np.random.randint(0, 255, (64, 64)), c, z, t)
    assert path.exists()

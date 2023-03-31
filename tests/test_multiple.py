import numpy as np
from tiffwrite import IJTiffFile
from itertools import product
from contextlib import ExitStack
from tqdm.auto import tqdm


def test_mult(tmp_path):
    shape = (3, 5, 12)
    paths = [tmp_path / f'test{i}.tif' for i in range(8)]
    with ExitStack() as stack:
        tifs = [stack.enter_context(IJTiffFile(path, shape)) for path in paths]
        for c, z, t in tqdm(product(range(shape[0]), range(shape[1]), range(shape[2])), total=np.prod(shape)):
            for tif in tifs:
                tif.save(np.random.randint(0, 255, (1024, 1024)), c, z, t)
    assert all([path.exists() for path in paths])
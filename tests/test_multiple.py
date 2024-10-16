from contextlib import ExitStack
from itertools import product
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from tiffwrite import IJTiffFile


def test_mult(tmp_path: Path) -> None:
    shape = (2, 3, 5)
    paths = [tmp_path / f'test{i}.tif' for i in range(6)]
    with ExitStack() as stack:
        tifs = [stack.enter_context(IJTiffFile(path)) for path in paths]  # noqa
        for c, z, t in tqdm(product(range(shape[0]), range(shape[1]), range(shape[2])), total=np.prod(shape)):  # noqa
            for tif in tifs:
                tif.save(np.random.randint(0, 255, (64, 64)), c, z, t)
    assert all([path.exists() for path in paths])

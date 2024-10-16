from pathlib import Path

import numpy as np
import pytest
from tifffile import imread

from tiffwrite import IJTiffFile


@pytest.mark.parametrize('dtype', ('uint8', 'uint16', 'uint32', 'uint64',
                                   'int8', 'int16', 'int32', 'int64', 'float32', 'float64'))
def test_single(tmp_path: Path, dtype) -> None:
    with IJTiffFile(tmp_path / 'test.tif', dtype=dtype, pxsize=0.1, deltaz=0.5, timeinterval=6.5) as tif:
        a0, b0 = np.meshgrid(range(100), range(100))
        a0[::2, :] = 0
        b0[:, ::2] = 1
        tif.save(a0, 0, 0, 0)
        tif.save(b0, 1, 0, 0)

        a1, b1 = np.meshgrid(range(100), range(100))
        a1[:, ::2] = 0
        b1[::2, :] = 1
        tif.save(a1, 0, 0, 1)
        tif.save(b1, 1, 0, 1)

    t = imread(tmp_path / 'test.tif')
    assert t.dtype == np.dtype(dtype), "data type does not match"
    assert np.all(np.stack(((a0, b0), (a1, b1))) == t), "data does not match"

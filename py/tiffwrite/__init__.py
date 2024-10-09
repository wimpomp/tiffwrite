from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from tqdm.auto import tqdm

from . import tiffwrite_rs as rs  # noqa


__all__ = ['Header', 'IJTiffFile', 'IFD', 'FrameInfo', 'Tag', 'Strip', 'tiffwrite']


class Header:
    pass

class IFD(dict):
    pass


class Tag(rs.Tag):
    pass


Strip = tuple[list[int], list[int]]
CZT = tuple[int, int, int]
FrameInfo = tuple[IFD, Strip, CZT]


class IJTiffFile(rs.IJTiffFile):
    def __new__(cls, path: str | Path, shape: tuple[int, int, int], dtype: DTypeLike = 'uint16',
                colors: Sequence[str] = None, colormap: str = None, pxsize: float = None,
                deltaz: float = None, timeinterval: float = None, comment: str = None,
                **extratags:  Tag.Value | Tag) -> IJTiffFile:
        new = super().__new__(cls, str(path), shape)
        if colors is not None:
            new = new.with_colors(colors)
        if colormap is not None:
            new = new.with_colormap(colormap)
        if pxsize is not None:
            new = new.with_pxsize(pxsize)
        if deltaz is not None:
            new = new.with_deltaz(deltaz)
        if timeinterval is not None:
            new = new.with_timeinterval(timeinterval)
        if comment is not None:
            new = new.with_comment(comment)
        if extratags:
            new = new.extend_extratags(extratags)
        return new

    def __init__(self, path: str | Path, shape: tuple[int, int, int], dtype: DTypeLike = 'uint16',  # noqa
                 colors: Sequence[str] = None, colormap: str = None, pxsize: float = None,  # noqa
                 deltaz: float = None, timeinterval: float = None, comment: str = None,  # noqa
                 **extratags:  Tag.Value | Tag) -> None:  # noqa
        self.path = Path(path)
        self.dtype = np.dtype(dtype)

    def __enter__(self) -> IJTiffFile:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def save(self, frame: ArrayLike, c: int, z: int, t: int) -> None:
        frame = np.asarray(frame).astype(self.dtype)
        match self.dtype:
            case np.uint8:
                self.save_u8(frame, c, z, t)
            case np.uint16:
                self.save_u16(frame, c, z, t)
            case np.uint32:
                self.save_u32(frame, c, z, t)
            case np.uint64:
                self.save_u64(frame, c, z, t)
            case np.int8:
                self.save_i8(frame, c, z, t)
            case np.int16:
                self.save_i16(frame, c, z, t)
            case np.int32:
                self.save_i32(frame, c, z, t)
            case np.int64:
                self.save_i64(frame, c, z, t)
            case np.float32:
                self.save_f32(frame, c, z, t)
            case np.float64:
                self.save_f64(frame, c, z, t)
            case _:
                raise TypeError(f'Cannot save type {self.dtype}')


def tiffwrite(file: str | Path, data: np.ndarray, axes: str = 'TZCXY', dtype: DTypeLike = None, bar: bool = False,
              *args: Any, **kwargs: Any) -> None:
    """ file:       string; filename of the new tiff file
        data:       2 to 5D numpy array
        axes:       string; order of dimensions in data, default: TZCXY for 5D, ZCXY for 4D, CXY for 3D, XY for 2D data
        dtype:      string; datatype to use when saving to tiff
        bar:        bool; whether to show a progress bar
        other args: see IJTiffFile
    """

    axes = axes[-np.ndim(data):].upper()
    if not axes == 'CZTXY':
        axes_shuffle = [axes.find(i) for i in 'CZTXY']
        axes_add = [i for i, j in enumerate(axes_shuffle) if j < 0]
        axes_shuffle = [i for i in axes_shuffle if i >= 0]
        data = np.transpose(data, axes_shuffle)
        for axis in axes_add:
            data = np.expand_dims(data, axis)

    shape = data.shape[:3]
    with IJTiffFile(file, shape, data.dtype if dtype is None else dtype, *args, **kwargs) as f:  # type: ignore
        at_least_one = False
        for n in tqdm(product(*[range(i) for i in shape]), total=np.prod(shape),  # type: ignore
                      desc='Saving tiff', disable=not bar):
            if np.any(data[n]) or not at_least_one:
                f.save(data[n], *n)
                at_least_one = True

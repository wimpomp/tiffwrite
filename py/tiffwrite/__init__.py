from __future__ import annotations

from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any, Callable, Sequence
from warnings import warn

import colorcet
import matplotlib
import numpy as np
from matplotlib import colors as mpl_colors
from numpy.typing import ArrayLike, DTypeLike
from tqdm.auto import tqdm

from . import tiffwrite_rs as rs  # noqa

__all__ = ['IJTiffFile', 'IJTiffParallel', 'FrameInfo', 'Tag', 'tiffwrite']

try:
    __version__ = version(Path(__file__).parent.name)
except Exception:  # noqa
    __version__ = "unknown"

Tag = rs.Tag
FrameInfo = tuple[ArrayLike, int, int, int]


class Header:
    """ deprecated """


class IFD(dict):
    """ deprecated """


class TiffWriteWarning(UserWarning):
    pass


class IJTiffFile(rs.IJTiffFile):
    """ Writes a tiff file in a format that the BioFormats reader in Fiji understands.
        Zstd compression is done in parallel using Rust.
        path:           path to the new tiff file
        dtype:          datatype to use when saving to tiff
        colors:         a tuple with a color per channel, chosen from matplotlib.colors, html colors are also possible
        colormap:       name of a colormap from colorcet
        pxsize:         pixel size in um
        deltaz:         z slice interval in um
        timeinterval:   time between frames in seconds
        compression:    ('zstd', level) for zstd with compression level: -7 to 22, 'deflate' for deflate compresion
        comment:        comment to be saved in tif
        extratags:      other tags to be saved, example: (Tag.ascii(315, 'John Doe'), Tag.bytes(4567, [400, 500])
                            or (Tag.ascii(33432, 'Made by me'),).
    """
    def __new__(cls, path: str | Path, *args, **kwargs) -> IJTiffFile:
        return super().__new__(cls, str(path))

    def __init__(self, path: str | Path, *, dtype: DTypeLike = 'uint16',
                 colors: Sequence[str] = None, colormap: str = None, pxsize: float = None,
                 deltaz: float = None, timeinterval: float = None,
                 compression: int | str | tuple[int, int] | tuple[str, int] = None, comment: str = None,
                 extratags: Sequence[Tag] = None) -> None:

        def get_codec(idx: int | str):
            codecs = {'z': 50000, 'd': 8, 8: 8, 50000: 50000}
            if isinstance(idx, str):
                return codecs.get(idx[0].lower(), 50000)
            else:
                return codecs.get(int(idx), 50000)

        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        if compression is not None:
            if isinstance(compression, tuple):
                compression = get_codec(compression[0]), (int(compression[1]) if len(compression) > 1 else 22)
            else:
                compression = get_codec(compression), 22
            self.set_compression(*compression)
        if colors is not None:
            self.colors = np.array([get_color(color) for color in colors])
        if colormap is not None:
            self.colormap = get_colormap(colormap)
        if pxsize is not None:
            self.px_size = float(pxsize)
        if deltaz is not None:
            self.delta_z = float(deltaz)
        if timeinterval is not None:
            self.time_interval = float(timeinterval)
        if comment is not None:
            self.comment = comment
        if extratags is not None:
            for extra_tag in extratags:
                self.append_extra_tag(extra_tag, None)
        if self.dtype.itemsize == 1 and colors is not None:
            warn('Fiji will not interpret colors saved in an (u)int8 tif, save as (u)int16 instead.',
                 TiffWriteWarning, stacklevel=2)
        if colors is not None and colormap is not None:
            warn('Cannot have colors and colormap simultaneously.', TiffWriteWarning, stacklevel=2)

    def __enter__(self) -> IJTiffFile:
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.close()

    def save(self, frame: ArrayLike, c: int, z: int, t: int, extratags: Sequence[Tag] = None) -> None:
        """ save a 2d numpy array to the tiff at channel=c, slice=z, time=t, with optional extra tif tags """
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
        if extratags is not None:
            for extra_tag in extratags:
                self.append_extra_tag(extra_tag, (c, z, t))


def get_colormap(colormap: str) -> np.ndarray:
    if hasattr(colorcet, colormap.rstrip('_r')):
        cm = np.array([[int(''.join(i), 16) for i in zip(*[iter(s[1:])] * 2)]
                        for s in getattr(colorcet, colormap.rstrip('_r'))]).astype('uint8')
        if colormap.endswith('_r'):
            cm = cm[::-1]
        if colormap.startswith('glasbey') or colormap.endswith('glasbey'):
            cm[0] = 255, 255, 255
            cm[-1] = 0, 0, 0
    else:
        cmap = matplotlib.colormaps.get_cmap(colormap)
        if cmap.N < 256:
            cm = (255 * np.vstack(((1, 1, 1),
                                   matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(1, 254),
                                                                cmap).to_rgba(np.arange(1, 254))[:, :3],
                                   (0, 0, 0)))).astype('uint8')
        else:
            cm = (255 * matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(0, 255), cmap)
                  .to_rgba(np.arange(256))[:, :3]).astype('uint8')
    return cm


def get_color(color: str) -> np.ndarray:
    return np.array([int(''.join(i), 16) for i in zip(*[iter(mpl_colors.to_hex(color)[1:])] * 2)]).astype('uint8')


def tiffwrite(file: str | Path, data: np.ndarray, axes: str = 'TZCYX', dtype: DTypeLike = None, bar: bool = False,
              *args: Any, **kwargs: Any) -> None:
    """ file:       string; filename of the new tiff file
        data:       2 to 5D numpy array
        axes:       string; order of dimensions in data, default: TZCYX for 5D, ZCYX for 4D, CYX for 3D, YX for 2D data
        dtype:      string; datatype to use when saving to tiff
        bar:        bool; whether to show a progress bar
        other args: see IJTiffFile
    """

    axes = axes[-np.ndim(data):].upper()
    if not axes == 'CZTYX':
        axes_shuffle = [axes.find(i) for i in 'CZTYX']
        axes_add = [i for i, j in enumerate(axes_shuffle) if j < 0]
        axes_shuffle = [i for i in axes_shuffle if i >= 0]
        data = np.transpose(data, axes_shuffle)
        for axis in axes_add:
            data = np.expand_dims(data, axis)

    shape = data.shape[:3]
    with IJTiffFile(file, dtype=data.dtype if dtype is None else dtype, *args, **kwargs) as f:
        for n in tqdm(product(*[range(i) for i in shape]), total=np.prod(shape),  # type: ignore
                      desc='Saving tiff', disable=not bar):
            f.save(data[n], *n)


try:
    from abc import ABCMeta, abstractmethod
    from functools import wraps

    from parfor import ParPool, Task


    class Pool(ParPool):
        def __init__(self, ijtifffile: IJTiffFile, parallel: Callable[[Any], Sequence[FrameInfo]]):
            self.ijtifffile = ijtifffile
            super().__init__(parallel)  # noqa

        def done(self, task: Task) -> None:
            c, z, t = task.handle
            super().done(task)
            for frame, cn, zn, tn in self[c, z, t]:
                self.ijtifffile.save(frame, c + cn, z + zn, t + tn)

        def close(self) -> None:
            while len(self.tasks):
                self.get_newest()
            super().close()
            self.ijtifffile.close()


    class IJTiffParallel(metaclass=ABCMeta):
        """ wraps IJTiffFile.save in a parallel pool, the method 'parallel' needs to be overloaded """

        @abstractmethod
        def parallel(self, frame: Any) -> Sequence[FrameInfo]:
            """ does something with frame in a parallel process,
                and returns a sequence of frames and offsets to c, z and t to save in the tif """

        @wraps(IJTiffFile.__init__)
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.ijtifffile = IJTiffFile(*args, **kwargs)
            self.pool = Pool(self.ijtifffile, self.parallel)

        def __enter__(self) -> IJTiffParallel:
            return self

        def __exit__(self, *args: Any, **kwargs: Any) -> None:
            self.close()

        @wraps(IJTiffFile.save)
        def save(self, frame: Any, c: int, z: int, t: int, extratags: Sequence[Tag] = None) -> None:
            self.pool[c, z, t] = frame
            if extratags is not None:
                for extra_tag in extratags:
                    self.ijtifffile.append_extra_tag(extra_tag, (c, z, t))

        def close(self) -> None:
            self.pool.close()

except ImportError:
    IJTiffParallel = None

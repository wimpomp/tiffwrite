from __future__ import annotations

import struct
import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from fractions import Fraction
from functools import cached_property
from hashlib import sha1
from importlib.metadata import version
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import Any, BinaryIO, Callable, Generator, Literal, Optional, Sequence

import colorcet
import numpy as np
import tifffile
from matplotlib import colors as mpl_colors
from numpy.typing import DTypeLike
from parfor import ParPool, PoolSingleton
from tqdm.auto import tqdm

__all__ = ["IJTiffFile", "Tag", "tiffwrite"]


try:
    __version__ = version("tiffwrite")
except Exception:  # noqa
    __version__ = "unknown"


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
        for n in tqdm(product(*[range(i) for i in shape]), total=np.prod(shape), desc='Saving tiff', disable=not bar):
            if np.any(data[n]) or not at_least_one:
                f.save(data[n], *n)
                at_least_one = True


class Header:
    def __init__(self, filehandle_or_byteorder: BinaryIO | Literal['>', '<'] | None = None,
                 bigtiff: bool = True) -> None:
        if filehandle_or_byteorder is None or isinstance(filehandle_or_byteorder, str):
            self.byteorder = filehandle_or_byteorder or '<'
            self.bigtiff = bigtiff
            if self.bigtiff:
                self.tagsize = 20
                self.tagnoformat = 'Q'
                self.offsetsize = 8
                self.offsetformat = 'Q'
                self.offset = 16
            else:
                self.tagsize = 12
                self.tagnoformat = 'H'
                self.offsetsize = 4
                self.offsetformat = 'I'
                self.offset = 8
        else:
            fh = filehandle_or_byteorder
            fh.seek(0)
            self.byteorder = '>' if fh.read(2) == b'MM' else '<'
            self.bigtiff = {42: False, 43: True}[struct.unpack(self.byteorder + 'H', fh.read(2))[0]]
            if self.bigtiff:
                self.tagsize = 20
                self.tagnoformat = 'Q'
                self.offsetsize = struct.unpack(self.byteorder + 'H', fh.read(2))[0]
                self.offsetformat = {8: 'Q', 16: '2Q'}[self.offsetsize]
                assert struct.unpack(self.byteorder + 'H', fh.read(2))[0] == 0, 'Not a TIFF-file'
                self.offset = struct.unpack(self.byteorder + self.offsetformat, fh.read(self.offsetsize))[0]
            else:
                self.tagsize = 12
                self.tagnoformat = 'H'
                self.offsetformat = 'I'
                self.offsetsize = 4
                self.offset = struct.unpack(self.byteorder + self.offsetformat, fh.read(self.offsetsize))[0]

    def write(self, fh: BinaryIO) -> None:
        fh.write({'<': b'II', '>': b'MM'}[self.byteorder])
        if self.bigtiff:
            fh.write(struct.pack(self.byteorder + 'H', 43))
            fh.write(struct.pack(self.byteorder + 'H', 8))
            fh.write(struct.pack(self.byteorder + 'H', 0))
            fh.write(struct.pack(self.byteorder + 'Q', self.offset))
        else:
            fh.write(struct.pack(self.byteorder + 'H', 42))
            fh.write(struct.pack(self.byteorder + 'I', self.offset))


class Tag:
    Value = bytes | str | float | Fraction | Sequence[bytes | str | float | Fraction]
    tiff_tag_registry = tifffile.TiffTagRegistry({key: value.lower() for key, value in tifffile.TIFF.TAGS.items()})

    @staticmethod
    def from_dict(tags: dict[str | int, Value | Tag]) -> dict[int, Tag]:
        return {(key if isinstance(key, int)
                 else (int(key[3:]) if key.lower().startswith('tag')
                       else Tag.tiff_tag_registry[key.lower()])): tag if isinstance(tag, Tag) else Tag(tag)
                for key, tag in tags.items()}

    @staticmethod
    def fraction(numerator: float = 0, denominator: int = None) -> Fraction:
        return Fraction(numerator, denominator).limit_denominator(  # type: ignore
            2 ** (31 if numerator < 0 or (denominator is not None and denominator < 0) else 32) - 1)

    def __init__(self, ttype_or_value: str | Value, value: Value = None,
                 offset: int = None) -> None:
        self._value: bytes | str | Sequence[bytes | str | float | Fraction]
        self.fh: Optional[BinaryIO] = None
        self.header: Optional[Header] = None
        self.bytes_data: Optional[bytes] = None
        if value is None:
            self.value = ttype_or_value  # type: ignore
            if isinstance(self.value, (str, bytes)) or all([isinstance(value, (str, bytes)) for value in self.value]):
                ttype = 'ascii'
            elif all([isinstance(value, int) for value in self.value]):
                min_value: int = np.min(self.value)  # type: ignore
                max_value: int = np.max(self.value)  # type: ignore
                type_map = {'uint8': 'byte', 'int8': 'sbyte', 'uint16': 'short', 'int16': 'sshort',
                            'uint32': 'long', 'int32': 'slong', 'uint64': 'long8', 'int64': 'slong8'}
                for dtype, ttype in type_map.items():
                    if np.iinfo(dtype).min <= min_value and max_value <= np.iinfo(dtype).max:
                        break
                    else:
                        ttype = 'undefined'
            elif all([isinstance(value, Fraction) for value in self.value]):
                if all([value.numerator < 0 or value.denominator < 0 for value in self.value]):  # type: ignore
                    ttype = 'srational'
                else:
                    ttype = 'rational'
            elif all([isinstance(value, (float, int)) for value in self.value]):
                min_value = np.min(np.asarray(self.value)[np.isfinite(self.value)])  # type: ignore
                max_value = np.max(np.asarray(self.value)[np.isfinite(self.value)])  # type: ignore
                type_map = {'float32': 'float', 'float64': 'double'}
                for dtype, ttype in type_map.items():
                    if np.finfo(dtype).min <= min_value and max_value <= np.finfo(dtype).max:
                        break
                    else:
                        ttype = 'undefined'
            elif all([isinstance(value, complex) for value in self.value]):
                ttype = 'complex'
            else:
                ttype = 'undefined'
            self.ttype = tifffile.TIFF.DATATYPES[ttype.upper()]  # noqa
        else:
            self.value = value  # type: ignore
            self.ttype = tifffile.TIFF.DATATYPES[ttype_or_value.upper()] if isinstance(ttype_or_value, str) \
                else ttype_or_value  # type: ignore
        self.dtype = tifffile.TIFF.DATA_FORMATS[self.ttype]
        self.offset = offset
        self.type_check()

    @property
    def value(self) -> bytes | str | Sequence[bytes | str | float | Fraction]:
        return self._value

    @value.setter
    def value(self, value: Value) -> None:
        self._value = value if isinstance(value, Iterable) else (value,)

    def __repr__(self) -> str:
        if self.offset is None:
            return f'{tifffile.TIFF.DATATYPES(self.ttype).name}: {self.value!r}'
        else:
            return f'{tifffile.TIFF.DATATYPES(self.ttype).name} @ {self.offset}: {self.value!r}'

    def type_check(self) -> None:
        try:
            self.bytes_and_count(Header())
        except Exception:
            raise ValueError(f"tif tag type '{tifffile.TIFF.DATATYPES(self.ttype).name}' and "
                             f"data type '{type(self.value[0]).__name__}' do not correspond")

    def bytes_and_count(self, header: Header) -> tuple[bytes, int]:
        if isinstance(self.value, bytes):
            return self.value, len(self.value) // struct.calcsize(self.dtype)
        elif self.ttype in (2, 14):
            if isinstance(self.value, str):
                bytes_value = self.value.encode('ascii') + b'\x00'
            else:
                bytes_value = b'\x00'.join([value.encode('ascii') for value in self.value]) + b'\x00'  # type: ignore
            return bytes_value, len(bytes_value)
        elif self.ttype in (5, 10):
            return b''.join([struct.pack(header.byteorder + self.dtype,  # type: ignore
                                         *((value.denominator, value.numerator) if isinstance(value, Fraction)
                                           else value)) for value in self.value]), len(self.value)
        else:
            return b''.join([struct.pack(header.byteorder + self.dtype, value) for value in self.value]), \
                   len(self.value)

    def write_tag(self, fh: BinaryIO, key: int, header: Header, offset: int = None) -> None:
        self.fh = fh
        self.header = header
        if offset is None:
            self.offset = fh.tell()
        else:
            fh.seek(offset)
            self.offset = offset
        fh.write(struct.pack(header.byteorder + 'HH', key, self.ttype))
        bytes_tag, count = self.bytes_and_count(header)
        fh.write(struct.pack(header.byteorder + header.offsetformat, count))
        len_bytes = len(bytes_tag)
        if len_bytes <= header.offsetsize:
            fh.write(bytes_tag)
            self.bytes_data = None
            empty_bytes = header.offsetsize - len_bytes
        else:
            self.bytes_data = bytes_tag
            empty_bytes = header.offsetsize
        if empty_bytes:
            fh.write(empty_bytes * b'\x00')

    def write_data(self, write: Callable[[BinaryIO, bytes], None] = None) -> None:
        if self.bytes_data and self.fh is not None and self.header is not None and self.offset is not None:
            self.fh.seek(0, 2)
            if write is None:
                offset = self.write(self.bytes_data)
            else:
                offset = write(self.fh, self.bytes_data)
            self.fh.seek(self.offset + self.header.tagsize - self.header.offsetsize)
            self.fh.write(struct.pack(self.header.byteorder + self.header.offsetformat, offset))

    def write(self, bytes_value: bytes) -> Optional[int]:
        if self.fh is not None:
            if self.fh.tell() % 2:
                self.fh.write(b'\x00')
            offset = self.fh.tell()
            self.fh.write(bytes_value)
            return offset

    def copy(self) -> Tag:
        return self.__class__(self.ttype, self.value[:], self.offset)


class IFD(dict):
    def __init__(self, fh: BinaryIO = None) -> None:
        super().__init__()
        self.fh = fh
        self.header: Optional[Header] = None
        self.offset: Optional[int] = None
        self.where_to_write_next_ifd_offset: Optional[int] = None
        if fh is not None:
            header = Header(fh)
            fh.seek(header.offset)
            n_tags = struct.unpack(header.byteorder + header.tagnoformat,
                                   fh.read(struct.calcsize(header.tagnoformat)))[0]
            assert n_tags < 4096, 'Too many tags'
            addr = []
            addroffset = []

            length = 8 if header.bigtiff else 2
            length += n_tags * header.tagsize + header.offsetsize

            for i in range(n_tags):
                pos = header.offset + struct.calcsize(header.tagnoformat) + header.tagsize * i
                fh.seek(pos)

                code, ttype = struct.unpack(header.byteorder + 'HH', fh.read(4))
                count = struct.unpack(header.byteorder + header.offsetformat, fh.read(header.offsetsize))[0]

                dtype = tifffile.TIFF.DATA_FORMATS[ttype]
                dtypelen = struct.calcsize(dtype)

                toolong = struct.calcsize(dtype) * count > header.offsetsize
                if toolong:
                    addr.append(fh.tell() - header.offset)
                    caddr = struct.unpack(header.byteorder + header.offsetformat, fh.read(header.offsetsize))[0]
                    addroffset.append(caddr - header.offset)
                    cp = fh.tell()
                    fh.seek(caddr)

                if ttype == 1:
                    value: Tag.Value = fh.read(count)
                elif ttype == 2:
                    value = fh.read(count).decode('ascii').rstrip('\x00')
                elif ttype in (5, 10):
                    value = [struct.unpack(header.byteorder + dtype, fh.read(dtypelen))  # type: ignore
                             for _ in range(count)]
                else:
                    value = [struct.unpack(header.byteorder + dtype, fh.read(dtypelen))[0] for _ in range(count)]

                if toolong:
                    fh.seek(cp)  # noqa

                self[code] = Tag(ttype, value, pos)
            fh.seek(header.offset)

    def __setitem__(self, key: str | int, tag: str | float | Fraction | Tag) -> None:
        super().__setitem__(Tag.tiff_tag_registry[key.lower()] if isinstance(key, str) else key,
                            tag if isinstance(tag, Tag) else Tag(tag))

    def items(self) -> Generator[tuple[int, Tag], None, None]:  # type: ignore[override]
        return ((key, self[key]) for key in sorted(self))

    def keys(self) -> Generator[int, None, None]:  # type: ignore[override]
        return (key for key in sorted(self))

    def values(self) -> Generator[Tag, None, None]:  # type: ignore[override]
        return (self[key] for key in sorted(self))

    def write(self, fh: BinaryIO, header: Header, write: Callable[[BinaryIO, bytes], None] = None) -> BinaryIO:
        self.fh = fh
        self.header = header
        if fh.seek(0, 2) % 2:
            fh.write(b'\x00')
        self.offset = fh.tell()
        fh.write(struct.pack(header.byteorder + header.tagnoformat, len(self)))
        for key, tag in self.items():
            tag.write_tag(fh, key, header)
        self.where_to_write_next_ifd_offset = fh.tell()
        fh.write(b'\x00' * header.offsetsize)
        for tag in self.values():
            tag.write_data(write)
        return fh

    def write_offset(self, where_to_write_offset: int) -> None:
        if self.fh is not None and self.header is not None:
            self.fh.seek(where_to_write_offset)
            self.fh.write(struct.pack(self.header.byteorder + self.header.offsetformat, self.offset))

    def copy(self) -> IFD:
        new = self.__class__()
        new.update({key: tag.copy() for key, tag in self.items()})
        return new


class IJTiffFile:
    """ Writes a tiff file in a format that the BioFormats reader in Fiji understands.
        file:           filename of the new tiff file
        shape:          shape (CZT) of the data to be written
        dtype:          datatype to use when saving to tiff
        colors:         a tuple with a color per channel, chosen from matplotlib.colors, html colors are also possible
        colormap:       name of a colormap from colorcet
        pxsize:         pixel size in um
        deltaz:         z slice interval in um
        timeinterval:   time between frames in seconds
        extratags:      other tags to be saved, example: Artist='John Doe', Tag4567=[400, 500]
                            or Copyright=Tag('ascii', 'Made by me'). See tiff_tag_registry.items().
        wp@tl20200214
    """
    def __init__(self, path: str | Path, shape: tuple[int, int, int], dtype: DTypeLike = 'uint16',
                 colors: Sequence[str] = None, colormap: str = None, pxsize: float = None,
                 deltaz: float = None, timeinterval: float = None,
                 compression: tuple[int, int] = (50000, 22), comment: str = None,
                 **extratags:  Tag.Value | Tag) -> None:
        assert len(shape) >= 3, 'please specify all c, z, t for the shape'
        assert len(shape) <= 3, 'please specify only c, z, t for the shape'
        assert np.dtype(dtype).char in 'BbHhf', 'datatype not supported'
        assert colors is None or colormap is None, 'cannot have colors and colormap simultaneously'

        self.path = Path(path)
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.colors = colors
        self.colormap = colormap
        self.pxsize = pxsize
        self.deltaz = deltaz
        self.timeinterval = timeinterval
        self.compression = compression
        self.comment = comment
        self.extratags = {} if extratags is None else Tag.from_dict(extratags)  # type: ignore
        if pxsize is not None:
            pxsize_fraction = Tag.fraction(pxsize)
            self.extratags.update({282: Tag(pxsize_fraction), 283: Tag(pxsize_fraction)})

        self.header = Header()
        self.spp = self.shape[0] if self.colormap is None and self.colors is None else 1  # samples/pixel
        self.nframes = np.prod(self.shape[1:]) if self.colormap is None and self.colors is None else np.prod(self.shape)
        self.frame_extra_tags: dict[tuple[int, int, int], dict[int, Tag]] = {}
        self.fh = FileHandle(self.path)
        self.hashes = PoolSingleton().manager.dict()
        self.pool = ParPool(self.compress_frame)
        self.main_process = True

        with self.fh.lock() as fh:  # noqa
            self.header.write(fh)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.main_process = False

    def __hash__(self) -> int:
        return hash(self.path)

    def get_frame_number(self, n: tuple[int, int, int]) -> tuple[int, int]:
        if self.colormap is None and self.colors is None:
            return n[1] + n[2] * self.shape[1], n[0]
        else:
            return n[0] + n[1] * self.shape[0] + n[2] * self.shape[0] * self.shape[1], 0

    def ij_tiff_frame(self, frame: np.ndarray) -> bytes:
        with BytesIO() as frame_bytes:
            with tifffile.TiffWriter(frame_bytes, bigtiff=self.header.bigtiff,
                                     byteorder=self.header.byteorder) as t:  # type: ignore
                # predictor=True might save a few bytes, but requires the package imagecodes to save floats
                t.write(frame, compression=self.compression, contiguous=True, predictor=False)  # type: ignore
            return frame_bytes.getvalue()

    def save(self, frame: np.ndarray | Any, c: int, z: int, t: int,
             **extratags: Tag.Value | Tag) -> None:
        """ save a 2d numpy array to the tiff at channel=c, slice=z, time=t, with optional extra tif tags
        """
        assert (c, z, t) not in self.pool.tasks, f'frame {c} {z} {t} is added already'
        assert all([0 <= i < s for i, s in zip((c, z, t), self.shape)]), \
            'frame {} {} {} is outside shape {} {} {}'.format(c, z, t, *self.shape)
        self.pool(frame.astype(self.dtype) if hasattr(frame, 'astype') else frame, handle=(c, z, t))
        if extratags:
            self.frame_extra_tags[(c, z, t)] = Tag.from_dict(extratags)  # type: ignore

    @property
    def description(self) -> bytes:
        desc = ['ImageJ=1.11a']
        if self.colormap is None and self.colors is None:
            desc.extend((f'images={np.prod(self.shape[:1])}', f'slices={self.shape[1]}', f'frames={self.shape[2]}'))
        else:
            desc.extend((f'images={np.prod(self.shape)}', f'channels={self.shape[0]}', f'slices={self.shape[1]}',
                         f'frames={self.shape[2]}'))
        if self.shape[0] == 1:
            desc.append('mode=grayscale')
        else:
            desc.append('mode=composite')
        desc.extend(('hyperstack=true', 'loop=false', 'unit=micron'))
        if self.deltaz is not None:
            desc.append(f'spacing={self.deltaz}')
        if self.timeinterval is not None:
            desc.append(f'interval={self.timeinterval}')
        desc_bytes = [bytes(d, 'ascii') for d in desc]
        if self.comment is not None:
            desc_bytes.append(b'')
            if isinstance(self.comment, bytes):
                desc_bytes.append(self.comment)
            else:
                desc_bytes.append(bytes(self.comment, 'ascii'))
        return b'\n'.join(desc_bytes)

    @cached_property
    def colormap_bytes(self) -> Optional[bytes]:
        if self.colormap:
            colormap = getattr(colorcet, self.colormap)
            colormap[0] = '#ffffff'
            colormap[-1] = '#000000'
            colormap = 65535 * np.array(
                [[int(''.join(i), 16) for i in zip(*[iter(s[1:])] * 2)] for s in colormap]) // 255
            if np.dtype(self.dtype).itemsize == 2:
                colormap = np.tile(colormap, 256).reshape((-1, 3))
            return b''.join([struct.pack(self.header.byteorder + 'H', c) for c in colormap.T.flatten()])

    @cached_property
    def colors_bytes(self) -> list[bytes]:
        return [b''.join([struct.pack(self.header.byteorder + 'H', c)
                          for c in np.linspace(0, 65535 * np.array(mpl_colors.to_rgb(color)),
                                               65536 if np.dtype(self.dtype).itemsize == 2 else 256,
                                               dtype=int).T.flatten()]) for color in self.colors] if self.colors else []

    def close(self) -> None:
        if self.main_process:
            ifds, strips = {}, {}
            for n in list(self.pool.tasks):
                framenr, channel = self.get_frame_number(n)
                ifds[framenr], strips[(framenr, channel)] = self.pool[n]

            self.pool.close()
            with self.fh.lock() as fh:  # noqa
                for n, tags in self.frame_extra_tags.items():
                    framenr, channel = self.get_frame_number(n)
                    ifds[framenr].update(tags)
                if self.colormap is not None:
                    ifds[0][320] = Tag('SHORT', self.colormap_bytes)
                    ifds[0][262] = Tag('SHORT', 3)
                if self.colors is not None:
                    for c, color in enumerate(self.colors_bytes):
                        ifds[c][320] = Tag('SHORT', color)
                        ifds[c][262] = Tag('SHORT', 3)
                if 0 in ifds and 306 not in ifds[0]:
                    ifds[0][306] = Tag('ASCII', datetime.now().strftime('%Y:%m:%d %H:%M:%S'))
                wrn = False
                for framenr in range(self.nframes):
                    if framenr in ifds and all([(framenr, channel) in strips for channel in range(self.spp)]):
                        stripbyteoffsets, stripbytecounts = zip(*[strips[(framenr, channel)]
                                                                  for channel in range(self.spp)])
                        ifds[framenr][258].value = self.spp * ifds[framenr][258].value
                        ifds[framenr][270] = Tag('ASCII', self.description)
                        ifds[framenr][273] = Tag('LONG8', sum(stripbyteoffsets, []))
                        ifds[framenr][277] = Tag('SHORT', self.spp)
                        ifds[framenr][279] = Tag('LONG8', sum(stripbytecounts, []))
                        ifds[framenr][305] = Tag('ASCII', 'tiffwrite_tllab_NKI')
                        if self.extratags is not None:
                            ifds[framenr].update(self.extratags)
                        if self.colormap is None and self.colors is None and self.shape[0] > 1:
                            ifds[framenr][284] = Tag('SHORT', 2)
                        ifds[framenr].write(fh, self.header, self.write)
                        if framenr:
                            ifds[framenr].write_offset(ifds[framenr - 1].where_to_write_next_ifd_offset)
                        else:
                            ifds[framenr].write_offset(self.header.offset - self.header.offsetsize)
                    else:
                        wrn = True
                if wrn:
                    warnings.warn('Some frames were not added to the tif file, either you forgot them, '
                                  'or an error occured and the tif file was closed prematurely.')

    def __enter__(self) -> IJTiffFile:
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.close()

    @staticmethod
    def hash_check(fh: BinaryIO, bvalue: bytes, offset: int) -> bool:
        addr = fh.tell()
        fh.seek(offset)
        same = bvalue == fh.read(len(bvalue))
        fh.seek(addr)
        return same

    def write(self, fh: BinaryIO, bvalue: bytes) -> int:
        hash_value = sha1(bvalue).hexdigest()  # hash uses a random seed making hashes different in different processes
        if hash_value in self.hashes and self.hash_check(fh, bvalue, self.hashes[hash_value]):
            return self.hashes[hash_value]  # reuse previously saved data
        else:
            if fh.tell() % 2:
                fh.write(b'\x00')
            offset = fh.tell()
            self.hashes[hash_value] = offset
            fh.write(bvalue)
            return offset

    def compress_frame(self, frame: np.ndarray) -> tuple[IFD, tuple[list[int], list[int]]]:
        """ This is run in a different process"""
        stripbytecounts, ifd, chunks = self.get_chunks(self.ij_tiff_frame(frame))
        stripbyteoffsets = []
        with self.fh.lock() as fh:  # noqa
            for chunk in chunks:
                stripbyteoffsets.append(self.write(fh, chunk))
        return ifd, (stripbyteoffsets, stripbytecounts)

    @staticmethod
    def get_chunks(frame: bytes) -> tuple[list[int], IFD, list[bytes]]:
        with BytesIO(frame) as fh:
            ifd = IFD(fh)
            stripoffsets = ifd[273].value
            stripbytecounts = ifd[279].value
            chunks = []
            for stripoffset, stripbytecount in zip(stripoffsets, stripbytecounts):
                fh.seek(stripoffset)
                chunks.append(fh.read(stripbytecount))
        return stripbytecounts, ifd, chunks


class FileHandle:
    """ Process safe file handle """
    def __init__(self, path: Path) -> None:
        manager = PoolSingleton().manager
        if path.exists():
            path.unlink()
        with open(path, 'xb'):
            pass
        self.path = path
        self._lock = manager.RLock()
        self._pos = manager.Value('i', 0)

    @contextmanager
    def lock(self) -> Generator[BinaryIO, None, None]:
        with self._lock:
            with open(self.path, 'rb+') as f:
                try:
                    f.seek(self._pos.value)
                    yield f
                finally:
                    self._pos.value = f.tell()

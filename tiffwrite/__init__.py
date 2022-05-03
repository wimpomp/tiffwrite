import tifffile
import colorcet
import struct
import numpy as np
import multiprocessing
from io import BytesIO
from tqdm.auto import tqdm
from itertools import product
from collections.abc import Iterable
from numbers import Number
from fractions import Fraction
from traceback import print_exc, format_exc
from functools import cached_property
from datetime import datetime
from matplotlib import colors as mpl_colors
from contextlib import contextmanager
from warnings import warn

__all__ = ['IJTiffFile', 'Tag', 'tiffwrite']


def tiffwrite(file, data, axes='TZCXY', dtype=None, bar=False, *args, **kwargs):
    """ file:       string; filename of the new tiff file
        data:       2 to 5D numpy array
        axes:       string; order of dimensions in data, default: TZCXY for 5D, ZCXY for 4D, CXY for 3D, XY for 2D data
        dtype:      string; datatype to use when saving to tiff
        bar:        bool; whether to show a progress bar
        other args: see IJTiffFile
    """

    axes = axes[-np.ndim(data):].upper()
    if not axes == 'CZTXY':
        T = [axes.find(i) for i in 'CZTXY']
        E = [i for i, j in enumerate(T) if j < 0]
        T = [i for i in T if i >= 0]
        data = np.transpose(data, T)
        for e in E:
            data = np.expand_dims(data, e)

    shape = data.shape[:3]
    with IJTiffFile(file, shape, data.dtype if dtype is None else dtype, *args, **kwargs) as f:
        at_least_one = False
        for n in tqdm(product(*[range(i) for i in shape]), total=np.prod(shape), desc='Saving tiff', disable=not bar):
            if np.any(data[n]) or not at_least_one:
                f.save(data[n], *n)
                at_least_one = True


class Header:
    def __init__(self, *args):
        if len(args) == 1:
            fh = args[0]
            fh.seek(0)
            self.byteorder = {b'II': '<', b'MM': '>'}[fh.read(2)]
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
        else:
            self.byteorder, self.bigtiff = args if len(args) == 2 else ('<', True)
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

    def write(self, fh):
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
    tiff_tag_registry = tifffile.TiffTagRegistry({key: value.lower() for key, value in tifffile.TIFF.TAGS.items()})

    @staticmethod
    def to_tags(tags):
        return {(key if isinstance(key, Number) else (int(key[3:]) if key.lower().startswith('tag')
                                       else Tag.tiff_tag_registry[key.lower()])):
                    tag if isinstance(tag, Tag) else Tag(tag) for key, tag in tags.items()}

    @staticmethod
    def fraction(numerator=0, denominator=None):
        return Fraction(numerator, denominator).limit_denominator(2 ** (31 if numerator < 0 or denominator < 0
                                                                        else 32) - 1)

    def __init__(self, ttype, value=None, offset=None):
        if value is None:
            self.value = ttype
            if all([isinstance(value, int) for value in self.value]):
                min_value = np.min(self.value)
                max_value = np.max(self.value)
                type_map = {'uint8': 'byte', 'int8': 'sbyte', 'uint16': 'short', 'int16': 'sshort',
                            'uint32': 'long', 'int32': 'slong', 'uint64': 'long8', 'int64': 'slong8'}
                for dtype, ttype in type_map.items():
                    if np.iinfo(dtype).min <= min_value and max_value <= np.iinfo(dtype).max:
                        break
                    else:
                        ttype = 'undefined'
            elif isinstance(self.value, (str, bytes)) or all([isinstance(value, (str, bytes)) for value in self.value]):
                ttype = 'ascii'
            elif all([isinstance(value, Fraction) for value in self.value]):
                if all([value.numerator < 0 or value.denominator < 0 for value in self.value]):
                    ttype = 'srational'
                else:
                    ttype = 'rational'
            elif all([isinstance(value, (float, int)) for value in self.value]):
                min_value = np.min(np.asarray(self.value)[np.isfinite(self.value)])
                max_value = np.max(np.asarray(self.value)[np.isfinite(self.value)])
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
            self.ttype = tifffile.TIFF.DATATYPES[ttype.upper()]
        else:
            self.value = value
            self.ttype = tifffile.TIFF.DATATYPES[ttype.upper()] if isinstance(ttype, str) else ttype
        self.dtype = tifffile.TIFF.DATA_FORMATS[self.ttype]
        self.offset = offset
        self.type_check()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value if isinstance(value, Iterable) else (value,)

    def __repr__(self):
        if self.offset is None:
            return f'{tifffile.TIFF.DATATYPES(self.ttype).name}: {self.value}'
        else:
            return f'{tifffile.TIFF.DATATYPES(self.ttype).name} @ {self.offset}: {self.value}'

    def type_check(self):
        try:
            self.bytes_and_count(Header())
        except Exception:
            raise ValueError(f"tif tag type '{tifffile.TIFF.DATATYPES(self.ttype).name}' and "
                             f"data type '{type(self.value[0]).__name__}' do not correspond")

    def bytes_and_count(self, header):
        if isinstance(self.value, bytes):
            return self.value, len(self.value) // struct.calcsize(self.dtype)
        elif self.ttype in (2, 14):
            if isinstance(self.value, str):
                bytes_value = self.value.encode('ascii') + b'\x00'
            else:
                bytes_value = b'\x00'.join([value.encode('ascii') for value in self.value]) + b'\x00'
            return bytes_value, len(bytes_value)
        elif self.ttype in (5, 10):
            return b''.join([struct.pack(header.byteorder + self.dtype,
                                         *((value.denominator, value.numerator) if isinstance(value, Fraction)
                                           else value)) for value in self.value]), len(self.value)
        else:
            return b''.join([struct.pack(header.byteorder + self.dtype, value) for value in self.value]), \
                   len(self.value)

    def write_tag(self, fh, key, header, offset=None):
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

    def write_data(self, write=None):
        if self.bytes_data:
            self.fh.seek(0, 2)
            if write is None:
                offset = self.write(self.bytes_data)
            else:
                offset = write(self.fh, self.bytes_data)
            self.fh.seek(self.offset + self.header.tagsize - self.header.offsetsize)
            self.fh.write(struct.pack(self.header.byteorder + self.header.offsetformat, offset))

    def write(self, bytes_value):
        if self.fh.tell() % 2:
            self.fh.write(b'\x00')
        offset = self.fh.tell()
        self.fh.write(bytes_value)
        return offset

    def copy(self):
        return self.__class__(self.ttype, self.value[:], self.offset)


class IFD(dict):
    def __init__(self, fh=None):
        super().__init__()
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
                    value = fh.read(count)
                elif ttype == 2:
                    value = fh.read(count).decode('ascii').rstrip('\x00')
                elif ttype in (5, 10):
                    value = [struct.unpack(header.byteorder + dtype, fh.read(dtypelen)) for _ in range(count)]
                else:
                    value = [struct.unpack(header.byteorder + dtype, fh.read(dtypelen))[0] for _ in range(count)]

                if toolong:
                    fh.seek(cp)

                self[code] = Tag(ttype, value, pos)
            fh.seek(header.offset)

    def __setitem__(self, key, tag):
        super().__setitem__(Tag.tiff_tag_registry[key.lower()] if isinstance(key, str) else key,
                            tag if isinstance(tag, Tag) else Tag(tag))

    def items(self):
        return ((key, self[key]) for key in sorted(self))

    def keys(self):
        return (key for key in sorted(self))

    def values(self):
        return (self[key] for key in sorted(self))

    def write(self, fh, header, write=None):
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

    def write_offset(self, where_to_write_offset):
        self.fh.seek(where_to_write_offset)
        self.fh.write(struct.pack(self.header.byteorder + self.header.offsetformat, self.offset))

    def copy(self):
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
    def __init__(self, path, shape, dtype='uint16', colors=None, colormap=None, pxsize=None, deltaz=None,
                 timeinterval=None, **extratags):
        assert len(shape) >= 3, 'please specify all c, z, t for the shape'
        assert len(shape) <= 3, 'please specify only c, z, t for the shape'
        assert np.dtype(dtype).char in 'BbHhf', 'datatype not supported'
        assert colors is None or colormap is None, 'cannot have colors and colormap simultaneously'

        self.path = path
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.colors = colors
        self.colormap = colormap
        self.pxsize = pxsize
        self.deltaz = deltaz
        self.timeinterval = timeinterval
        self.extratags = {} if extratags is None else Tag.to_tags(extratags)
        if pxsize is not None:
            pxsize = Tag.fraction(pxsize)
            self.extratags.update({282: Tag(pxsize), 283: Tag(pxsize)})

        self.header = Header()
        self.frames = []
        self.spp = self.shape[0] if self.colormap is None and self.colors is None else 1  # samples/pixel
        self.nframes = np.prod(self.shape[1:]) if self.colormap is None and self.colors is None else np.prod(self.shape)
        self.offsets = {}
        self.fh = FileHandle(path, 'w+b')
        self.namespace_manager = multiprocessing.Manager()
        self.hashes = self.namespace_manager.dict()
        self.strips = {}
        self.ifds = {}
        self.frame_extra_tags = {}
        self.frames_added = []
        self.frames_written = []
        self.main_pid = multiprocessing.current_process().pid
        self.pool_manager = PoolManager(self)
        with self.fh.lock() as fh:
            self.header.write(fh)

    def __hash__(self):
        return hash(self.path)

    def get_frame_number(self, n):
        if self.colormap is None and self.colors is None:
            return n[1] + n[2] * self.shape[1], n[0]
        else:
            return n[0] + n[1] * self.shape[0] + n[2] * self.shape[0] * self.shape[1], 0

    def ij_tiff_frame(self, frame):
        with BytesIO() as framedata:
            with tifffile.TiffWriter(framedata, self.header.bigtiff, self.header.byteorder) as t:
                # predictor=True might save a few bytes, but requires the package imagecodes to save floats
                t.write(frame, compression=(8, 9), contiguous=True, predictor=False)
            return framedata.getvalue()

    def save(self, frame, c, z, t, **extratags):
        """ save a 2d numpy array to the tiff at channel=c, slice=z, time=t, with optional extra tif tags
        """
        assert (c, z, t) not in self.frames_written, f'frame {c} {z} {t} is added already'
        assert all([0 <= i < s for i, s in zip((c, z, t), self.shape)]), \
            'frame {} {} {} is outside shape {} {} {}'.format(c, z, t, *self.shape)
        self.frames_added.append((c, z, t))
        self.pool_manager.add_frame(self.path, frame.astype(self.dtype), (c, z, t))
        if extratags:
            self.frame_extra_tags[(c, z, t)] = Tag.to_tags(extratags)

    @cached_property
    def loader(self):
        tif = self.ij_tiff_frame(np.zeros((8, 8), self.dtype))
        with BytesIO(tif) as fh:
            return tif, Header(fh), IFD(fh)

    def load(self, *n):
        """ read back a frame that's just written
            useful for simulating a large number of frames without using much memory
        """
        if n not in self.frames_added:
            raise KeyError(f'frame {n} has not been added yet')
        while n not in self.frames_written:
            self.pool_manager.get_ifds_from_queue()
        tif, header, ifd = self.loader
        framenumber, channel = self.get_frame_number(n)
        with BytesIO(tif) as fh:
            fh.seek(0, 2)
            fstripbyteoffsets, fstripbytecounts = [], []
            with open(self.path, 'rb') as file:
                for stripbyteoffset, stripbytecount in zip(*self.strips[(framenumber, channel)]):
                    file.seek(stripbyteoffset)
                    bdata = file.read(stripbytecount)
                    fstripbyteoffsets.append(fh.tell())
                    fstripbytecounts.append(len(bdata))
                    fh.write(bdata)
            ifd = ifd.copy()
            for key, value in zip((257, 256, 278, 270, 273, 279),
                                  (*self.frame_shape, self.frame_shape[0] // len(fstripbyteoffsets),
                                   f'{{"shape": [{self.frame_shape[0]}, {self.frame_shape[1]}]}}',
                                   fstripbyteoffsets, fstripbytecounts)):
                tag = ifd[key]
                tag.value = value
                tag.write_tag(fh, key, header, tag.offset)
                tag.write_data()
            fh.seek(0)
            return tifffile.TiffFile(fh).asarray()

    @property
    def description(self):
        desc = ['ImageJ=1.11a']
        if self.colormap is None and self.colors is None:
            desc.extend((f'images={np.prod(self.shape[:1])}', f'slices={self.shape[1]}', f'frames={self.shape[2]}'))
        else:
            desc.extend((f'images={np.prod(self.shape)}', f'channels={self.shape[0]}', f'slices={self.shape[1]}',
                         f'frames={self.shape[2]}'))
        desc.extend(('hyperstack=true', 'mode=grayscale', 'loop=false', 'unit=micron'))
        if self.deltaz is not None:
            desc.append(f'spacing={self.deltaz}')
        if self.timeinterval is not None:
            desc.append(f'finterval={self.timeinterval}')
        return bytes('\n'.join(desc), 'ascii')

    @cached_property
    def empty_frame(self):
        ifd = self.ifds[list(self.ifds.keys())[-1]].copy()
        return self.compress_frame(np.zeros((ifd[257].value[0], ifd[256].value[0]), self.dtype))

    @cached_property
    def frame_shape(self):
        ifd = self.ifds[list(self.ifds.keys())[-1]].copy()
        return ifd[257].value[0], ifd[256].value[0]

    def add_empty_frame(self, n):
        framenr, channel = self.get_frame_number(n)
        ifd, strips = self.empty_frame
        self.ifds[framenr] = ifd.copy()
        self.strips[(framenr, channel)] = strips

    @cached_property
    def colormap_bytes(self):
        colormap = getattr(colorcet, self.colormap)
        colormap[0] = '#ffffff'
        colormap[-1] = '#000000'
        colormap = 65535 * np.array(
            [[int(''.join(i), 16) for i in zip(*[iter(s[1:])] * 2)] for s in colormap]) // 255
        if np.dtype(self.dtype).itemsize == 2:
            colormap = np.tile(colormap, 256).reshape((-1, 3))
        return b''.join([struct.pack(self.header.byteorder + 'H', c) for c in colormap.T.flatten()])

    @cached_property
    def colors_bytes(self):
        return [b''.join([struct.pack(self.header.byteorder + 'H', c)
                          for c in np.linspace(0, 65535 * np.array(mpl_colors.to_rgb(color)),
                                               65536 if np.dtype(self.dtype).itemsize == 2 else 256,
                                               dtype=int).T.flatten()]) for color in self.colors]

    def close(self):
        assert len(self.frames_added) >= 1, 'at least one frame should be added to the tiff'
        if multiprocessing.current_process().pid == self.main_pid:
            self.pool_manager.close(self)
            with self.fh.lock() as fh:
                if len(self.frames_written) < np.prod(self.shape):  # add empty frames if needed
                    for n in product(*[range(i) for i in self.shape]):
                        if n not in self.frames_written:
                            self.add_empty_frame(n)

                for n, tags in self.frame_extra_tags.items():
                    framenr, channel = self.get_frame_number(n)
                    self.ifds[framenr].update(tags)
                if self.colormap is not None:
                    self.ifds[0][320] = Tag('SHORT', self.colormap_bytes)
                    self.ifds[0][262] = Tag('SHORT', 3)
                if self.colors is not None:
                    for c, color in enumerate(self.colors_bytes):
                        self.ifds[c][320] = Tag('SHORT', color)
                        self.ifds[c][262] = Tag('SHORT', 3)
                if 306 not in self.ifds[0]:
                    self.ifds[0][306] = Tag('ASCII', datetime.now().strftime('%Y:%m:%d %H:%M:%S'))
                for framenr in range(self.nframes):
                    stripbyteoffsets, stripbytecounts = zip(*[self.strips[(framenr, channel)]
                                                              for channel in range(self.spp)])
                    self.ifds[framenr][258].value = self.spp * self.ifds[framenr][258].value
                    self.ifds[framenr][270] = Tag('ASCII', self.description)
                    self.ifds[framenr][273] = Tag('LONG8', sum(stripbyteoffsets, []))
                    self.ifds[framenr][277] = Tag('SHORT', self.spp)
                    self.ifds[framenr][279] = Tag('LONG8', sum(stripbytecounts, []))
                    self.ifds[framenr][305] = Tag('ASCII', 'tiffwrite_tllab_NKI')
                    if self.extratags is not None:
                        self.ifds[framenr].update(self.extratags)
                    if self.colormap is None and self.colors is None and self.shape[0] > 1:
                        self.ifds[framenr][284] = Tag('SHORT', 2)
                    self.ifds[framenr].write(fh, self.header, self.write)
                    if framenr:
                        self.ifds[framenr].write_offset(self.ifds[framenr - 1].where_to_write_next_ifd_offset)
                    else:
                        self.ifds[framenr].write_offset(self.header.offset - self.header.offsetsize)
            fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    @staticmethod
    def hash_check(fh, bvalue, offset):
        addr = fh.tell()
        fh.seek(offset)
        same = bvalue == fh.read(len(bvalue))
        fh.seek(addr)
        return same

    def write(self, fh, bvalue):
        hash_value = hash(bvalue)
        if hash_value in self.hashes and self.hash_check(fh, bvalue, self.hashes[hash_value]):
            return self.hashes[hash_value]  # reuse previously saved data
        else:
            if fh.tell() % 2:
                fh.write(b'\x00')
            offset = fh.tell()
            self.hashes[hash_value] = offset
            fh.write(bvalue)
            return offset

    def compress_frame(self, frame):
        """ This is run in a different process"""
        stripbytecounts, ifd, chunks = self.get_chunks(self.ij_tiff_frame(frame))
        stripbyteoffsets = []
        with self.fh.lock() as fh:
            for chunk in chunks:
                stripbyteoffsets.append(self.write(fh, chunk))
        return ifd, (stripbyteoffsets, stripbytecounts)

    @staticmethod
    def get_chunks(frame):
        with BytesIO(frame) as fh:
            ifd = IFD(fh)
            stripoffsets = ifd[273].value
            stripbytecounts = ifd[279].value
            chunks = []
            for stripoffset, stripbytecount in zip(stripoffsets, stripbytecounts):
                fh.seek(stripoffset)
                chunks.append(fh.read(stripbytecount))
        return stripbytecounts, ifd, chunks


class PoolManager:
    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self, tif, processes=None):
        if not hasattr(self, 'tifs'):
            self.tifs = {}
        if not hasattr(self, 'is_alive'):
            self.is_alive = False
        if self.is_alive:
            raise ValueError('Cannot start new tifffile until previous tifffiles have been closed.')
        self.tifs[tif.path] = tif
        self.processes = processes

    def close(self, tif):
        while len(tif.frames_written) < len(tif.frames_added):
            self.get_ifds_from_queue()
        self.tifs.pop(tif.path)
        if not self.tifs:
            self.__class__.instance = None
            self.is_alive = False
            self.done.set()
            while not self.queue.empty():
                self.queue.get()
            self.queue.close()
            self.queue.join_thread()
            while not self.error_queue.empty():
                print(self.error_queue.get())
            self.error_queue.close()
            self.ifd_queue.close()
            self.ifd_queue.join_thread()
            self.pool.close()
            self.pool.join()

    def get_ifds_from_queue(self):
        while not self.ifd_queue.empty():
            file, n, ifd, strip = self.ifd_queue.get()
            framenr, channel = self.tifs[file].get_frame_number(n)
            self.tifs[file].ifds[framenr] = ifd
            self.tifs[file].strips[(framenr, channel)] = strip
            self.tifs[file].frames_written.append(n)

    def add_frame(self, *args):
        if not self.is_alive:
            self.start_pool()
        self.get_ifds_from_queue()
        self.queue.put(args)

    def start_pool(self):
        self.is_alive = True
        nframes = sum([np.prod(tif.shape) for tif in self.tifs.values()])
        self.processes = self.processes or max(2, min(multiprocessing.cpu_count() // 6, nframes))
        self.queue = multiprocessing.Queue(10 * self.processes)
        self.ifd_queue = multiprocessing.Queue(10 * self.processes)
        self.error_queue = multiprocessing.Queue()
        self.offsets_queue = multiprocessing.Queue()
        self.done = multiprocessing.Event()
        self.pool = multiprocessing.Pool(self.processes, self.run)

    def run(self):
        """ Only this is run in parallel processes. """
        try:
            while not self.done.is_set():
                try:
                    file, frame, n = self.queue.get(True, 0.02)
                    self.ifd_queue.put((file, n, *self.tifs[file].compress_frame(frame)))
                except multiprocessing.queues.Empty:
                    continue
        except Exception:
            print_exc()
            self.error_queue.put(format_exc())


class FileHandle:
    """ Process safe file handle """
    def __init__(self, name, mode='rb'):
        self.name = name
        self.mode = mode
        self._lock = multiprocessing.RLock()
        self._fh = open(name, mode, 0)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        self._fh.close()

    @contextmanager
    def lock(self):
        self._lock.acquire()
        try:
            yield self._fh
        finally:
            self._lock.release()


class IJTiffWriter:
    def __init__(self, file, shape, dtype='uint16', colormap=None, nP=None, extratags=None, pxsize=None):
        warn('IJTiffWriter is deprecated and will be removed in a future version, use IJTiffFile instead',
             DeprecationWarning)
        files = [file] if isinstance(file, str) else file
        shapes = [shape] if isinstance(shape[0], Number) else shape  # CZT
        dtypes = [np.dtype(dtype)] if isinstance(dtype, (str, np.dtype)) else [np.dtype(d) for d in dtype]
        colormaps = [colormap] if colormap is None or isinstance(colormap, str) else colormap
        extratagss = [extratags] if extratags is None or isinstance(extratags, dict) else extratags
        pxsizes = [pxsize] if pxsize is None or isinstance(pxsize, Number) else pxsize

        nFiles = len(files)
        if not len(shapes) == nFiles:
            shapes *= nFiles
        if not len(dtypes) == nFiles:
            dtypes *= nFiles
        if not len(colormaps) == nFiles:
            colormaps *= nFiles
        if not len(extratagss) == nFiles:
            extratagss *= nFiles
        if not len(pxsizes) == nFiles:
            pxsizes *= nFiles

        self.files = {file: IJTiffFile(file, shape, dtype, None, colormap, pxsize, **(extratags or {}))
                      for file, shape, dtype, colormap, pxsize, extratags
                      in zip(files, shapes, dtypes, colormaps, pxsizes, extratagss)}

        assert np.all([len(s) == 3 for s in shapes]), 'please specify all c, z, t for the shape'
        assert np.all([d.char in 'BbHhf' for d in dtypes]), 'datatype not supported'

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def save(self, file, frame, *n):
        if isinstance(file, np.ndarray):  # save to first/only file
            n = (frame, *n)
            frame = file
            file = next(iter(self.files.keys()))
        elif isinstance(file, Number):
            file = list(self.files.keys())[int(file)]
        self.files[file].save(frame, *n)

    def close(self):
        for file in self.files.values():
            try:
                file.close()
            except Exception:
                print_exc()

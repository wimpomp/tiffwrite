import sys
import struct
import numpy as np
from io import BytesIO
from multiprocessing import Pool, Process, Queue, Event, cpu_count, Value, queues
from time import sleep
from tqdm.auto import tqdm
import tifffile
import colorcet
from itertools import product
from collections import OrderedDict
from multipledispatch import dispatch
from numbers import Number
from fractions import Fraction


def get_colormap(colormap, dtype='int8', byteorder='<'):
    colormap = getattr(colorcet, colormap)
    colormap[0] = '#ffffff'
    colormap[-1] = '#000000'
    colormap = 65535 * np.array([[int(''.join(i), 16) for i in zip(*[iter(s[1:])] * 2)] for s in colormap]) // 255
    if np.dtype(dtype).itemsize == 2:
        colormap = np.tile(colormap, 256).reshape((-1, 3))
    return b''.join([struct.pack(byteorder + 'H', c) for c in colormap.T.flatten()])


def tiffwrite(file, data, axes='TZCXY', bar=False, colormap=None, pxsize=None):
    """ file:     string; filename of the new tiff file
        data:     2 to 5D numpy array
        axes:     string; order of dimensions in data, default: TZCXY for 5D, ZCXY for 4D, CXY for 3D, XY for 2D data
        bar:      bool; whether or not to show a progress bar
        colormap: string; choose any colormap from the colorcet module
        pxsize:   float; set tiff tag so ImageJ can read the pixel size
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
    with IJTiffWriter(file, shape, data.dtype, colormap, pxsize=pxsize) as f:
        at_least_one = False
        for n in tqdm(product(*[range(i) for i in shape]), total=np.prod(shape), desc='Saving tiff', disable=not bar):
            if np.any(data[n]) or not at_least_one:
                f.save(data[n], *n)
                at_least_one = True


def readheader(b):
    b.seek(0)
    byteorder = {b'II': '<', b'MM': '>'}[b.read(2)]
    bigtiff = {42: False, 43: True}[struct.unpack(byteorder + 'H', b.read(2))[0]]

    if bigtiff:
        tagsize = 20
        tagnoformat = 'Q'
        offsetsize = struct.unpack(byteorder + 'H', b.read(2))[0]
        offsetformat = {8: 'Q', 16: '2Q'}[offsetsize]
        assert struct.unpack(byteorder + 'H', b.read(2))[0] == 0, 'Not a TIFF-file'
        offset = struct.unpack(byteorder + offsetformat, b.read(offsetsize))[0]
    else:
        tagsize = 12
        tagnoformat = 'H'
        offsetformat = 'I'
        offsetsize = 4
        offset = struct.unpack(byteorder + offsetformat, b.read(offsetsize))[0]
    return byteorder, bigtiff, tagsize, tagnoformat, offsetformat, offsetsize, offset


def readifd(b):
    """ Reads the first IFD of the tiff file in the file handle b
        wp@tl20200214
    """
    byteorder, bigtiff, tagsize, tagnoformat, offsetformat, offsetsize, offset = readheader(b)

    b.seek(offset)
    nTags = struct.unpack(byteorder + tagnoformat, b.read(struct.calcsize(tagnoformat)))[0]
    assert nTags < 4096, 'Too many tags'
    addr = []
    addroffset = []

    length = 8 if bigtiff else 2
    length += nTags * tagsize + offsetsize

    tags = {}
    for i in range(nTags):
        pos = offset + struct.calcsize(tagnoformat) + tagsize * i
        b.seek(pos)

        code, ttype = struct.unpack(byteorder + 'HH', b.read(4))
        count = struct.unpack(byteorder + offsetformat, b.read(offsetsize))[0]

        dtype = tifffile.TIFF.DATA_FORMATS[ttype]
        dtypelen = struct.calcsize(dtype)

        toolong = struct.calcsize(dtype) * count > offsetsize
        if toolong:
            addr.append(b.tell() - offset)
            caddr = struct.unpack(byteorder + offsetformat, b.read(offsetsize))[0]
            addroffset.append(caddr - offset)
            cp = b.tell()
            b.seek(caddr)

        if ttype == 1:
            value = b.read(count)
        elif ttype == 2:
            value = b.read(count).decode('ascii').rstrip('\x00')
        elif ttype == 5:
            value = [struct.unpack(byteorder + dtype, b.read(dtypelen)) for _ in range(count)]
        else:
            value = [struct.unpack(byteorder + dtype, b.read(dtypelen))[0] for _ in range(count)]

        if toolong:
            b.seek(cp)

        tags[code] = (ttype, value)

    b.seek(offset)
    return tags


def getchunks(frame):
    with BytesIO(frame) as b:
        tags = readifd(b)
        stripoffsets = tags[273][1]
        stripbytecounts = tags[279][1]
        chunks = []
        for o, c in zip(stripoffsets, stripbytecounts):
            b.seek(o)
            chunks.append(b.read(c))
    return stripbytecounts, tags, chunks


def fmt_err(exc_info):
    t, m, tb = exc_info
    while tb.tb_next:
        tb = tb.tb_next
    return 'line {}: {}'.format(tb.tb_lineno, m)


def multiplexer(files, byteorder, bigtiff, Qo, V, W, E):
    try:
        w = {file: writer(file, v['shape'], byteorder, bigtiff, W, v['colormap'], v['dtype'], v['extratags'])
             for file, v in files.items()}
        for v in w.values():  # start writing all files
            next(v)
        while not V.is_set():  # take frames from queue and write to file
            try:
                frame, file, n, fmin, fmax = Qo.get(True, 0.02)
                w[file].send((frame, n, fmin, fmax))
            except queues.Empty:
                continue
        for v in w.values():  # finish writing files
            v.close()
    except Exception:
        E.put(fmt_err(sys.exc_info()))


def writer(file, shape, byteorder, bigtiff, W, colormap=None, dtype=None, extratags=None):
    """ Writes a tiff file, writer function for IJTiffWriter
        file:      filename of the new tiff file
        shape:     shape (CZT) of the data to be written
        byteorder: byteorder of the file to be written, '<' or '>'
        bigtiff:   False: file will be normal tiff, True: file will be bigtiff
        Qo:        Queue from which to take the compressed frames for writing
        V:         Value; 1 when more frames need to be written, 0 when writer can finish
        W:         Value in which writer will log how many frames are written
        colormap:  array with 2^bitspersample x 3 values RGB
        wp@tl20200214
    """

    spp = shape[0] if colormap is None else 1  # samples/pixel
    nframes = np.prod(shape[1:]) if colormap is None else np.prod(shape)
    offsetformat, offsetsize, tagnoformat, tagsize = (('I', 4, 'H', 8), ('Q', 8, 'Q', 20))[bigtiff]
    strips = {}
    tags = {}
    hashes = {}
    N = []

    def hashcheck(bvalue, offset):
        addr = fh.tell()
        fh.seek(offset)
        same = bvalue == fh.read(len(bvalue))
        fh.seek(addr)
        return same

    def frn(n):
        if colormap is None:
            return n[1] + n[2] * shape[1], n[0]
        else:
            return n[0] + n[1] * shape[0] + n[2] * shape[0] * shape[1], 0

    def addframe(frame, n):
        framenr, channel = frn(n)
        stripbytecounts, tags[framenr], chunks = getchunks(frame)
        stripbyteoffsets = []
        for c in chunks:
            hc = hash(c)
            if hc in hashes and hashcheck(c, hashes[hc]):  # reuse previously saved data
                stripbyteoffsets.append(hashes[hc])
            else:
                if fh.tell() % 2:
                    fh.write(b'\x00')
                stripbyteoffsets.append(fh.tell())
                hashes[hc] = stripbyteoffsets[-1]
                fh.write(c)  # write the data now, ifds later

        strips[(framenr, channel)] = (stripbyteoffsets, stripbytecounts)
        W.value += 1
        N.append(n)
        return framenr, channel

    def addtag(code, ttype, value):
        if isinstance(ttype, str):
            ttype = tifffile.TIFF.DATATYPES[ttype.upper()]
        dtype = tifffile.TIFF.DATA_FORMATS[ttype]
        count = len(value) // struct.calcsize(dtype) if isinstance(value, (bytes, str)) else len(value)
        offset = fh.tell()

        fh.write(struct.pack(byteorder + 'HH', code, ttype))
        fh.write(struct.pack(byteorder + offsetformat, count))
        if isinstance(value, bytes):
            bvalue = value
        elif isinstance(value, str):
            bvalue = value.encode('ascii')
        elif ttype == 5:
            bvalue = b''.join([struct.pack(byteorder + dtype, *v) for v in value])
        else:
            bvalue = b''.join([struct.pack(byteorder + dtype, v) for v in value])
        if len(bvalue) <= offsetsize:
            fh.write(bvalue)
            tagdata = None
        else:
            tagdata = (fh.tell(), bvalue)
        fh.seek(offset + tagsize)
        return tagdata

    def addtagdata(addr, bvalue):
        if fh.tell() % 2:
            fh.write(b'\x00')
        hc = hash(bvalue)
        if hc in hashes and hashcheck(bvalue, hashes[hc]):
            tagoffset = hashes[hc]
        else:
            tagoffset = fh.tell()
            hashes[hc] = tagoffset
            fh.write(bvalue)
        fh.seek(addr)
        fh.write(struct.pack(byteorder + offsetformat, tagoffset))
        fh.seek(0, 2)

    if colormap is None:
        description = \
            'ImageJ=1.11a\nimages={}\nslices={}\nframes={}\nhyperstack=true\nmode=grayscale\nloop=false\n'. \
                format(np.prod(shape[1:]), *shape[1:])
    else:
        description = \
            'ImageJ=1.11a\nimages={}\nchannels={}\nslices={}\nframes={}\nhyperstack=true\nmode=grayscale\nloop=false\n'. \
                format(np.prod(shape), *shape)
    try:
        description = bytes(description, 'ascii')  # python 3
    except:
        pass

    with open(file, 'w+b') as fh:
        try:
            fh.write({'<': b'II', '>': b'MM'}[byteorder])
            if bigtiff:
                offset = 16
                fh.write(struct.pack(byteorder + 'H', 43))
                fh.write(struct.pack(byteorder + 'H', 8))
                fh.write(struct.pack(byteorder + 'H', 0))
                fh.write(struct.pack(byteorder + 'Q', offset))
            else:
                offset = 8
                fh.write(struct.pack(byteorder + 'H', 42))
                fh.write(struct.pack(byteorder + 'I', offset))

            fminmax = np.tile((np.inf, -np.inf), (shape[0], 1))
            while True:
                frame, n, fmin, fmax = yield
                fminmax[n[0]] = min(fminmax[n[0]][0], fmin), max(fminmax[n[0]][1], fmax)
                addframe(frame, n)
        except GeneratorExit:
            if dtype.kind == 'i':
                dmin, dmax = np.iinfo(dtype).min, np.iinfo(dtype).max
            else:
                dmin, dmax = 0, 65535
            fminmax[np.isposinf(fminmax)] = dmin
            fminmax[np.isneginf(fminmax)] = dmax
            for i in range(fminmax.shape[0]):
                if fminmax[i][0] == fminmax[i][1]:
                    fminmax[i] = dmin, dmax

            if len(N) < np.prod(shape):  # add empty frames if needed
                empty_frame = None
                for n in product(*[range(i) for i in shape]):
                    if not n in N:
                        framenr, channel = frn(n)
                        if empty_frame is None:
                            tag = tags[framenr] if framenr in tags.keys() else tags[list(tags.keys())[-1]]
                            frame = IJTiffFrame(np.zeros(tag[257][1] + tag[256][1], dtype), byteorder, bigtiff)
                            empty_frame = addframe(frame, n)
                        else:
                            strips[(framenr, channel)] = strips[empty_frame]
                            if not framenr in tags.keys():
                                tags[framenr] = tags[empty_frame[0]]

            offset_addr = offset - offsetsize

            if not colormap is None:
                tags[0][320] = (3, get_colormap(colormap, dtype, byteorder))
                tags[0][262] = (3, [3])

            # unfortunately, ImageJ doesn't read this from bigTiff, maybe we'll figure out how to force IJ in the future
            for tag in tifffile.tifffile.imagej_metadata_tag(
                    {'Ranges': tuple(fminmax.flatten().astype(int))}, byteorder):
                tags[0][tag[0]] = ({50839: 1, 50838: 4}[tag[0]], tag[3])

            for framenr in range(nframes):
                stripbyteoffsets, stripbytecounts = zip(*[strips[(framenr, channel)] for channel in range(spp)])
                tp, value = tags[framenr][258]
                tags[framenr][258] = (tp, spp * value)
                tags[framenr][270] = (2, description)
                tags[framenr][273] = (16, sum(stripbyteoffsets, []))
                tags[framenr][277] = (3, [spp])
                tags[framenr][279] = (16, sum(stripbytecounts, []))
                tags[framenr][305] = (2, b'tiffwrite_tllab_NKI')
                if extratags is not None:
                    tags[framenr].update(extratags)
                if colormap is None and shape[0] > 1:
                    tags[framenr][284] = (3, [2])

                # write offset to this ifd in the previous one
                if fh.tell() % 2:
                    fh.write(b'\x00')
                offset = fh.tell()
                fh.seek(offset_addr)
                fh.write(struct.pack(byteorder + offsetformat, offset))

                # write ifd
                fh.seek(offset)
                fh.write(struct.pack(byteorder + tagnoformat, len(tags[framenr])))
                tagdata = [addtag(code, *tags[framenr][code]) for code in sorted(tags[framenr].keys())]
                offset_addr = fh.tell()
                fh.write(b'\x00' * offsetsize)
                for i in [j for j in tagdata if j is not None]:
                    addtagdata(*i)
            fh.seek(offset_addr)
            fh.write(struct.pack(byteorder + tagnoformat, 0))


def IJTiffFrame(frame, byteorder, bigtiff):
    with BytesIO() as framedata:
        with tifffile.TiffWriter(framedata, bigtiff, byteorder) as t:
            t.save(frame, compress=9, contiguous=True)
        return framedata.getvalue()


def compressor(byteorder, bigtiff, Qi, Qo, V, E):
    """ Compresses tiff frames
        byteorder: byteorder of the file to be written, '<' or '>'
        bigtiff:   False: file will be normal tiff, True: file will be bigtiff
        Qi:        Queue from which new frames which need to be compressed are taken
        Qo:        Queue where compressed frames are stored
        V:         Value; 1 when more frames need to be compressed, 0 when compressor can finish
    """
    try:
        while not V.is_set():
            try:
                frame, file, n = Qi.get(True, 0.02)
                if isinstance(frame, tuple):
                    fun, args, kwargs = frame[:3]
                    frame = fun(*args, **kwargs)
                fmin = frame.flatten()
                fmin = fmin[fmin > 0]
                fmin = np.nanmin(fmin) if len(fmin) else np.inf
                fmax = np.nanmax(frame)
                Qo.put((IJTiffFrame(frame, byteorder, bigtiff), file, n, fmin, fmax))
            except queues.Empty:
                continue
    except Exception:
        E.put(fmt_err(sys.exc_info()))


class IJTiffWriter():
    """ Class for writing ImageJ big tiff files using good compression and multiprocessing to compress quickly
        Usage:
            with IJTiffWriter(file, shape) as t:
                t.save(frame, c, z, t)

        file: string; filename of the new tiff file, or list of filenames.
        shape: iterable; shape (C, Z, T) of data to be written in file, or list of shapes.
        dtype: cast data to dtype before saving, only (u)int8, (u)int16 and float32 are supported.
        colormap: string; choose any colormap from the colorcet module.
        nP: int; number of compressor workers to use
        extratags: dict {int tagnr: (int type, data)}, extra tags to save on every frame, will cause a crash if not used correctly!
        pxsize:   float; set tiff tag so ImageJ can read the pixel size (in um).

        frame:    2D numpy array with data
        c, z, t: channel, z, time coordinates of the frame
    """

    # TODO: better error handling
    # TODO: extratags per frame, handled by save method
    # TODO: extratags sanity check

    def __init__(self, file, shape, dtype='uint16', colormap=None, nP=None, extratags=None, pxsize=None):
        files = [file] if isinstance(file, str) else file
        shapes = [shape] if isinstance(shape[0], Number) else shape  # CZT
        dtypes = [np.dtype(dtype)] if isinstance(dtype, (str, np.dtype)) else [np.dtype(d) for d in dtype]
        colormaps = [colormap] if colormap is None or isinstance(colormap, str) else colormap
        extratagss = [extratags] if extratags is None or isinstance(extratags, dict) else extratags
        pxsizes = [pxsize] if pxsize is None or isinstance(pxsize, Number) else pxsize
        for i, pxsize in enumerate(pxsizes):
            if pxsize is not None:
                res = Fraction(pxsize).limit_denominator(2 ** 31 - 1)
                res = [res.denominator, res.numerator]
                extratagss[i] = {**(extratagss[i] or {}), **{282: (5, [res]), 283: (5, [res])}}

        nFiles = len(files)
        if not len(shapes) == nFiles:
            shapes *= nFiles
        if not len(dtypes) == nFiles:
            dtypes *= nFiles
        if not len(colormaps) == nFiles:
            colormaps *= nFiles
        if not len(extratagss) == nFiles:
            extratagss *= nFiles

        self.files = OrderedDict((file,
                    {'shape': shape, 'dtype': dtype, 'colormap': colormap, 'frames': [], 'extratags': extratags})
                    for file, shape, dtype, colormap, extratags in zip(files, shapes, dtypes, colormaps, extratagss)
                                 if len(file))

        assert np.all([len(s) == 3 for s in shapes]), 'please specify all c, z, t for the shape'
        assert np.all([d.char in 'BbHhf' for d in dtypes]), 'datatype not supported'
        self.bigtiff = True  # normal tiff also possible, but should be opened by bioformats in ImageJ
        self.byteorder = '<'
        self.nP = nP or max(2, min(cpu_count() // 6, np.prod(shape)))
        self.Qi = Queue(10 * self.nP)
        self.Qo = Queue(10 * self.nP)
        self.E = Queue()
        self.V = Event()
        self.W = Value('i', 0)
        self.Compressor = Pool(self.nP, compressor, (self.byteorder, self.bigtiff, self.Qi, self.Qo, self.V, self.E))
        self.Writer = Process(target=multiplexer, args=(self.files, self.byteorder, self.bigtiff, self.Qo, self.V,
                                                        self.W, self.E))
        self.Writer.start()

    @dispatch(object, Number, Number, Number)
    def save(self, frame, *n):
        self.save(next(iter(self.files.keys())), frame, *n)

    @dispatch(Number, object, Number, Number, Number)
    def save(self, filenr, frame, *n):
        self.save(list(self.files.keys())[filenr], frame, *n)

    @dispatch(str, object, Number, Number, Number)
    def save(self, file, frame, *n):
        assert file in self.files, 'file was not opened by {}'.format(self)
        assert n not in self.files[file]['frames'], 'frame {} {} {} is present already'.format(*n)
        assert all([0 <= i < s for i, s in zip(n, self.files[file]['shape'])]), \
            'frame {} {} {} is outside shape {} {} {}'.format(n[0], n[1], n[2], *self.files[file]['shape'])
        if not self.E.empty():
            print(self.E.get())
        # fun, args, kwargs, dshape = frame
        if not isinstance(frame, tuple):
            assert frame.ndim == 2, 'data must be 2 dimensional'
            if not self.files[file]['dtype'] is None:
                frame = frame.astype(self.files[file]['dtype'])
        self.files[file]['frames'].append(n)
        self.Qi.put((frame, file, n))

    def close(self):
        nFrames = sum([len(v['frames']) for v in self.files.values()])
        if self.W.value < nFrames:
            with tqdm(total=nFrames, leave=False, desc='Finishing writing frames',
                      disable=(nFrames - self.W.value) < 100) as bar:
                while self.W.value < nFrames:
                    if not self.E.empty():
                        print(self.E.get())
                        break
                    bar.n = self.W.value
                    bar.refresh()
                    sleep(0.02)
                bar.n = sum([len(v['frames']) for v in self.files.values()])
                bar.refresh()

        self.V.set()
        while not self.Qi.empty():
            self.Qi.get()
        self.Qi.close()
        self.Qi.join_thread()
        while not self.Qo.empty():
            self.Qo.get()
        self.Qo.close()
        self.Qo.join_thread()
        while not self.E.empty():
            print(self.E.get())
        self.E.close()
        self.Compressor.close()
        self.Compressor.join()
        self.Writer.join(5)
        if self.Writer.is_alive():
            self.Writer.terminate()
            self.Writer.join(5)
            if self.Writer.is_alive():
                print('Writer process won''t close.')

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

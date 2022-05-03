# Tiffwrite
Exploiting [tifffile](https://pypi.org/project/tifffile/) in parallel to write BioFormats/ImageJ compatible tiffs with
good compression.

## Features
- Writes bigtiff files that open in ImageJ as hyperstack with correct dimensions.
- Parallel compression.
- Write individual frames in random order.
- Compresses even more by referencing tag or image data which otherwise would have been saved several times.
For example empty frames, or a long string tag on every frame.
- Enables memory efficient scripts by saving frames whenever they're ready to be saved, not waiting for the whole stack.
- Colormaps, extra tags globally or frame dependent.

## Installation
    pip install tiffwrite
or

    pip install tiffwrite@git+https://github.com/wimpomp/tiffwrite

# Usage
## Write an image stack
    tiffwrite(file, data, axes='TZCXY', dtype=None, bar=False, *args, **kwargs)

- file:         string; filename of the new tiff file.
- data:         2 to 5D numpy array in one of these datatypes: (u)int8, (u)int16, float32.
- axes:         string; order of dimensions in data, default: TZCXY for 5D, ZCXY for 4D, CXY for 3D, XY for 2D data.
- dtype:        string; cast data to dtype before saving, only (u)int8, (u)int16 and float32 are supported.
- bar:          bool; whether or not to show a progress bar.
- args, kwargs: arguments to be passed to IJTiffFile, see below.


## Write one frame at a time
    with IJTiffFile(file, shape, dtype='uint16', colors=None, colormap=None, pxsize=None, deltaz=None,
                    timeinterval=None, **extratags) as tif:
    some loop:
        tif.save(frame, c, z, t)

- file:         string; filename of the new tiff file.
- shape:        iterable; shape (C, Z, T) of data to be written in file.
- dtype:        string; cast data to dtype before saving, only (u)int8, (u)int16 and float32 are supported.
- colors:       iterable of strings; one color per channel, valid colors (also html) are defined in matplotlib.colors.
                    Without colormap BioFormats will set the colors in this order: rgbwcmy.
                    Note that the color green is dark, the usual green is named 'lime' here.
- colormap:     string; choose any colormap from the colorcet module. Colors and colormap cannot be used simultaneously.
- pxsize:       float; pixel size im um.
- deltaz:       float; z slice interval in um.
- timeinterval: float; time between frames in seconds.
- extratags:    other tags to be saved, example: Artist='John Doe', Tag4567=[400, 500] or
                    Copyright=Tag('ascii', 'Made by me'). See tiff_tag_registry.items().

- frame:        2D numpy array with data.
- c, z, t:      int; channel, z, time coordinates of the frame.

    
# Examples
## Write an image stack
    from tiffwrite import tiffwrite
    import numpy as np

    image = np.random.randint(0, 255, (5, 3, 64, 64), 'uint16')
    tiffwrite('file.tif', image, 'TCXY')

## Write one frame at a time
    from itertools import product
    from tiffwrite import IJTiffFile
    import numpy as np

    shape = (3, 5, 10)  # channels, z, time
    with IJTiffFile('file.tif', shape, pxsize=0.09707) as tif:
        for c in range(shape[0]):
            for z in range(shape[1]):
                for t in range(shape[2]):
                    tif.save(np.random.randint(0, 10, (32, 32)), c, z, t)

## Saving multiple tiffs simultaneously
    from itertools import product
    from tiffwrite import IJTiffFile
    import numpy as np
    
    shape = (3, 5, 10)  # channels, z, time
    with IJTiffFile('fileA.tif', shape) as tif_a, IJTiffFile('fileB.tif', shape) as tif_b:
        for c in range(shape[0]):
            for z in range(shape[1]):
                for t in range(shape[2]):
                    tif_a.save(np.random.randint(0, 10, (32, 32)), c, z, t)
                    tif_b.save(np.random.randint(0, 10, (32, 32)), c, z, t)

## Tricks & tips
- The order of feeding frames to IJTiffFile is unimportant, IJTiffFile will order de ifd's such that the file will
be opened as a correctly ordered hyperstack.
- Using the colormap parameter you can make ImageJ open the file and apply the colormap. colormap='glasbey' is very
useful.
- IJTiffFile does not allow more than one pool of parallel processes to be open at a time. Therefore, when writing
multiple tiff's simultaneously you have to open all before you start saving any frame, in this way all files share the
same pool.

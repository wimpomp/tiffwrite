# Tiffwrite
Exploiting [tifffile](https://pypi.org/project/tifffile/) in parallel to write ImageJ compatible tiffs with good
compression.

## Features
- Writes bigtiff file that opens in ImageJ as hyperstack with correct dimensions.
- Parallel compression.
- Write individual frames in random order.
- Compresses even more by referencing tag or image data which otherwise would have been save several times.
For example empty frames, or a long string tag on every frame.
- Enables memory efficient scripts by saving frames whenever they're ready to be saved, not waiting for the whole stack.

## Installation
    pip install tiffwrite
or

    pip install tiffwrite@git+https://github.com/wimpomp/tiffwrite

# Usage
## Write an image stack
    tiffwrite(file, data, axes='TZCXY', bar=False, colormap=None, pxsize=None)

- file:     string; filename of the new tiff file.
- data:     2 to 5D numpy array in one of these datatypes: (u)int8, (u)int16, float32.
- axes:     string; order of dimensions in data, default: TZCXY for 5D, ZCXY for 4D, CXY for 3D, XY for 2D data.
- bar:      bool; whether or not to show a progress bar.
- colormap: string; choose any colormap from the colorcet module.
- pxsize:   float; set tiff tag so ImageJ can read the pixel size (in um).

## Write one frame at a time
    with IJTiffWriter(file, shape, dtype='uint16', colormap=None, nP=None, extratags=None, pxsize=None) as tif:
    some loop:
        tif.save(frame, c, z, t)

- file:      string; filename of the new tiff file, or list of filenames.
- shape:     iterable; shape (C, Z, T) of data to be written in file.
- dtype:     string; cast data to dtype before saving, only (u)int8, (u)int16 and float32 are supported.
- colormap:  string; choose any colormap from the colorcet module.
- nP:        int; number of compressor workers to use
- extratags: dict {int tagnr: (int type, data)}, extra tags to save on every frame, will cause a crash if not used
correctly!
- pxsize:    float; set tiff tag so ImageJ can read the pixel size (in um).

- frame:     2D numpy array with data
- c, z, t:   int; channel, z, time coordinates of the frame

    
# Examples
## Write an image stack
    from tiffwrite import tiffwrite
    import numpy as np

    image = np.random.randint(0, 255, (5, 3, 64, 64), 'uint16')
    tiffwrite('file.tif', image, 'TCXY')

## Write one frame at a time
    from itertools import product
    from tiffwrite import IJTiffWriter
    import numpy as np

    shape = (3, 5, 10)  # channels, z, time
    with IJTiffWriter('file.tif', shape, pxsize=0.09707) as tif:
        for c in range(shape[0]):
            for z in range(shape[1]):
                for t in range(shape[2]):
                    tif.save(np.random.randint(0, 10, (32, 32)), c, z, t)

## Saving multiple tiffs simultaneously
    from itertools import product
    from tiffwrite import IJTiffWriter
    import numpy as np
    
    shape = (3, 5, 10)  # channels, z, time
    with IJTiffWriter(('fileA.tif', 'fileB.tif'), shape) as tif:
        for c in range(shape[0]):
            for z in range(shape[1]):
                for t in range(shape[2]):
                    tif.save('fileA.tif', np.random.randint(0, 10, (32, 32)), c, z, t)
                    tif.save('fileB.tif', np.random.randint(0, 10, (32, 32)), c, z, t)

## Tricks & tips
- ImageJ colors channels in the order rgbwcym, and IJTiffwriter automatically and efficiently writes 0's when a frame is
skipped. You can use this when specific colors are important, for example: you want to use only red and blue.
- The order of feeding frames to IJTiffWriter is unimportant, IJTiffWriter will order de ifd's such that the file will
be opened as a correctly ordered hyperstack.
- Using the colormap parameter you can make ImageJ open the file and apply the colormap. colormap='glasbey' is very
useful.

[![pytest](https://github.com/wimpomp/tiffwrite/actions/workflows/pytest.yml/badge.svg)](https://github.com/wimpomp/tiffwrite/actions/workflows/pytest.yml)

# Tiffwrite
Write BioFormats/ImageJ compatible tiffs with zstd compression in parallel using Rust.

## Features
- Writes bigtiff files that open in ImageJ as hyperstack with correct dimensions.
- Parallel compression.
- Write individual frames in random order.
- Compresses even more by referencing tag or image data which otherwise would have been saved several times.
For example empty frames, or a long string tag on every frame. Editing tiffs becomes mostly impossible, but compression
makes that very hard anyway.
- Enables memory efficient scripts by saving frames whenever they're ready to be saved, not waiting for the whole stack.
- Colormaps
- Extra tags, globally or frame dependent.

# Python
## Installation
```pip install tiffwrite```

or

- install [rust](https://rustup.rs/)
- ``` pip install tiffwrite@git+https://github.com/wimpomp/tiffwrite ```

## Usage
### Write an image stack
    tiffwrite(file, data, axes='TZCXY', dtype=None, bar=False, *args, **kwargs)

- file:         string; filename of the new tiff file.
- data:         2 to 5D numpy array in one of these datatypes: (u)int8, (u)int16, float32.
- axes:         string; order of dimensions in data, default: TZCXY for 5D, ZCXY for 4D, CXY for 3D, XY for 2D data.
- dtype:        string; cast data to dtype before saving, only (u)int8, (u)int16 and float32 are supported.
- bar:          bool; whether to show a progress bar.
- args, kwargs: arguments to be passed to IJTiffFile, see below.


### Write one frame at a time
    with IJTiffFile(file, dtype='uint16', colors=None, colormap=None, pxsize=None, deltaz=None,
                    timeinterval=None, **extratags) as tif:
    some loop:
        tif.save(frame, c, z, t)

- path:         string; path to the new tiff file.
- dtype:        string; cast data to dtype before saving, only (u)int8, (u)int16 and float32 are supported by Fiji.
- colors:       iterable of strings; one color per channel, valid colors (also html) are defined in matplotlib.colors.
                    Without colormap BioFormats will set the colors in this order: rgbwcmy.
                    Note that the color green is dark, the usual green is named 'lime' here.
- colormap:     string; choose any colormap from the colorcet module. Colors and colormap cannot be used simultaneously.
- pxsize:       float; pixel size im um.
- deltaz:       float; z slice interval in um.
- timeinterval: float; time between frames in seconds.
- compression:  int; zstd compression level: -7 to 22.
- comment:      str; comment to be saved in tif
- extratags:    Sequence\[Tag\]; other tags to be saved, example: Tag.ascii(315, 'John Doe') or Tag.ascii(33432, 'Made by me').


- frame:        2D numpy array with data.
- c, z, t:      int; channel, z, time coordinates of the frame.

    
## Examples
### Write an image stack
    from tiffwrite import tiffwrite
    import numpy as np

    image = np.random.randint(0, 255, (5, 3, 64, 64), 'uint16')
    tiffwrite('file.tif', image, 'TCXY')

### Write one frame at a time
    from tiffwrite import IJTiffFile
    import numpy as np

    with IJTiffFile('file.tif', pxsize=0.09707) as tif:
        for c in range(3):
            for z in range(5):
                for t in range(10):
                    tif.save(np.random.randint(0, 10, (32, 32)), c, z, t)

### Saving multiple tiffs simultaneously
    from tiffwrite import IJTiffFile
    import numpy as np

    with IJTiffFile('fileA.tif') as tif_a, IJTiffFile('fileB.tif') as tif_b:
        for c in range(3):
            for z in range(5):
                for t in range(10):
                    tif_a.save(np.random.randint(0, 10, (32, 32)), c, z, t)
                    tif_b.save(np.random.randint(0, 10, (32, 32)), c, z, t)


# Rust
    use ndarray::Array2;
    use tiffwrite::IJTiffFile;

    {  // f will be closed when f goes out of scope
        let mut f = IJTiffFile::new("file.tif")?;
        for c in 0..3 {
            for z in 0..5 {
                for t in 0..10 {
                    let arr = Array2::<u16>::zeros((100, 100));
                    f.save(&arr, c, z, t)?;
                }
            }
        }
    }

# Tricks & tips
- The order of feeding frames to IJTiffFile is unimportant, IJTiffFile will order the ifd's such that the file will be opened as a correctly ordered hyperstack.
- Using the colormap parameter you can make ImageJ open the file and apply the colormap. colormap='glasbey' is very useful.

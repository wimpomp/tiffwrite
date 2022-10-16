#!/usr/bin/python
import tiffwrite
import numpy as np
from itertools import product


def test():
    with tiffwrite.IJTiffWriter('test.tif', (3, 4, 5)) as tif:
        for c, z, t in product(range(3), range(4), range(5)):
            tif.save(np.random.randint(0, 255, (64, 64)), c, z, t)


if __name__ == '__main__':
    test()

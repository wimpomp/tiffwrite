import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='tiffwrite',
    version='2022.5.0',
    author='Wim Pomp @ Lenstra lab NKI',
    author_email='w.pomp@nki.nl',
    description='Parallel tiff writer compatible with ImageJ.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/wimpomp/tiffwrite',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=['tifffile', 'numpy', 'tqdm', 'colorcet', 'matplotlib'],
)

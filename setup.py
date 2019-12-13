from setuptools import setup
import os
import re
import codecs

# Create new package with python setup.py sdist

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='cinderella',
    version=find_version("Auto2DSelect", "__init__.py"),
    python_requires='>3.4.0',
    packages=['Auto2DSelect'],
    url='https://github.com/MPI-Dortmund/sphire_classes_ausptoselect',
    license='MIT',
    author='Thorsten Wagner',
    extras_require = {
        'gpu':  ['tensorflow-gpu == 1.10.1'],
        'cpu': ['tensorflow == 1.10.1']
    },
    install_requires=[
        "Keras == 2.2.5",
        "numpy == 1.14.5",
        "h5py >= 2.5.0",
        "pillow",
        "tqdm",
        "mrcfile"
    ],
    author_email='thorsten.wagner@mpi-dortmund.mpg.de',
    description='Select 2d classes automatically',
    entry_points={
        'console_scripts': [
            'sp_cinderella_train.py = Auto2DSelect.train:_main_',
            'sp_cinderella_predict.py = Auto2DSelect.predict:_main_',
            'sp_cinderella_extract.py = Auto2DSelect.extract:_main_',
        ]},
)

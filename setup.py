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
    name='auto2dclassselect',
    version=find_version("Auto2DSelect", "__init__.py"),
    python_requires='>3.4.0',
    packages=['Auto2DSelect'],
    url='',
    license='MIT',
    author='Thorsten Wagner',
    install_requires=[
        "mrcfile >= 1.0.0",
        "Keras >= 2.2.4",
        "numpy <= 1.14.5",
        "h5py >= 2.5.0",
        "tensorflow-gpu == 1.10.1",
        "pillow",
        "tqdm"
    ],
    author_email='thorsten.wagner@mpi-dortmund.mpg.de',
    description='Select 2d classes automatically',
    entry_points={
        'console_scripts': [
            'sp_auto2d_train.py = Auto2DSelect.train:_main_',
            'sp_auto2d_predict.py = Auto2DSelect.predict:_main_']},
)

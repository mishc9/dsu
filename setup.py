from distutils.core import setup
from pathlib import Path

import setuptools

dpath = Path(__file__).parent.absolute()

try:
    with open(dpath / 'requirements.txt') as f:
        # read recipe at https://www.reddit.com/r/Python/comments/3uzl2a/setuppy_requirementstxt_or_a_combination/
        requirements = f.read().splitlines()
except FileNotFoundError:
    requirements = list()

setup(name='dsu',
      version='0.0.2',
      url='jet.msk.su',
      license='Apache 2',
      author='Mikhail Stepanov',
      author_email='mishc9@gmail.com',
      description='frequently required utils for ds purposes',
      packages=setuptools.find_packages(),
      # dependency_links=["git+https://github.com/mishc9/addata.git@master#egg=addata-0"],
      include_package_data=True,
      python_requires='>=3.6.5',
      # install_requires=requirements + ["addata @ https://github.com/mishc9/addata/tarball/master#egg=addata-0.0.2"],
      )

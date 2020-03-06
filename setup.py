#!/usr/bin/env python
#
# Copyright (C) 2020  Momoko Hayamizu <hayamizu@ism.ac.jp>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

import os
import re
import sys

from setuptools import setup, find_packages

with open('treefit/__init__.py') as init_py:
    version = re.search('__version__ = \'(.*?)\'',
                        init_py.read())[1]

with open('README.md') as readme:
    long_description = readme.read()

setup(name='treefit',
      version=version,
      packages=find_packages(),
      description='The first software for quantitative trajectory inference',
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=[
          'matplotlib',
          'numpy',
          'pandas',
          'scipy',
          'sklearn',
      ],
      tests_require=[
          'pytest',
      ],
      author='Momoko Hayamizu',
      author_email='hayamizu@ism.ac.jp',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Visualization',
      ],
      license="LGPLv3+",
      url='https://hayamizu-lab.github.io/treefit-python/',
)

import os

import numpy as np

from setuptools import setup


descr = """Free support Sinkhorn Barycenters via Frank-Wolf's algorithm"""

version = None
with open(os.path.join('otbar', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'otbar'
DESCRIPTION = descr
MAINTAINER = ''
MAINTAINER_EMAIL = ''
LICENSE = ''
DOWNLOAD_URL = ''
VERSION = version
URL = ''

setup(name='otbar',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open('README.md').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      packages=['otbar'],
      include_dirs=[np.get_include()],
      )

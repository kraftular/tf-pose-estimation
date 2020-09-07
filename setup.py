from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import setuptools
from distutils.core import setup, Extension

import numpy as np

_VERSION = '0.1.1'

cwd = os.path.dirname(os.path.abspath(__file__))

POSE_DIR = os.path.realpath(os.path.dirname(__file__))

REQUIRED_PACKAGES = [
    'argparse>=1.1',
    'dill==0.2.7.1',
    'matplotlib >= 2.2.2',
    'scipy >= 1.1.0'
]

EXT = Extension('_pafprocess',
                sources=[
                    'tf2_pose/pafprocess/pafprocess_wrap.cpp',
                    'tf2_pose/pafprocess/pafprocess.cpp',
                ],
                swig_opts=['-c++'],
                include_dirs=[np.get_include()])

setuptools.setup(
    name='tf2-pose',
    version=_VERSION,
    description=
    'Deep Pose Estimation for Tensorflow 2. Inference only.',
    install_requires=REQUIRED_PACKAGES,
    url='https://github.com/kraftular/tf2-pose-estimation/',
    author='Ildoo Kim, Adam Kraft',
    author_email='(ildoo@ildoo.net), adk@mit.edu',
    license='Apache License 2.0',
    package_dir={'tf2_pose_data': 'models'},
    packages=['tf2_pose_data'] +
             [pkg_name for pkg_name in setuptools.find_packages()  # main package
              if 'tf2_pose' in pkg_name],
    ext_modules=[EXT],
    package_data={'tf2_pose_data': ['graph/mobilenet_thin/graph_opt.pb',
                                    'graph/mobilenet_v2_large/graph_opt.pb',
                                    'graph/mobilenet_v2_small/graph_opt.pb'
    ]},
    py_modules=[
        "pafprocess"
    ],
    zip_safe=False)

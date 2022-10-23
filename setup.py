#!/usr/bin/env python

import os

import pkg_resources
from distutils.core import setup

def is_package(path):
    return (
        os.path.isdir(path) and
        os.path.isfile(os.path.join(path, '__init__.py'))
        )

def find_packages(path, base="" ):
    """ Find all packages in path """
    packages = {}
    for item in os.listdir(path):
        dir = os.path.join(path, item)
        if is_package( dir ):
            if base:
                module_name = "%(base)s.%(item)s" % vars()
            else:
                module_name = item
            packages[module_name] = dir
            packages.update(find_packages(dir, module_name))
    return packages

setup(name='diffae',
      version='1.0',
      description='Diffusion Autoencoder',
      author='Oleguer Canal',
      author_email='oleguer.canal@hotmail.com',
      url='https://github.com/OleguerCanal/diffusion_autoencoder/',
      packages=find_packages("."),
      install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
      ],
    )
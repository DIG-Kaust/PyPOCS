import os
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = 'POCS seismic interpolation with different proximal algorithms (HQS, ADMM, PD).'

setup(
    name="pypocs", # Choose your package name
    description=descr,
    long_description=open(src('README.md')).read(),
    keywords=['inverse problems',
              'seismic',
              'interpolation'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    author='Matteo Ravasi',
    author_email='matteo.ravasi@kaust.edu.sa',
    install_requires=['numpy >= 1.15.0',
                      'scipy >= 1.4.0',
                      'pylops >= 2.0.0',
                      'pyproximal >= 0.6.0'],
    packages=find_packages(),
    use_scm_version=dict(root='.',
                         relative_to=__file__,
                         write_to=src('pypocs/version.py')),
    setup_requires=['setuptools_scm'],

)
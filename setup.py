#!/usr/bin/env python
"""Setup script to install skymap."""

from distutils.core import setup
import json

with open('wfiskymatch/version.json') as fp:
    _info = json.load(fp)

with open("README.md", "r") as fh:
    long_description = fh.read()

config = {
        'name':'wfiskymatch',
        'version':_info['version'],
        'description':'Skymatch Algorithm for Roman WFI',
        'long_description':long_description,
        'long_description_type':"text/markdown",
        'author':_info['author'],
        'author_email':'dfadda@stsci.edu',
        'url':'https://github.com/wfiskymatch',
        'download_url':'git@github.com:darioflute/wfiskymatch.git',
        'python_requires':'>=3.13',
        'license':'GPLv3+',
        'packages':['wfiskymatch'],
        'include_package_data':True,
        'package_data':{'wfiskymatch':['copyright.txt','version.json']},
        'classifiers':[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: GPLv3+ License",
                "Operating System :: OS Independent",
                "Intended Audience :: Science/Research", 
                "Development Status :: Production",
                "Topic :: Scientific/Engineering",
                ],
     }

setup(**config)

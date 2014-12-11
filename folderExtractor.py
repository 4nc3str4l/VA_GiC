# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 00:55:35 2014

@author: ancestral
"""

import os
import shutil

for dirname, dirnames, filenames in os.walk('.'):
    # print path to all subdirectories first.
    for subdirname in dirnames:
        print os.path.join(dirname, subdirname)

    # print path to all filenames.
    for filename in filenames:
        print os.path.join(dirname, filename)
        
        if "folderExtractor.py" != filename: 
            shutil.move(os.path.join(dirname,filename),'..')

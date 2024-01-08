# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 21:50:27 2021

@author: davehanzhang
"""
from distutils.dir_util import copy_tree
import shutil
import utilities.misc as um
import os

copy_from="Z:\\dave\\python"
copy_to_1="Y:\\Dave Yin\\code_backup"
copy_to_2="M:\\code_backup"


### delete existing folders
try:
    shutil.rmtree(copy_to_1)
except FileNotFoundError:
    print ('path 1 no longer exist')
try:
    shutil.rmtree(copy_to_2)
except FileNotFoundError:
    print ('path 2 no longer exist')

### create new folders
os.makedirs(copy_to_1)
os.makedirs(copy_to_2)



copy_tree(copy_from, copy_to_1)
copy_tree(copy_from, copy_to_2)


um.quick_auto_notice(msg='code back up successful')




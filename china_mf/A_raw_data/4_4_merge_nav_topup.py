# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 18:52:17 2021

@author: hyin1
"""

from webscraping.eastmoney import TopUp_update_nav,get_current_nav_status
import utilities.misc as um
import numpy as np
import pandas as pd
import utilities.constants as uc

'''
we will update nav status after the top-up update
'''

nav=TopUp_update_nav()
get_current_nav_status()







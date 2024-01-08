# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 17:24:36 2021

@author: hyin1
"""

from webscraping.eastmoney import load_nav,get_current_nav_status
import utilities.misc as um
import numpy as np
import pandas as pd
import utilities.constants as uc

'''
Before doing the topup update launch for fund nav
make sure the aggregate feather contains the most recent update
and the fund_details folder has no individual nav_update_n file.

run "load_fund_holdings(build_from_dump=True)" below if nav.feather is not created
otherwise no need to run that line

but need to run get_current_nav_status everytime
'''


#load_nav(build_from_dump=True)
get_current_nav_status()










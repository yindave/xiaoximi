# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 15:05:35 2021

@author: hyin1
"""

from webscraping.eastmoney import load_fund_holdings,TopUp_update_holdings
import utilities.misc as um
import numpy as np
import pandas as pd
import utilities.constants as uc

'''
run the one-liner below, this will update existing hlds with the individual top-up dumps
    - so here we need to make sure there is no gap between existing hlds and the top-up dumps
    - top-up update covers current year and last year (can expand if needed)
'''


hlds=TopUp_update_holdings(make_up_mode=False) # turnoff make up mode here













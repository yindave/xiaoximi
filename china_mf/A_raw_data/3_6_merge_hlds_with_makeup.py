# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 15:55:01 2021

@author: hyin1
"""

from webscraping.eastmoney import load_fund_holdings,TopUp_update_holdings
import utilities.misc as um
import numpy as np
import pandas as pd
import utilities.constants as uc

'''
run the one-liner below, this will update existing hlds with the make-up data

IMPORTANT:
after running the below one-liner, repeat 3_4_check_hlds_quality_1 to make sure there is no discontinued holdings
(note it usually takes more than 1 round to finish the data make up process)
'''

hlds=TopUp_update_holdings(make_up_mode=True)








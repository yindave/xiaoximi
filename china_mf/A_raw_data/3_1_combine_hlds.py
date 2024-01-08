# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:14:59 2021

@author: hyin1
"""

from webscraping.eastmoney import load_fund_holdings,TopUp_update_holdings
import utilities.misc as um
import numpy as np
import pandas as pd
import utilities.constants as uc

'''
Before doing the topup update launch for equity holdings
make sure the aggregate feather contains the most recent update
and the fund_details folder has no individual hlds_update_n file.

run "load_fund_holdings(build_from_dump=True)" below if hlds.feather is not created
otherwise no need to run this script


'''

# step 1: just run it once initially
#load_fund_holdings(build_from_dump=True)


















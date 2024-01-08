# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 18:06:30 2021

@author: hyin1
"""


from webscraping.eastmoney import get_fund_list,update_all_fund_info,path_static_list
import utilities.misc as um
import pandas as pd
import utilities.constants as uc

'''
This one is on scheduler as well.
Check if daily scripts failed. EM website sometimes changes/adds fund type label

'''
get_fund_list(use_static=False).to_csv(path_static_list)



















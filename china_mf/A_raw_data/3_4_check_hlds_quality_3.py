# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 15:11:28 2021

@author: hyin1
"""

import pandas as pd
import matplotlib.pyplot as plt
import utilities.misc as um
import utilities.display as ud
import numpy as np
import utilities.constants as uc
import utilities.mathematics as umath

import webscraping.eastmoney as em


'''
collect and dump the funds to re-run
'''

dump_path=em.path_static_list.replace('static_list.csv','hlds_check_dump\\') # need to empty the folder before a new run
check_since=pd.datetime(2001,1,1) # can set this date to more recent one once we are confident about the historical data quality


files=um.iterate_csv(dump_path)
collector=[]
for file in files:
    collector.append(pd.read_csv(dump_path+'%s.csv' % (file)).set_index('ticker_fund')[['is_incomplete_download']])
to_make_up=pd.concat(collector,axis=0)['is_incomplete_download']
if len(to_make_up[to_make_up==-1])!=0:
    raise NameError('Need run step 2 until no -1 in the is_incomplete_download column')


make_up_list=to_make_up[to_make_up==1].to_frame()
make_up_list.index.name='ticker'

make_up_list.to_csv(em.path_static_list.replace('static_list','static_list_hlds_makeup'))


if len(make_up_list[make_up_list==1])==0:
    print ('All fund holdings are clean now!')







# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:08:11 2021

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
import pdb


'''
We will continue running the below script until there is no -1 in the is_incomplete_download col
    - this is because the em.get_fund_equity_holdings may disconnect
'''

dump_path=em.path_static_list.replace('static_list.csv','hlds_check_dump\\') # need to empty the folder before a new run


files=um.iterate_csv(dump_path)
for file in files:
    print ('checking file %s' % (file))
    to_check=pd.read_csv(dump_path+'%s.csv' % (file)).set_index('ticker_fund')
    check_date_last=pd.to_datetime(to_check.columns[1])
    try:
        for i,ticker_i in enumerate(to_check.index):
            if to_check.loc[ticker_i]['is_incomplete_download']==-1:
                check_i=em.get_fund_equity_holdings(ticker_i,customize_years=[True,[check_date_last.year-1,check_date_last.year]])
                check_i['asof']=pd.to_datetime(check_i['asof'])
                if check_date_last in check_i.groupby('asof').count().index:
                    to_check.at[ticker_i,'is_incomplete_download']=1
                else:
                    to_check.at[ticker_i,'is_incomplete_download']=0
                print ('%s/%s done' % (i+1,len(to_check.index)))
            else:
                print ('%s done already' % (ticker_i))
    except:
        to_check.to_csv(dump_path+'%s.csv' % (file))
    to_check.to_csv(dump_path+'%s.csv' % (file))







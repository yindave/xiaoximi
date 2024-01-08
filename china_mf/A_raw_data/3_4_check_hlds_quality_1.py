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

'''
check hlds data quality
    - it's possible we may have discontinued hld series for individual funds due to the downloading process
    - so we need to first check for this and then run "make-up" script

when running a new update, empty the hlds_check_dump folder before running the scirpts below

After runnning this script, check the hlds_check_dump folder, if any file contains too many tickers, that
means too many tickers are skipped in the 3_2_get_fund_hlds_topup run. Reduce the job_N and re-run, and re-do
the merge fund hlds as well
'''

dump_path=em.path_static_list.replace('static_list.csv','hlds_check_dump\\') # need to empty the folder before a new run
check_since=pd.datetime(2001,1,1) # can set this date to more recent one once we are confident about the historical data quality

### part 1: dump the dates to check. run this with part 2/3 hashed

hlds_check=em.load_fund_holdings()

dates=pd.date_range(hlds_check['asof'].min(),hlds_check['asof'].max(),freq='Q')

collector=[]
for i,date in enumerate(dates):
    if i>0 and date>=check_since:
        check_date_previous=dates[i-1]
        check_date_last=dates[i]
        check=hlds_check.groupby(['asof','ticker_fund']).count()['ticker'].unstack().loc[check_date_previous:check_date_last].fillna(0).T
        to_check=check[(check[check_date_last]==0) & (check[check_date_previous]!=0)]
        to_check['is_incomplete_download']=-1
        if len(to_check)!=0:
            collector.append(to_check)
        print ('finish %s' % (date))


for i,df in enumerate(collector):
    df.to_csv(dump_path+'%s.csv' % (i))






# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:04:59 2021

@author: hyin1
"""

import pandas as pd
import feather
import webscraping.eastmoney as em

'''
Run this when there is new update on holdings
after the month of 1,3,4,7,8,10
'''

print ('working on A_a_get_holding_to_use')

def _get_rpt_date(x,mode='top10'):
    if x.month==3: #Q1
        return pd.datetime(x.year,4,30)
    elif x.month==9: #Q3
        return pd.datetime(x.year,10,31)
    elif x.month==12: #Q4 or annual
        if mode=='top10': #Q4
            return pd.datetime(x.year+1,1,31)
        else: #annual
            return pd.datetime(x.year+1,3,31) # we ignore the 2019 annual reporting delay here.
    elif x.month==6: #Q2 or semi
        if mode=='top10': #Q2
            return pd.datetime(x.year,7,31)
        else:
            return pd.datetime(x.year,8,31)


path=em.path_static_list.replace('static_list.csv','')
hlds=feather.read_dataframe(path+'TidyUp_%s.feather' % ('hlds'))
hlds['top_n']=hlds.groupby(['ticker_fund','asof'])['values'].rank(ascending=False)

hlds_to_use=hlds.copy()
hlds_to_use['top_n']=hlds_to_use.apply(lambda x: 0 if x['IsTop10HolderButNotTop10Holding'] else x['top_n'], axis=1)
hlds_to_use['mode']=hlds_to_use['top_n'].map(lambda x: 'top10' if x<=10 else 'rest')
hlds_to_use['date']=hlds_to_use.apply(lambda x: _get_rpt_date(x['asof'],mode=x['mode']),axis=1) # this will be the earliest date we get the holding information
hlds_to_use=hlds_to_use[['date','ticker','ticker_fund','shares','top_n','asof','values']]

#dump the file for launcher to load
feather.write_dataframe(hlds_to_use,path+'process_data\\hlds_to_use.feather')

print ('finish A_a_get_holding_to_use')










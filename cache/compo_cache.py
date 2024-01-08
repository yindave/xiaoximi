# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:15:29 2020

@author: hyin1
"""

import pandas as pd
import numpy as np
import utilities.misc as um
from blp.bdx import bdh, bdp, bds
from blp.util import get_bbg_nice_compo_hist_no_limit
from fql.util import bbg_to_fs, fs_to_bbg
import os
import feather
import utilities.constants as uc
from datetime import datetime
import utilities.mathematics as umath

path=uc.compo_path

# ticker and since date
idx_dict={
        'SHSZ300 Index':datetime(2006,12,31),
        'SH000905 Index':datetime(2012,12,31),
                    #'MXCN Index':datetime(2014,12,31),
                   # 'MXCN1A Index':datetime(2007,12,31),
                    #'MBCN1A Index':datetime(2012,12,31),
        'XIN9I Index':datetime(2005,10,31),
        'HSCEI Index':datetime(2005,10,31),
        'SZ399006 Index':datetime(2010,12,31), # ChiNext 100 stock
        'SZ399102 Index':datetime(2014,12,31), # ChiNext all stock Composite
          'SSE50 Index':datetime(2014,12,31), # SH large
          'SZ399005 Index':datetime(2014,12,31),  # SZ SME
          'SSE180 Index':datetime(2014,12,31), # NB SH
          'SH000009 Index':datetime(2014,12,31), # NB SH
            'SICOM Index':datetime(2014,12,31),# NB SZ
            'SZ399015 Index':datetime(2016,10,28), # NB SZ
            'HSCI Index':datetime(2005,12,31),
            'HSSI Index':datetime(2014,1,1),
            'HSMI Index':datetime(2014,1,1),
            'HSLI Index':datetime(2014,1,1),
                    #  'MXHK Index':datetime(2014,1,1),
            'TPX Index': datetime(2005,12,31),
            'KOSPI Index': datetime(2009,12,31),
            'AS51 Index': datetime(2009,12,31),
            'TWSE Index': datetime(2009,12,31),
                        #'TAMSCI Index': datetime(2009,12,31),
                        #'MXEU Index': datetime(2009,12,31),
                          #'RAY Index': datetime(2009,12,31),
                          #'STI Index': datetime(2008,12,31),
                          #'MXIN Index': datetime(2008,12,31),
                          #'MXID Index': datetime(2008,12,31),
                        # 'MXAP Index': datetime(2008,12,31),
                        #'MXAPJ Index': datetime(2008,12,31),
                        #'MXJP Index': datetime(2009,12,31),
                        # 'MXEF0HC Index': datetime(2006,12,31),
                        # 'MXIN0HC Index': datetime(2006,12,31),
            'STAR50 Index':datetime(2020,7,1),
            'HSI Index':datetime(2010,12,31),
            'XIN0I Index':datetime(2006,12,31),
            'TPX500 Index':datetime(2010,12,31),
            'SPX Index': datetime(2009,12,31),
                            #'MXASJ Index':datetime(2009,12,31),
            'SHCOMP Index': datetime(2005,12,31),
            'SZCOMP Index': datetime(2005,12,31),
                          #'MXEF Index':datetime(2009,12,31),
                          #'MXWO Index':datetime(2009,12,31),
            'NSE500 Index':datetime(2009,12,31)
        }




for idx,start_date in idx_dict.items():
    # try to load existing dump
    try:
        old_compo=feather.read_dataframe(path % (idx))
        start_date=old_compo['asof'].max()
    except OSError:
        print ('No existing dump found for %s' % (idx))
        old_compo=pd.DataFrame()
    collector=[]
    if idx not in ['MXAP Index','RAY Index','MXAPJ Index','TPX500 Index',
                   'SPX Index','MXASJ Index','SHCOMP Index','SZCOMP Index',
                   'MXWO Index','MXEF Index','TPX Index'
                   ]:
        dates=pd.date_range(start_date,um.today_date(),freq='BM')
    else:
        dates=pd.date_range(start_date,um.today_date(),freq='A')
    for date in dates:
        compo_i=get_bbg_nice_compo_hist_no_limit(idx,date)
        collector.append(compo_i)
        print ('finish %s on %s' % (idx,date))
    compo_new=pd.concat(collector,axis=0)
    compo_new['idx']=idx
    compo_new=compo_new.reset_index()
    compo=pd.concat([old_compo,compo_new],axis=0)
    compo=compo.groupby(['ticker','asof']).last().reset_index()
        
    feather.write_dataframe(compo,path % (idx))

um.quick_auto_notice('Index compo update finished')

# create a quick CSI300+CSI500 compo
csi800=pd.concat([
        feather.read_dataframe(path % ('SHSZ300 Index')),
        feather.read_dataframe(path % ('SH000905 Index')),
        ],axis=0)
feather.write_dataframe(csi800,path % ('CSI800 Index'))

# create lite version of SHCOMP, SZCOMP, TPX (top 95% wgt coverage point in time)
# fewer stocks make is easier to maintain alpha model
tickers=['SHCOMP Index','SZCOMP Index','TPX Index']
for ticker in tickers:
    compo=feather.read_dataframe(path % (ticker)).set_index('asof')
    compo['cutoff_level']=compo.reset_index().groupby('asof')['wgt'].apply(lambda x: umath.get_cutoff_level_for_cumu_coverage(x, 0.95)[0])
    compo=compo[compo['wgt']>=compo['cutoff_level']].reset_index().drop('cutoff_level',1)
    compo['wgt']=compo.groupby('asof')['wgt'].apply(lambda x: x/x.sum())
    feather.write_dataframe(compo, path % (ticker.replace(' Index','_L Index')))


































































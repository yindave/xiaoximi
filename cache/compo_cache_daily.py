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

path=uc.compo_path_daily

# ticker and since date
idx_dict={
        'SHSZ300 Index':datetime(2006,12,31),
        'XIN9I Index':datetime(2005,10,31),
        'HSCEI Index':datetime(2005,10,31),
        # no more MSCI China
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
                   'MXWO Index','MXEF Index'
                   ]:
        dates=pd.date_range(start_date,um.today_date(),freq='B')
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

um.quick_auto_notice('Index compo update finished (daily freq)')





































































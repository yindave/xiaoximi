# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:17:36 2021

@author: hyin1
"""

import pandas as pd
import numpy as np
import utilities.misc as um
import utilities.constants as uc
from blp.bdx import bdh, bdp, bds
from blp.util import get_bbg_nice_compo_hist_no_limit,get_bbg_nice_compo_hist
from fql.util import bbg_to_fs, fs_to_bbg
import os
import feather
import pdb

from ah_arb.ah import AH

ah=AH()
dump_path=ah.path+'PROD\\IndexSwitch\\'

idx_all=['MXCN','HSCEI','XIN9I','SHSZ300']
collector=[]
for idx in idx_all:
    compo_i=feather.read_dataframe(uc.compo_daily_path % (idx+' Index'))
    collector.append(compo_i)
compo_all=pd.concat(collector,axis=0)


tickers=compo_all.groupby('ticker').last().index.tolist()

# need to add the AH pairs as well!
ah_list=ah.ah_list
all_ah_pairs=fs_to_bbg(ah_list['HK'])+fs_to_bbg(ah_list['CN'])

tickers=tickers+all_ah_pairs
tickers=list(set(tickers))

# we use BBG to download the data
px_info=bdh(tickers,['px_last','EQY_WEIGHTED_AVG_PX'],pd.datetime(2009,12,31),um.today_date(),
            currency='USD')
feather.write_dataframe(px_info.reset_index(),dump_path+'px_info.feather')


# we use BBG to download the data
px_info=bdh(tickers,['px_last','EQY_WEIGHTED_AVG_PX','turnover'],pd.datetime(2009,12,31),um.today_date(),
            currency='USD',adjustmentFollowDPDF='No')
feather.write_dataframe(px_info.reset_index(),dump_path+'px_info_unadj.feather')
#
#




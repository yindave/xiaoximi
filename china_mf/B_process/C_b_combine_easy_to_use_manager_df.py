# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:44:47 2021

@author: hyin1
"""

from multiprocessing import Process
import webscraping.eastmoney as em
import utilities.misc as um
import pandas as pd
import utilities.constants as uc
import feather
import numpy as np
import pdb

'''
Daily
'''

print ('working on C_b_combine_easy_to_use_manager_df')

path=em.path_static_list.replace('static_list.csv','')
dump_path=path+'\\process_data\\dump_manager\\'

collector=[]
files=um.iterate_csv(dump_path, iterate_others=[True, '.feather'])
for file in files:
    collector.append(feather.read_dataframe(dump_path+'%s.feather' % (file)))
manager_easy_to_use=pd.concat(collector)
manager_easy_to_use=manager_easy_to_use.rename(columns={'exp':'manager_exp','return':'manager_return','sharpe':'manager_sharpe','fund_count':'manager_fund_count'})


feather.write_dataframe(manager_easy_to_use,path+'\\process_data\\managers_easy_to_use.feather')


print ('finished on C_b_combine_easy_to_use_manager_df')




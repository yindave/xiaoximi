# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:07:29 2021

@author: hyin1
"""

from webscraping.eastmoney import update_all_fund_info,path_static_list
import utilities.misc as um
import numpy as np
import pandas as pd
import utilities.constants as uc

from multiprocessing import Process

'''
Fund snapshot info

Takes time to run so we only run this once
Future update will be top-up style
'''

data_type='hlds'
job_N=20

finished=[]

if __name__ == "__main__":
    # parallel run
    p_collector=[]
    for job_i in np.arange(0,job_N,1):
        if job_i not in finished:
            p=Process(target=update_all_fund_info,args=(data_type,job_N,job_i,))
            p_collector.append(p)
            p.start()
    for p in p_collector:
        p.join()
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:50:11 2021

@author: hyin1
"""

from webscraping.eastmoney import update_all_fund_info,path_static_list
import utilities.misc as um
import numpy as np
import pandas as pd
import utilities.constants as uc

from multiprocessing import Process

'''
make sure static_list_hlds_makeup.csv is the latest
'''


data_type='hlds_MakeUp'
job_N=2

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


















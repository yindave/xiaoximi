# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 09:11:53 2021

@author: hyin1
"""

from webscraping.eastmoney import update_all_fund_info,path_static_list
import utilities.misc as um
import numpy as np
import pandas as pd
import utilities.constants as uc

from multiprocessing import Process

'''
Fund filing record

We don't really need to update this one for each refresh
Filing records are used just one time to see the distribution of the actual reporting date
'''

data_type='filings'
job_N=30

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
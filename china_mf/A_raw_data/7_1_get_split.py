# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 09:19:52 2021

@author: hyin1
"""


from webscraping.eastmoney import update_all_fund_info,path_static_list
import utilities.misc as um
import numpy as np
import pandas as pd
import utilities.constants as uc

from multiprocessing import Process

'''
Fund holder type split

This one only updates in the semi-annual and annual report
'''

data_type='split'
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
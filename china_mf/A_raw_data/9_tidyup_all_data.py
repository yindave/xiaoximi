# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:39:36 2021

@author: hyin1
"""

from webscraping.eastmoney import tidy_up_all_data_quick,path_static_list,load_fund_specs,load_split,load_filings,load_bio,load_share_out



load_fund_specs(build_from_dump=True)
load_split(build_from_dump=True)
load_filings(build_from_dump=True)
load_bio(build_from_dump=True)
load_share_out(build_from_dump=True)


tidy_up_all_data_quick(quick_path=[True,path_static_list.replace('static_list.csv','')])












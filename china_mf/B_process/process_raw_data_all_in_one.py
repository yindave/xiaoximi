# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:33:41 2021

@author: hyin1
"""

import subprocess

'''
We can run this weekly
'''


from china_mutual_fund.B_process_raw_data import A_a_get_holding_to_use

print ('calling multi processing')
subprocess.call(['python.exe', 'Z:\\dave\\python\\china_mutual_fund\\B_process_raw_data\\A_b_dump_filled_holdings.py'])
print ('finish multi processing')


from china_mutual_fund.B_process_raw_data import A_c_get_unadj_px_and_shout
from china_mutual_fund.B_process_raw_data import A_d_process_filled_holdings

print ('calling multi processing')
subprocess.call(['python.exe', 'Z:\\dave\\python\\china_mutual_fund\\B_process_raw_data\\B_a_dump_network_stats.py'])
print ('finish multi processing')


print ('calling multi processing')
subprocess.call(['python.exe', 'Z:\\dave\\python\\china_mutual_fund\\B_process_raw_data\\C_a_dump_easy_to_use_manager_df.py'])
print ('finish multi processing')


from china_mutual_fund.B_process_raw_data import C_b_combine_easy_to_use_manager_df












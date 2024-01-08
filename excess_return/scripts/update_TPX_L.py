# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 08:31:48 2021

@author: davehanzhang
"""

from excess_return.excess_return import Alpha_Model,Carhart
from datetime import datetime
import utilities.misc as um


# speciify universe parameters
universe=['TPX_L']
fx='JPY'
force_start_date=[True,datetime(2005,12,31)] 

def run():
    # run the script
    alpha_models={'alpha':Alpha_Model,
                  'carhart':Carhart,}
    for k,Alpha_Model_To_Use in alpha_models.items():
        print ('Excess return model is: %s' % (k))
        alpha=Alpha_Model_To_Use(universe,fx,force_start_date=force_start_date)
        # udpate universe
        alpha.update_factor_universe()
        
        # no JP strategy universe for now

    
    um.quick_auto_notice('Excess return model for %s finished' % ('_'.join(universe)))
    

um.retry_func(run,error_msg_email='Excess return model for %s failed' % ('_'.join(universe)))
    









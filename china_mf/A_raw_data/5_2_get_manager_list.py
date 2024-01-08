# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 09:04:38 2021

@author: hyin1
"""

from webscraping.eastmoney import get_manager_list
import utilities.misc as um
import numpy as np
import pandas as pd
import utilities.constants as uc

from multiprocessing import Process


get_manager_list(build_from_dump=True)






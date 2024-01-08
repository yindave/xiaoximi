# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 18:13:36 2021

@author: hyin1
"""

from multiprocessing import Process
import webscraping.eastmoney as em
import utilities.misc as um
import pandas as pd
import utilities.constants as uc
import feather
import argparse #optparse deprecated
import numpy as np
import pdb
import networkx as nx
from networkx.algorithms.bipartite.projection import projected_graph
from networkx.classes.function import density as nw_nd
from networkx import eigenvector_centrality as nw_ec

'''
Can do monthly. This will impact the strategy universe
'''


path=em.path_static_list.replace('static_list.csv','')+'process_data\\'

max_wgt=0.2 # data quality issue or mix fund with significant bond tilting
top_n=20 # we want to cover the analysis to more stocks

rebuilt=feather.read_dataframe(path+'rebuilt_results.feather')
rebuilt=rebuilt[(rebuilt['wgt']<=max_wgt) & (rebuilt['wgt']!=0)].copy()
rebuilt_for_dates=rebuilt.groupby('date').last()
dump_path=path+'dump_network\\'


print ('data block loaded')

def run(batch,batch_total):
    all_dates=rebuilt_for_dates.iloc[batch:].iloc[::batch_total].index
    by_method=[
                ['ticker','ticker_fund'],
                ]
    density_collector=[]
    centrality_collector=[]
    for date in all_dates:
        print('working on %s' % (date))
        rebuilt_i=rebuilt[(rebuilt['date']==date)].copy()
        rebuilt_i['wgt_rank']=rebuilt_i.groupby(['ticker_fund'])['wgt'].rank(ascending=False,method='min')
        rebuilt_i=rebuilt_i[rebuilt_i['wgt_rank']<=top_n]
        for bys in by_method:
            graph_input=rebuilt_i.groupby(bys)['wgt'].last().reset_index()
            B = nx.Graph()
            all_edges=graph_input[bys].apply(lambda x: tuple(x.values),axis=1).values.tolist()
            B.add_edges_from(all_edges)
            M_dict={}
            for i,by_i in enumerate(bys):
                nodes_i=graph_input.groupby(by_i).last().index
                B.add_nodes_from(nodes_i, bipartite=i)
                M_dict[by_i]=projected_graph(B, nodes_i)
            for by_i in bys:
                density_i=pd.Series(index=[date],data=[nw_nd(M_dict[by_i])]).rename('density').to_frame()
                density_i['by']=by_i
                density_i['bipartite']='-'.join(bys)
                density_i.index.name='date'
                centrality_i=pd.Series(nw_ec(M_dict[by_i])).rename('centrality').to_frame()
                centrality_i.index.name='id'
                centrality_i['id_type']=by_i
                centrality_i['bipartite']='-'.join(bys)
                centrality_i['date']=date
                density_collector.append(density_i)
                centrality_collector.append(centrality_i)
    density=pd.concat(density_collector).reset_index()
    centrality=pd.concat(centrality_collector).reset_index()
    feather.write_dataframe(density,dump_path+'density_%s.feather' % (batch))
    centrality['id']=centrality['id'].map(str)
    feather.write_dataframe(centrality,dump_path+'centrality_%s.feather' % (batch))

if __name__ == "__main__":

    # parallel run
    job_N=5
    p_collector=[]
    for job_i in np.arange(0,job_N,1):
        p=Process(target=run,args=(job_i,job_N,))
        p_collector.append(p)
        p.start()
    for p in p_collector:
        p.join()














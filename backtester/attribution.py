# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:11:33 2020

@author: hyin1
"""


import pandas as pd
import numpy as np
import utilities.misc as um
import utilities.display as ud
import utilities.constants as uc
import utilities.mathematics as umath
import pdb
import feather

from blp.bdx import bdh,bdp,bds
from fql.fql import Factset_Query
from fql.util import bbg_to_fs, fs_to_bbg,fql_date
from blp.util import get_bbg_usual_col, group_marcap
import os

from utilities.mathematics import diag,inv
from numpy import sqrt

import time

import plotly.express as px

'''
No longer need to use this script
Remember the attribution/risk calculation for default method

Key style factor includes:
    value
    leverage
    growth
    profitability
    dividend yield
    size
    liquidity
    market sensitivity
    volatility
    momentum
    fx sensitivity



'''

class ATTRIBUTION():
    PATH_BASE="Z:\\Axioma\\%s\\%s\\%s\\%s.%s.%s"
    def __init__(self,name,region,model,verbose=True,
                 MASTER_PATH="Z:\\dave\\data\\attribution\\"):
        self.MASTER_PATH=MASTER_PATH
        self.path=self.MASTER_PATH+name+'\\'
        self.region=region
        self.model=model
        self.verbose=verbose
        self.factor_names=pd.read_csv("Z:\\Axioma\\factor_names.csv")
        self.risk_factor=pd.read_csv("Z:\\Axioma\\risk_factors.csv")
        risk_factor_short=self.risk_factor.groupby('Description')['Factor_type'].last()
        self.fx_factor=risk_factor_short[risk_factor_short=='CURRENCY'].index
        self.country_factor=risk_factor_short[risk_factor_short=='COUNTRY'].index
        return None
    def load(self,dump_sedol_hist=True,
             strategy_name='port'):
        '''
        Here we load port, mkt (shold be from backtest otherwise create your own mkt dump), and sedol_hist
        We also need a dump folder for multi-processing
        For port ticker, use sedol for regional, and use fs for China/JP
        Need to make sure mkt dates >= port dates
        strategy_name variable is used when we need to run different strategies in the same folder
        '''
        self.port=feather.read_dataframe(self.path+'%s.feather' % (strategy_name)).set_index('date')
        self.mkt=feather.read_dataframe(self.path+'mkt.feather').set_index('date')

        if self.mkt.index.max()<self.port.index.max():
            print ('Warning: last date in port is beyond last date in mkt, need to add more data points to mkt')
            print ('for now we reduce the port date to mkt date')
            self.port=self.port.loc[:self.mkt.index.max()]
            #return None
        if dump_sedol_hist:
            fq=Factset_Query()
            sedol_hist=fq.get_sedol_hist(self.port.columns.tolist(),
                                         start=fql_date(self.port.index[0]))
            sedol_hist.reset_index().to_csv(self.path+"sedol.csv")

        self.sedol_hist=pd.read_csv(self.path+"sedol.csv",
                               parse_dates=['sedol_hist_start','sedol_hist_end'])
        return None
    def run(self,start,end,method='custom',
            factor_to_drop=['OffShore China'],
            dump_res=True
            ):
        '''
        We will follow port frequency
        method can be custom or default
        Cannot run on 1 date if using custom (as return will not be calculated)
        When there is no date from axioma dump, we load the data from the previous model
        Note that the end date need to be less than the date in the mkt,
        otherwise we may have some duplicated date issue
        '''
        if method not in ['custom','default']:
            print ('method needs to be either custom or default!')
            return None
        if self.verbose:
            print ('Start running from %s to %s' % (start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d')))
        port_all=self.port.loc[start:end].copy()
        region=self.region
        model=self.model
        fx_factor=self.fx_factor.copy()
        country_factor=self.country_factor.copy()
        mkt=self.mkt.copy()
        res_collector=[]
        res_output=pd.DataFrame()
        for i,date in enumerate(port_all.index):
            start_time=time.time()
            if self.verbose:
                print ('Working on %s' % (date.strftime ('%Y-%m-%d')))

            '''
            The dropna here is non-trival
            - FMP is not that sensitive to the universe. So the conclusion won't change much for time-varing universe
            - If not dropna, we may see some completely different results.
                (my guess is it may be caused by some wrong ticker or wrong price especially your mkt input has wrong price.)
            '''
            port=port_all.loc[date].dropna().fillna(0)
            display_date=date
            date=self._get_full_data_date(region,model,date)
            att,bet,cov,exp,idm,isc,rsk,ret=self._load_axioma(region,model,date)

            ### get P
            sedol_hist=self.sedol_hist.copy()
            sedol_hist['to_keep']=(sedol_hist[['sedol_hist_start','sedol_hist_end']]
                        .apply(lambda x: True if (display_date>x['sedol_hist_start'] and display_date<x['sedol_hist_end']) else False,axis=1)
                      )
            ticker_map=sedol_hist[sedol_hist['to_keep']].set_index('sedol_hist_id')[['ticker']].join(idm.set_index('SEDOL(7)')['AxiomaID']).reset_index()
            ticker_map_fs_to_axioma=ticker_map.set_index('ticker')['AxiomaID'].to_dict()
            # drop the tickers that are not in the ticker map
            to_drop_missing_map=port[~port.index.isin(ticker_map['ticker'])].index.tolist()
            if len(to_drop_missing_map)!=0 and self.verbose:
                print ('%s dropped due to missing ticker mapping' % (to_drop_missing_map))
                print ('%s port dropped' % (abs(port.loc[to_drop_missing_map].sum())))
            port=port[port.index.isin(ticker_map['ticker'])].copy()
            port.index=port.index.map(lambda x: ticker_map_fs_to_axioma[x])
            port.index.name='index'
            P=port.rename('wgt').to_frame().reset_index()
            P['index']=P['index'].fillna('NA') # we may have no AxiomaID from idm sometimes e.g. 6470 JP on 2020/11/02
            P['wgt']=P.apply(lambda x: np.nan if x['index']=='NA' else x['wgt'],axis=1)
            P=P.groupby('index').sum() # so we don't keep any duplicated ticker here
            P=P.drop('NA',errors='ignore')
            P=pd.concat([
                        P[P['wgt']>0].dropna().apply(lambda x: x/x.sum()),
                        P[P['wgt']==0].dropna(),
                        P[P['wgt']<0].dropna().apply(lambda x: -1*x/x.sum()),
                        ],axis=0)['wgt']
            ### get X
            X=exp.reindex(P.index).fillna(0)
            X.index.name='ticker'
            X.columns.name='factor'
            X=X.drop(factor_to_drop,axis=1,errors='ignore')

            if method=='custom':
                X_=X.drop(fx_factor,axis=1,errors='ignore')
                cut_n=len(X.columns)-len(X_.columns)
                fx_factor_dropped=X.columns[~X.columns.isin(X_.columns)]
                X_full=X.copy()
                if self.verbose:
                    print('drop %s fx factors' % (cut_n))
                X=X_.copy()
            ### get F_Var
            F_Var=cov.copy()
            F_Var=F_Var.drop(factor_to_drop,axis=1,errors='ignore').drop(factor_to_drop,axis=0,errors='ignore')
            ### get D
            D=rsk.reindex(P.index)/100
            D=diag(D)
            isc_to_use=isc.copy()
            isc_to_use.columns=['id1','id2','cov']
            isc_to_use=isc_to_use.set_index(['id1','id2'])['cov'].map(pd.to_numeric).reset_index()
            isc_to_use['id1_in']=isc_to_use['id1'].isin(D.index)
            isc_to_use['id2_in']=isc_to_use['id2'].isin(D.index)
            isc_to_use['need_adj']=isc_to_use['id1_in'] & isc_to_use['id2_in']
            isc_to_use=isc_to_use[isc_to_use['need_adj']]
            if len(isc_to_use)!=0:
                if self.verbose:
                    print ('Diagonal pair for D: %s' % (len(isc_to_use)))
                for j in isc_to_use.index:
                    isc_to_use_i=isc_to_use.loc[j]
                    D.at[isc_to_use_i['id1'],isc_to_use_i['id2']]=isc_to_use_i['cov']
                    D.at[isc_to_use_i['id2'],isc_to_use_i['id1']]=isc_to_use_i['cov']
            ### Do the matrix operation
            if method=='default':
                # choose to use the Axioma default model
                beta=X.T @ P
                CTC=beta * (beta.T @ F_Var).T
                CTC['Specific']=P.T @ D @ P # some stock has lower specific risk than others so perhaps that's how we reduce the specific risk with more stocks added
                                            # for risk attributable to style exposure, we do have some diversificaiton benefit

                beta_vol=beta * sqrt (diag(F_Var))
                V=X @ F_Var @ X.T + D
            else:
                # choose to use customized model
                V=X @ F_Var.loc[X.columns][X.columns] @ X.T + D
                a=inv(V) @ X
                FMP=(inv(X.T @ a) @ a.T).T
                H_Var=P.T @ V @ P
                FMP_MP_Cov=FMP.T @ V @ P
                beta=X.T @ P
                FP_Var=FMP.T @ V @ FMP
                CTC=beta * FMP_MP_Cov
                beta_vol=beta * sqrt(diag(FP_Var))

                # do the fx adjustment if necessary
                if len(fx_factor_dropped)!=0:
                    Crncy_X=X_full[fx_factor_dropped]
                    Crncy_beta=Crncy_X.T @ P
                    Crncy_MP_Cov=F_Var.loc[Crncy_beta.index][beta.index.tolist()+Crncy_beta.index.tolist()] @ pd.concat([beta*2,Crncy_beta],axis=0)
                    CrncyCTC=Crncy_beta * Crncy_MP_Cov
                    CTC=pd.concat([CTC,CrncyCTC,pd.Series(H_Var-sum(CTC),index=['Specific'])],axis=0)
                    beta=pd.concat([beta,Crncy_beta],axis=0)
                    beta_vol=pd.concat([beta_vol,Crncy_beta * sqrt(diag(F_Var[Crncy_X.columns].loc[Crncy_X.columns]))],axis=0)
                    X=pd.concat([X,Crncy_X],axis=1)
                    ccyFMP=exp.loc[FMP.index][fx_factor_dropped]
                    to_keep=ccyFMP.applymap(lambda x: np.nan if x==0 else x).count()
                    to_keep=to_keep[to_keep!=0]
                    ccyFMP=ccyFMP[to_keep.index]
                    if type(ccyFMP) is pd.Series:
                        ccyFMP=ccyFMP.to_frame()
                    totalTemp=FMP[FMP.columns[FMP.columns.isin(country_factor)]].sum().sum()/len(ccyFMP.columns)
                    ccyFMP=ccyFMP.divide(ccyFMP.sum())*totalTemp
                    FMP=pd.concat([FMP,ccyFMP],axis=1)
                    FMP=FMP.reindex(X.columns,axis=1).fillna(0).copy()
                else:
                    CTC=pd.concat([CTC,pd.Series(H_Var-sum(CTC),index=['Specific'])],axis=0)
            ### get next period return, either from Axioma or from FMP
            try:
                next_date=port_all.index[i+1]
            except IndexError:
                next_date=mkt.loc[start:end].index[-1]
            ret_s_raw=(mkt.loc[next_date]/mkt.loc[display_date]-1).fillna(0).rename(0)
            ret_s=ret_s_raw.rename(index=ticker_map_fs_to_axioma) # this may rarely create duplicates due to delisting etc (e.g. LK)
            ret_s=ret_s.reset_index().groupby('index').mean()[0]
            port_raw=port_all.loc[display_date].fillna(0)
            port_ret=ret_s_raw.reindex(port_raw.index)*port_raw


            if method=='custom': # calculate ret from FMP
                ret_s=ret_s.reindex(FMP.index)
                F_ret=FMP.multiply(ret_s,axis='index').sum()
                F_att=F_ret*beta
            else: # load daily ret from axioma file
                date_range=pd.date_range(display_date,next_date,freq='B')
                collector=[]
                for dt_i in date_range:
                    try:
                        ret_i=self._axioma_file_reader(self.region,self.model,dt_i,'ret').set_index('FactorName').apply(pd.to_numeric)[['Return']]
                        ret_i['date']=dt_i
                        collector.append(ret_i)
                    except FileNotFoundError:
                        continue
                F_ret_all=pd.concat(collector,axis=0)
                F_ret_all=F_ret_all.reset_index().set_index(['date','FactorName'])['Return'].unstack()/100
                F_ret_all.loc[display_date]=0
                F_ret=((F_ret_all+1).cumprod()-1).iloc[-1]
                F_att=F_ret*beta
            F_att.loc['Total']=port_ret.sum()

            ### Tidy-up and dump the results
            CTC_nice=self._tidy_up_res(CTC,'ctc',display_date,next_date,method)
            Factor_vol_nice=self._tidy_up_res(sqrt(diag(F_Var)),'factor_vol',display_date,next_date,method)
            beta_nice=self._tidy_up_res(beta,'beta',display_date,next_date,method)
            beta_vol_nice=self._tidy_up_res(beta_vol,'beta_vol',display_date,next_date,method)
            F_att_nice=self._tidy_up_res(F_att,'att',display_date,next_date,method)
            F_ret_nice=self._tidy_up_res(F_ret,'factor_ret',display_date,next_date,method)
            res_all=pd.concat([CTC_nice,
                               Factor_vol_nice,
                               beta_nice,
                               beta_vol_nice,
                               F_att_nice,
                               F_ret_nice,
                               ],axis=0)
            dump_name='%s-%s-%s' % (model,method,display_date.strftime('%Y_%m_%d'))
            if dump_res:
                feather.write_dataframe(res_all,self.path+'dump\\%s.feather' % (dump_name))
            end_time=time.time()
            lapsed=round((end_time-start_time),2)
            if self.verbose:
                print ('Finish %s (%s sec)' % (display_date.strftime('%Y-%m-%d'),lapsed))
            res_collector.append(res_all)
            res_output=pd.concat(res_collector)
        return res_output
    def get_covariance(self,method='default'):
        '''
        Load daily covariance from axioma for the existing port
        '''
        port_all=self.port.copy()
        region=self.region
        model=self.model
        fx_factor=self.fx_factor.copy()
        V_collector=[]
        for i,date in enumerate(port_all.index):
            port=port_all.loc[date].dropna().fillna(0)
            display_date=date
            date=self._get_full_data_date(region,model,date)
            att,bet,cov,exp,idm,isc,rsk,ret=self._load_axioma(region,model,date)

            ### get P
            sedol_hist=self.sedol_hist.copy()
            sedol_hist['to_keep']=(sedol_hist[['sedol_hist_start','sedol_hist_end']]
                        .apply(lambda x: True if (display_date>x['sedol_hist_start'] and display_date<x['sedol_hist_end']) else False,axis=1)
                      )
            ticker_map=sedol_hist[sedol_hist['to_keep']].set_index('sedol_hist_id')[['ticker']].join(idm.set_index('SEDOL(7)')['AxiomaID']).reset_index()
            ticker_map_fs_to_axioma=ticker_map.set_index('ticker')['AxiomaID'].to_dict()
            # drop the tickers that are not in the ticker map
            to_drop_missing_map=port[~port.index.isin(ticker_map['ticker'])].index.tolist()
            if len(to_drop_missing_map)!=0 and self.verbose:
                print ('%s dropped due to missing ticker mapping' % (to_drop_missing_map))
                print ('%s port dropped' % (abs(port.loc[to_drop_missing_map].sum())))
            port=port[port.index.isin(ticker_map['ticker'])].copy()
            port.index=port.index.map(lambda x: ticker_map_fs_to_axioma[x])
            port.index.name='index'
            P=port.rename('wgt').to_frame().reset_index()
            P['index']=P['index'].fillna('NA') # we may have no AxiomaID from idm sometimes e.g. 6470 JP on 2020/11/02
            P['wgt']=P.apply(lambda x: np.nan if x['index']=='NA' else x['wgt'],axis=1)
            P=P.dropna()
            P=P.groupby('index').sum() # so we don't keep any duplicated ticker here
            P=pd.concat([
                        P[P['wgt']>0].dropna().apply(lambda x: x/x.sum()),
                        P[P['wgt']==0].dropna(),
                        P[P['wgt']<0].dropna().apply(lambda x: -1*x/x.sum()),
                        ],axis=0)['wgt']
            ### get X
            X=exp.reindex(P.index).fillna(0)
            X.index.name='ticker'
            X.columns.name='factor'
            if method=='custom':
                X_=X.drop(fx_factor,axis=1,errors='ignore')
                cut_n=len(X.columns)-len(X_.columns)
                if self.verbose:
                    print('drop %s fx factors' % (cut_n))
                X=X_.copy()
            ### get F_Var
            F_Var=cov.copy()
            ### get D
            D=rsk.reindex(P.index)/100
            D=diag(D)
            isc_to_use=isc.copy()
            isc_to_use.columns=['id1','id2','cov']
            isc_to_use=isc_to_use.set_index(['id1','id2'])['cov'].map(pd.to_numeric).reset_index()
            isc_to_use['id1_in']=isc_to_use['id1'].isin(D.index)
            isc_to_use['id2_in']=isc_to_use['id2'].isin(D.index)
            isc_to_use['need_adj']=isc_to_use['id1_in'] & isc_to_use['id2_in']
            isc_to_use=isc_to_use[isc_to_use['need_adj']]
            if len(isc_to_use)!=0:
                if self.verbose:
                    print ('Diagonal pair for D: %s' % (len(isc_to_use)))
                for i in isc_to_use.index:
                    isc_to_use_i=isc_to_use.loc[i]
                    D.at[isc_to_use_i['id1'],isc_to_use_i['id2']]=isc_to_use_i['cov']
                    D.at[isc_to_use_i['id2'],isc_to_use_i['id1']]=isc_to_use_i['cov']
            ### Do the matrix operation
            if method=='default':
                # choose to use the Axioma default model
                V=X @ F_Var @ X.T + D
            else:
                # choose to use customized model
                V=X @ F_Var.loc[X.columns][X.columns] @ X.T + D
            ticker_map_axioma_to_ticker=ticker_map.set_index('AxiomaID')['ticker'].to_dict()
            V=V.rename(index=ticker_map_axioma_to_ticker).rename(columns=ticker_map_axioma_to_ticker)
            V.index.name='index';V.columns.name='column'
            V=V.stack().reset_index().rename(columns={0:'cov'})
            V['date']=display_date
            V_collector.append(V)
        V_res=pd.concat(V_collector,axis=0)
        return V_res



    def display(self,method,start,end,hide_col=[],top_N=10,
                rank_by_abs_var_contribution=True,
                load_from_dump=[True,'pass the run results here if use False'],
                adjust_pct_format=[False,'{:.1%}'],
                ):
        '''
        method can be custom or default
        function also returns the key input
        '''
        output={}
        top_N_highlight=['STYLE','INDUSTRY']
        display_orders_perf=['Residual','STYLE','INDUSTRY']
        display_orders_risk=['Specific','STYLE','INDUSTRY']
        ### load data
        dump_path=self.path+'dump\\'
        if load_from_dump[0]:
            files=um.iterate_csv(dump_path,iterate_others=[True,'.feather'])
            collector=[]
            for file in files:
                collector.append(feather.read_dataframe(dump_path+'%s.feather' % (file)))
            data=pd.concat(collector)
        else:
            data=load_from_dump[1]
        data_to_use=data[data['method']==method].copy()
        factor_type_map=data_to_use.groupby('Shortest')['Factor_type'].last().rename('factor')
        print ('%s_%s_to_%s' % (method,start.strftime('%Y%m%d'),end.strftime('%Y%m%d')))
        ### Attribution
        #perf=data_to_use[(data_to_use['field']=='att')].set_index(['date_end','Shortest'])['value'].unstack().loc[start:end]
        # we do a groupby sum here to remove the duplicates, which are the reulst of ffill axioma data. Returns on such days are zero

        perf=data_to_use[(data_to_use['field']=='att')].groupby(['date_end','Shortest'])['value'].sum().unstack().loc[start:end]
        perf.iloc[0]=0
        perf=(perf+1).cumprod()-1
        perf_detail=perf.stack().rename('perf').reset_index().set_index('Shortest').join(factor_type_map)
        perf=perf_detail.groupby(['date_end','factor']).sum()['perf'].unstack()
        perf=perf.drop(hide_col,1)
        perf['Residual']=perf['Total']-perf.drop(['Total'],1).sum(1)
        perf.index.name='date'
        # Attribution overview
        fig,axes=ud.easy_plot_quick_subplots((1,2),'Attribution' )
        nice_order=display_orders_perf+perf.drop(display_orders_perf,1).columns.to_list()
        perf[nice_order].plot(ax=axes[0],title='Cumulative performance')
        perf_summary=perf.iloc[-1][nice_order].drop('Total')
        perf_summary.plot(ax=axes[1],kind='barh',title='Absolute contribution')
        #(perf_summary/perf_summary.sum()).plot(ax=axes[2],kind='barh',title='Relative contribution',color=uc.alt_colors_quick_pd_plot[0])
        axes[0].set_xlabel('');axes[1].set_ylabel('');#axes[2].set_ylabel('')
        ud.easy_plot_pct_tick_label(axes[0],direction='y',pct_format='{:.0%}' if not adjust_pct_format[0] else adjust_pct_format[1])
        ud.easy_plot_pct_tick_label(axes[1],direction='x',pct_format='{:.0%}' if not adjust_pct_format[0] else adjust_pct_format[1])
        #ud.easy_plot_pct_tick_label(axes[2],direction='x')
        # Attribution top N
        perf_topN=perf_detail[perf_detail['date_end']==perf_detail['date_end'].max()].reset_index()
        perf_topN['perf_abs']=perf_topN['perf'].abs()
        perf_topN['rank']=perf_topN.groupby(['factor'])['perf_abs'].rank(ascending=False)
        to_plot=perf_topN[perf_topN['rank']<=top_N].set_index('Shortest')
        to_plot=to_plot.join(data_to_use[data_to_use['field']=='beta'].groupby(['Shortest']).agg({'value':'mean','Factor_type':'last'})).rename(columns={'value':'beta'})
        to_plot['color']=to_plot['beta'].map(lambda x: uc.alt_colors_quick_pd_plot[4] if x>0 else uc.alt_colors_quick_pd_plot[6])

        fig,axes=ud.easy_plot_quick_subplots((1,2),'Top-%s perf contributor (ranked by absolute contribution, green/red if avg beta above/below 0)' % (top_N),sharex=True)
        for i, factor in enumerate(top_N_highlight):
            to_plot_i=to_plot[to_plot['factor']==factor].sort_values(by='rank',ascending=False)
            to_plot_i['perf'].plot(kind='barh',ax=axes[i],title=factor,color=to_plot_i['color'].values.tolist())
            axes[i].set_ylabel('')
            ud.easy_plot_pct_tick_label(axes[i],direction='x',pct_format='{:.1%}')
        # collect output
        output['PERF_grouped']=perf.copy()
        output['PERF_topN']=to_plot.copy()
        perf_detail_output=perf_detail[perf_detail['date_end']==perf_detail['date_end'].max()]['perf'].sort_values()
        perf_detail_output['Residual']=perf_detail_output['Total']-perf_detail_output.drop('Total').sum()
        output['PERF_detail']=perf_detail_output.copy()
        ### Risk
        # highlight the beta and risk adjusted beta using box plot
        beta_types=['beta','beta_vol']
        temp_res_collector=[]
        for beta_type in beta_types:
            #force_drop=[]#['Mkt Beta']
            beta=data_to_use[(data_to_use['Factor_type'].isin(top_N_highlight)) & (data_to_use['field']==beta_type)
                             #& (~data_to_use['Shortest'].isin(force_drop))
                            ].copy()
            beta=beta[(beta['date_start']>=start) & (beta['date_end']<=end)]
            beta_grouped=beta.groupby('Shortest').agg({'value':'mean','Factor_type':'last'}).reset_index()
            beta_grouped['value_abs']=beta_grouped['value'].abs()
            beta_grouped['rank']=beta_grouped.groupby('Factor_type')['value_abs'].rank(ascending=False)
            beta_grouped=beta_grouped[beta_grouped['rank']<=top_N]
            fig,axes=ud.easy_plot_quick_subplots((1,2),'Exposure (%s) distribution, top %s by abs of average' % (beta_type,top_N))
            for i,factor_type in enumerate(top_N_highlight):
                highlight_i=beta_grouped[beta_grouped['Factor_type']==factor_type].sort_values(by='rank')['Shortest'].tolist()
                to_plot_i=beta.set_index(['date_start','Shortest'])['value'].unstack()[highlight_i]
                to_plot_i.columns=to_plot_i.columns.map(lambda x: x.replace(' ','\n'))
                ud.quick_plot_boxplot(axes[i],to_plot_i,title=factor_type,)
                ud.easy_plot_tick_label_twist(axes[i],rotation=45,va='top')
                to_plot_i=to_plot_i.stack().rename('value').reset_index()
                to_plot_i['factor_type']=factor_type
                to_plot_i['beta_type']=beta_type
                temp_res_collector.append(to_plot_i)
        beta_all=pd.concat(temp_res_collector)
        # risk contribution over time
        ctc=data_to_use[(data_to_use['field']=='ctc')].set_index(['date_start','Shortest'])['value'].unstack().loc[start:end]

        ctc_nice=ctc.stack().rename('var').reset_index().set_index('Shortest').join(factor_type_map).reset_index()
        var_overtime=ctc_nice.groupby(['date_start','factor'])['var'].sum().unstack().apply(abs)
        var_overtime_norm=var_overtime.apply(lambda x: x/x.sum(),axis=1)
        nice_order=display_orders_risk+var_overtime.drop(display_orders_risk,1).columns.tolist()
        fig,axes=ud.easy_plot_quick_subplots((1,2),'Risk')
        var_overtime_norm[nice_order].drop(hide_col,1).plot(kind='area',ax=axes[0],title='Risk contribution over time')
        axes[0].set_ylim([0,1])
        ud.quick_plot_boxplot(axes[1],var_overtime_norm[nice_order].drop(hide_col,1),title='Distribution of risk contribution')
        ud.easy_plot_pct_tick_label(axes[1],direction='y')
        # risk top N
        ctc_nice_avg=ctc_nice.groupby('Shortest').agg({'var':'mean','factor':'last'})
        ctc_nice_avg=ctc_nice_avg.join(data_to_use[data_to_use['field']=='beta'].groupby(['Shortest'])['value'].mean().rename('beta')).reset_index()
        if rank_by_abs_var_contribution:
            ctc_nice_avg['var_abs']=ctc_nice_avg['var'].abs()
        else:
            ctc_nice_avg['var_abs']=ctc_nice_avg['var'].copy()
        ctc_nice_avg['rank']=ctc_nice_avg.groupby('factor')['var_abs'].rank(ascending=False)
        to_plot=ctc_nice_avg[ctc_nice_avg['rank']<=top_N].copy()
        to_plot['color']=to_plot['beta'].map(lambda x: uc.alt_colors_quick_pd_plot[4] if x>0 else uc.alt_colors_quick_pd_plot[6])
        to_plot=to_plot.set_index('Shortest')
        factors=['STYLE','INDUSTRY']
        fig,axes=ud.easy_plot_quick_subplots((1,2),'Top-%s risk contributor (ranked by absolute contribution, green/red if avg beta above/below 0)' % (top_N),sharex=True)
        for i, factor in enumerate(factors):
            to_plot_i=to_plot[to_plot['factor']==factor].sort_values(by='rank',ascending=False)
            to_plot_i['var'].plot(kind='barh',ax=axes[i],title=factor,color=to_plot_i['color'].values.tolist())
            axes[i].set_ylabel('')
        # collect output
        output['RISK_beta']=beta_all.copy()
        output['RISK_grouped']=var_overtime.copy()
        output['RISK_topN']=to_plot.copy()
        risk_detail_output=ctc_nice_avg.set_index('Shortest').rename(index={'Specific':'Residual'})
        output['RISK_detail']=risk_detail_output
        # record the raw data
        output['raw']=data_to_use.copy()
        output['PERF_RISK']=output['RISK_detail'].join(output['PERF_detail']).drop(['rank','var_abs'],1)
        # show the risk/return tradoff
        to_plot=output['PERF_RISK'].reset_index()
        to_plot['direction']=to_plot['beta'].map(lambda x: 'long' if x>0 else 'short')
        fig=px.scatter(to_plot,x='var',y='perf',color='direction',symbol='factor',hover_name='Shortest')
        fig.show()
        return output
    def _load_axioma(self,region,model,date,factor_ret_only=False,factor_exp_only=False,factor_idm_only=False):
        '''
        Aximoa file is following the regional trading day calendar
        '''
        if not factor_ret_only and not factor_exp_only and not factor_idm_only:
            att=self._axioma_file_reader(region,model,date,'att')
            bet=self._axioma_file_reader(region,model,date,'bet').set_index('AxiomaID').apply(pd.to_numeric)
            cov=self._axioma_file_reader(region,model,date,'cov').set_index('FactorName').apply(pd.to_numeric)/10000
            exp=self._axioma_file_reader(region,model,date,'exp').set_index('AxiomaID').apply(pd.to_numeric).fillna(0)
            idm=self._axioma_file_reader(region,model,date,'idm')
            isc=self._axioma_file_reader(region,model,date,'isc')
            rsk=self._axioma_file_reader(region,model,date,'rsk').set_index('AxiomaID')['Specific Risk'].map(pd.to_numeric)
            ret=self._axioma_file_reader(region,model,date,'ret').set_index('FactorName').apply(pd.to_numeric)
            return att,bet,cov,exp,idm,isc,rsk,ret
        elif factor_ret_only:
            ret=self._axioma_file_reader(region,model,date,'ret').set_index('FactorName').apply(pd.to_numeric)
            return ret
        elif factor_exp_only:
            exp=self._axioma_file_reader(region,model,date,'exp').set_index('AxiomaID').apply(pd.to_numeric).fillna(0)
            return exp
        elif factor_idm_only:
            idm=self._axioma_file_reader(region,model,date,'idm')
            return idm
    def _axioma_file_reader(self,region,model,date,file_type):
        path=self.PATH_BASE % (region,date.year,str(date.month).zfill(2),model,date.strftime('%Y%m%d'),file_type)
        f = open(path, "r")
        lines = f.readlines()
        collector=[]
        for count,line in enumerate(lines):
            #print("Line{}: {}".format(count, line.strip()))
            # get the model fx for potential double checking
            if count==3:
                model_fx_i= [x.replace('\n','').replace('#ModelNumeraire: ','') for x in line.split('|')][0]
            # define skip condition
            if file_type not in ['cov','isc','ret','idm']:
                skip_condition=count<5 or count==6 or count==7
            else:
                skip_condition=count<5
            # loop through lines
            if skip_condition:
                continue
            if count==5: # get the column names
                columns=[x.replace('\n','').replace('#Columns: ','') for x in line.split('|')]
            else:
                collector.append([x.replace('\n','') for x in line.split('|')])
        res=pd.DataFrame(columns=columns,data=collector)
        if file_type=='idm':
            res['date']=date
            res['model_fx']=model_fx_i
        return res


    def _tidy_up_res(self,series_input,name,date_start,date_end,method):
        res=series_input.copy()
        res.index.name='factor'
        res=(res.rename('value').to_frame()
                        .join(self.factor_names.set_index('Original'))
                        .join(self.risk_factor[self.risk_factor['Model']==self.model].set_index('Description'))
                        .reset_index()
                     )
        res['Short']=res['Short'].fillna(res['factor'])
        res['Shortest']=res['Shortest'].fillna(res['factor'])
        res['Model']=res['Model'].fillna(self.model)
        res['Factor_type']=res.apply(lambda x: x['factor'] if x['factor'] in ['Specific','Total'] else x['Factor_type'],axis=1)
        res['method']=method
        res['date_start']=date_start
        res['date_end']=date_end
        res['field']=name
        return res

    def _get_full_data_date(self,region,model,date):
        try:
            att,bet,cov,exp,idm,isc,rsk,ret=self._load_axioma(region,model,date)
            return date
        except FileNotFoundError:
            if self.verbose:
                print ('skip %s as no Axioma data found, try previous working day' % (date))
            return self._get_full_data_date(region,model,date-pd.tseries.offsets.BDay())
    def get_factor_ret(self,start,end):
        dates=pd.date_range(start,end,freq='B')
        collector=[]
        for date in dates:
            date=self._get_full_data_date(self.region,self.model,date)
            try:
                ret=self._load_axioma(self.region,self.model,date,factor_ret_only=True)
                ret['date']=date
                collector.append(ret)
            except FileNotFoundError:
                continue
        ret_all=pd.concat(collector,axis=0)
        ret_all['factor']=self.factor_names.set_index('Original')['Shortest']
        ret_all=ret_all.reset_index()
        return ret_all.dropna().set_index(['date','factor'])['Return'].unstack()/100
    def get_factor_exp(self,start,end,freq='M'):
        '''
        daily freq takes too long time
        '''
        dates=pd.date_range(start,end,freq=freq)
        collector=[]
        for date in dates:
            date=self._get_full_data_date(self.region,self.model,date)
            try:
                ret=self._load_axioma(self.region,self.model,date,factor_exp_only=True)
                ret['date']=date
                collector.append(ret)
            except FileNotFoundError:
                continue
        ret_all=pd.concat(collector,axis=0,sort=True)
        ret_all=ret_all.reset_index()
        rename_dict=self.factor_names.set_index('Original')['Shortest'].to_dict()
        rename_dict['OffShore China']='offshore_china'

        ret_all=ret_all.set_index(['date','AxiomaID']).stack().reset_index().rename(columns={'level_2':'factor',0:'level'})
        factor_type_map=self.risk_factor[self.risk_factor['Model']==self.model].set_index('Description')['Factor_type'].to_dict()
        ret_all['type']=ret_all['factor'].map(lambda x: factor_type_map[x])
        ret_all['factor']=ret_all['factor'].map(lambda x: rename_dict[x])
        return ret_all
    def get_factor_idm(self,start,end,freq='M'):

        dates=pd.date_range(start,end,freq=freq)
        collector=[]
        for date in dates:
            date=self._get_full_data_date(self.region,self.model,date)
            try:
                ret=self._load_axioma(self.region,self.model,date,factor_idm_only=True)
                ret['date']=date
                collector.append(ret)
            except FileNotFoundError:
                continue
        ret_all=pd.concat(collector,axis=0,sort=True)
        ret_all=ret_all.reset_index()
#        rename_dict=self.factor_names.set_index('Original')['Shortest'].to_dict()
#        rename_dict['OffShore China']='offshore_china'
#
#        ret_all=ret_all.set_index(['date','AxiomaID']).stack().reset_index().rename(columns={'level_2':'factor',0:'level'})
#        factor_type_map=self.risk_factor[self.risk_factor['Model']==self.model].set_index('Description')['Factor_type'].to_dict()
#        ret_all['type']=ret_all['factor'].map(lambda x: factor_type_map[x])
#
#        ret_all['factor']=ret_all['factor'].map(lambda x: rename_dict[x])
        return ret_all



if __name__ =='__main__':
    print ('ok')
    path="C:\\Users\\hyin1\\temp_data\\attri bution\\china_factor_from_axioma_exposure\\"
    # # dump the sedol mapping
    attr=ATTRIBUTION('china_factor_from_axioma_exposure','AXCN4','AXCN4-MH',verbose=True,MASTER_PATH='C:\\Users\\hyin1\\temp_data\\attribution\\')
    # attr.load(dump_sedol_hist=True,strategy_name='universe')
    #exposures=attr.get_factor_exp(pd.datetime(2020,4,1),pd.datetime(2021,5,5))
    #idm=attr.get_factor_idm(pd.datetime(2020,4,1),pd.datetime(2021,5,5))

#    MASTER_PATH="C:\\Users\\hyin1\\temp_data\\attribution\\"
#    region='AXCN4'
#    model='AXCN4-MH'
#    method='default'
#
#    strategy_name='nb_csi300'
#
#
#    start=pd.datetime(2021,1,30)
#    end=um.yesterday_date()
#
#    attr=ATTRIBUTION(strategy_name,region,model,MASTER_PATH=MASTER_PATH,verbose=True)
#    attr.load(dump_sedol_hist=False)
#    attr.run(start,end,method=method) # do run in launcher only
#    display_output=attr.display(method,start,end,hide_col=['MARKET'])

    # testing to get factor performance only
#    strategy_name=''
#    region='AXCN4'
#    model='AXCN4-MH'
#
#
#    attr=ATTRIBUTION(strategy_name,region,model,verbose=False,
#                     MASTER_PATH="C:\\Users\\hyin1\\temp_data\\eastmoney_quick\\backtest_vs_nav\\signals\\"
#                     )
#    start=pd.datetime(2018,1,1)
#    end=pd.datetime(2021,5,31)
#
#
#    fund_i=11
#    attr.load(dump_sedol_hist=False,strategy_name=str(fund_i))
#    res=attr.run(start,end,dump_res=False,method='default',)
#
#    output=attr.display('default',start,end,hide_col=[],top_N=10,
#                rank_by_abs_var_contribution=True,
#                load_from_dump=[False,res])
#    dates=pd.date_range(start,end,freq='B')
#
#
#    collector=[]
#    for date in dates:
#        try:
#            ret=attr._load_axioma(region,model,date,factor_ret_only=True)
#            ret['date']=date
#            collector.append(ret)
#        except FileNotFoundError:
#            continue
#
#    ret_all=pd.concat(collector,axis=0)
#    ret_all['factor']=attr.factor_names.set_index('Original')['Shortest']
#    ret_all=ret_all.reset_index()

#    strategy_name='dummy_for_attribution'
#    region='AXCN4'
#    model='AXCN4-MH'
#    method='default'
#    bt_path="Z:\\dave\\data\\backtester\\Edmond_EnergyRevolution\\"
#
#    attr=ATTRIBUTION(strategy_name,region,model,MASTER_PATH=bt_path,verbose=False)
#    attr.load(dump_sedol_hist=False)
#    # #just run the "run" once. pretty slow somehow
#    attr.run(pd.datetime(2014,12,31),attr.port.index[-1],method=method,)
#    res=attr.display(method,attr.port.index[0],
#                     attr.port.index[-1],
#                     rank_by_abs_var_contribution=True)
#




#    MASTER_PATH="Z:\\dave\\data\\backtester\\Edmond_IGBT\\"
#    strategy_name='dummy_for_attribution'
#    region='AXCN4'
#    model='AXCN4-MH'
#    method='default'
#
#    attr=ATTRIBUTION(strategy_name,region,model,MASTER_PATH=MASTER_PATH,verbose=False)
#    attr.load(dump_sedol_hist=False)
#    cov_all=attr.get_covariance()



#    MASTER_PATH="C:\\Users\\hyin1\\temp_data\\attribution\\"
#
#    strategy_name='retail_TPX'
#    region='AXJP4'
#    model='AXJP4-MH'
#    method='custom'
#
#    start=pd.datetime(2009,12,1)
#    end=pd.datetime(2021,1,31)
#
#    attr=ATTRIBUTION(strategy_name,region,model,MASTER_PATH=MASTER_PATH,verbose=False)
#    attr.load(dump_sedol_hist=False)
#    attr.run(start,end,method=method) # do run in launcher only
#    display_output=attr.display(method,start,end,hide_col=['MARKET'])




#### temp adj for port index
#    jp_cdr=bdh(['TPX Index'],['px_last'],pd.datetime(2011,12,31),um.yesterday_date())['px_last'].unstack().T
#    port=feather.read_dataframe("Z:\\dave\\data\\jcm\\prod\\QUICK_OUTPUT_CSJAJCMA_port_snapshot.feather").set_index('date')
#    new_index=[]
#    for dt in port.index:
#        if dt in jp_cdr.index:
#            new_index.append(dt)
#        else:
#            new_index.append(jp_cdr.loc[:dt].index[-1])
#    port.index=new_index
#    port.index.name='date'
#    #feather.write_dataframe(port.reset_index(),"Z:\\dave\\data\\attribution\\jcm\\port.feather")
#
#    # adding the short leg
#    tpx=feather.read_dataframe("Z:\\dave\\data\\index_compo\\TPX Index.feather")
#    tpx['ticker']=bbg_to_fs(tpx['ticker'])
#    tpx=((tpx.set_index(['asof','ticker'])['wgt'].unstack()
#        .fillna(0)*(-1)).resample('B').last().fillna(method='ffill')
#        .applymap(lambda x: np.nan if x==0 else x)
#        .stack().unstack()
#        .reindex(port.index)
#        )
#
#    port_ls=pd.concat([port,tpx],axis=1)
#
#    port_ls=port_ls.stack().reset_index().groupby(['date','level_1'])[0].sum().unstack()
#
#    feather.write_dataframe(port_ls.reset_index(),"Z:\\dave\\data\\attribution\\jcm\\port.feather")
#
#









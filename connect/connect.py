# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:00:31 2019

@author: hyin1
"""


import pandas as pd
import numpy as np
import utilities.misc as um
import utilities.constants as uc
import pdb
from blp.bdx import bdh,bdp,bds
from fql.fql import Factset_Query
from fql.util import bbg_to_fs, fs_to_bbg,fql_date
from blp.util import get_bbg_usual_col, group_marcap,get_ashare_exchange,load_compo
import feather
import os

#plotly related
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utilities.display as ud

from datetime import datetime

from webscraping.ccass import download_NB,download_SB,get_nb_universe,update_bbg_ccass_id_map


'''
To summary:
1. we shift on HK cash trading day
2. we use unadjusted price times shifted shares change to estimate $ flow (T+2 settlement)
3. different corporate action will have different impact on our data
    - cash dividend: no impact
    - (TO DO) stock dividend: ccass holdings will jump on div payable day. Adjustment requires reducing payable days holdings (before shifting) by stock div amount calculated on ex-dates
    - Split: will take effect on effective day. This will mess up the T+2 adjustment. No easy solution


June 25 2019 update: corporate action can mess up the CCASS data even after the T+2 adjustment.
e.g. 787 HK on 4/9/2019. Reverse split became effective on 4/9/2019 but
    CCASS holdings adjustment took 2 days for adjustment (4/10/19 records significant drop in shares attributable to reverse split)

quick solution: Better fix this issue by dropping the >100% turnover day


27 Dec 2018 update: shall we think about directly sourcing holding info from HKEx? Because sometimes Webb can be lagging by 1 day (but no SH/SZ breakdown available on HKEx)

Change of ticker, delisting etc may impact the history


Sanity check of webbsite data: just check 700 HK shares over time, see if there is any funny pattern

'''


class STOCK_CONNECT():
    MASTER_PATH=uc.root_path_data+"connect\\"
    
    START_SB=datetime(2014,11,1)
    START_NB=datetime(2016,6,1)
    # we need this because our NB data starts in mid-16. The first row will have super high to impact, which is wrong
    START_NB_CLEAN=datetime(2016,6,23)
    SC_SHARES_FILE='sc_hld_shares.feather'
    MKT_FILE='mkt_data.feather'
    DB_FILES='db.feather'
    MAX_DAILY_IMPACT=0.8
    
    TICKERS_TO_IGNORE_FOR_TOPUP=['601313-CN','000022-CN','000043-CN'] # ticker change that cannot be easily handeld
    
    def __init__(self,direction='sb',skip_fql=False):
        if not skip_fql:
            self.fq=Factset_Query()

        sub_path='southbound' if direction =='sb' else 'northbound'
        self.path="%s%s\\" % (self.MASTER_PATH,sub_path)
        self.direction=direction
        self.start=self.START_SB if direction=='sb' else self.START_NB
        self.mkt_index='HSI Index' if direction=='sb' else 'SHSZ300 Index'
        self.ah_map=pd.read_csv(uc.root_path_data+"ah_db\\AH Pair.csv")
        self._bbg_tickers=[
                'HSI Index','SHSZ300 Index','CNYUSD Curncy','HKDUSD Curncy',
                 'H1DBTO Index', 'H1DSTO Index', 'H2DBTO Index', 'H2DSTO Index', 'HKSEVALU Index',
                 'C1DBTO Index', 'C1DSTO Index', 'C2DBTO Index', 'C2DSTO Index', 'VUSHCOMP Index','VUSZCOMP Index'
                 ]

        self.buy_col=['H1DBTO Index','H2DBTO Index'] if self.direction=='sb' else ['C1DBTO Index','C2DBTO Index']
        self.sell_col=['H1DSTO Index','H2DSTO Index'] if self.direction=='sb' else ['C1DSTO Index','C2DSTO Index']
        self.cash_col=['HKSEVALU Index'] if self.direction=='sb' else ['VUSHCOMP Index','VUSZCOMP Index']
        # for compliance reason, only drop for sankey display
        self.names_to_drop=[]
        return None

    def load_db(self,add_momentum=True, tag_cn_ex=True, add_idx_wgt=True):
        self.db=feather.read_dataframe(self.path+self.DB_FILES)
        if add_momentum:
            if self.direction=='sb':
                mom_px=feather.read_dataframe(uc.root_path_data+"excess_return\\carhart\\HSCI_SB_DailyReturn.feather").set_index('date')/100
            else:
                mom_px=feather.read_dataframe(uc.root_path_data+"excess_return\\carhart\\SHCOMP_L_SZCOMP_L_NB_DailyRetDecomp.feather").set_index('date')/100
            
            mom_px=(mom_px+1).cumprod()
            mom=mom_px.pct_change(252)-mom_px.pct_change(21)
            mom=mom.apply(lambda x: x.dropna().rank(pct=True),axis=1)
            db=self.db.set_index(['date','ticker'])
            db['mom']=mom.stack()
            self.db=db.reset_index()
        if tag_cn_ex:
            if self.direction=='nb':
                self.db['exchage']=self.db['ticker'].map(lambda x: get_ashare_exchange(x))

        if add_idx_wgt:
            if self.direction=='nb':
                self.db=self.db.set_index(['date','ticker'])
                mkt_tickers=[
                            #'MXCN',
                             #'MXCN1A',
                             #'MBCN1A',
                             'SHSZ300','SH000905','SZ399006']
                
                compo_collector=[]
                for mkt_ticker in mkt_tickers:
                    compo_i=load_compo(mkt_ticker+' Index').reset_index()
                    compo_i['idx']=mkt_ticker+' Index'
                    compo_collector.append(compo_i)
                compo_all=pd.concat(compo_collector,axis=0)

                compo_all['ticker']=bbg_to_fs(compo_all['ticker'])
                compo_all=compo_all[compo_all['date']>=datetime(2015,12,31)]
                clean_index=pd.date_range(datetime(2015,12,31),um.yesterday_date(),freq='B')

                compo_all=compo_all.set_index(['date','ticker','idx']).unstack().unstack().fillna(0).reindex(clean_index).fillna(method='ffill').stack().stack()

                for mkt_ticker in mkt_tickers:
                    self.db[mkt_ticker+'_wgt']=compo_all.xs(level='idx',key=mkt_ticker+' Index',axis=0)['wgt']
                self.db=self.db.reset_index()
        self.db_clean=self._get_clean_db()

    def _get_clean_db(self):
        '''
        Drop the macro bbg index
        '''
        db=self.db.copy()
        db=db[~db['ticker'].isin(self._bbg_tickers)]
        db=db.set_index(['date','ticker']).unstack()
        if self.direction=='nb':
            db=db.loc[self.START_NB_CLEAN:]
        return db
    
    def get_macro(self):
        '''
        We return the followings:
        holdings: mtm, cost, stake (marcap and ff)
        flow: buy, sell, gross impact and net impact
        count
        ah: aggregate levels: equal/mkt/sc holdings
        others to add later
        '''
        db=self.db.copy()
        buy=self.buy_col
        sell=self.sell_col
        cash=self.cash_col
        all_col=buy+sell+cash
        #macro
        macro=db.groupby(['date','ticker'])['px_last'].last().unstack()[all_col].fillna(0)
        macro['fx']=db.groupby('date')['fx'].last()
        macro['Buy flow']=(macro[buy].sum(1)*macro['fx']/1000).map(lambda x: np.nan if x==0 else x).rolling(21,min_periods=1).mean()
        macro['Sell flow']=(macro[sell].sum(1)*macro['fx']/1000).map(lambda x: np.nan if x==0 else x).rolling(21,min_periods=1).mean()
        macro['Net flow']=macro['Buy flow']-macro['Sell flow']
        macro['Gross flow']=macro['Buy flow']+macro['Sell flow']
        macro['Holdings MtM']=db.groupby('date')['holdings_musd'].sum()/1000
        macro['Holdings cost']=macro['Net flow'].cumsum()/1000
        macro['_cash_to']=macro[cash].sum(1) if self.direction=='sb' else macro[cash].sum(1)
        macro['_cash_to']=macro['_cash_to'].rolling(21,min_periods=21).mean()
        macro['Gross impact']=macro['Gross flow']/macro['_cash_to']
        macro['Net impact']=macro['Net flow']/macro['_cash_to']
        macro['Count']=(db.set_index(['date','ticker'])['sc_hld_shares'].unstack()
                    .drop(self._bbg_tickers,1, errors='ignore')
                    .applymap(lambda x: np.nan if x==0 else x).count(1)
                    .map(lambda x: np.nan if x==0 else x).dropna())
        #marcap and ff stake
        stake_calc=db.groupby(['date'])[['holdings_musd','marcap_sec','ff_marcap_sec']].sum()
        macro['Stake marcap']=stake_calc['holdings_musd']/stake_calc['marcap_sec']
        macro['Stake float']=stake_calc['holdings_musd']/stake_calc['ff_marcap_sec']
        macro['Stake marcap']=macro['Stake marcap']/macro['fx']
        macro['Stake float']=macro['Stake float']/macro['fx']
        #ah related
        db_ah=db[db['ah_stats'].map(lambda x: False if np.isnan(x) else True)]
        macro['Equal-wgt']=db_ah.groupby('date')['ah_stats'].mean()
        mx=db_ah.set_index(['date','ticker']).unstack()
        macro['MarCap-wgt']=(mx['marcap_sec'].apply(lambda x: x/x.sum(),axis=1)
                              .multiply(mx['ah_stats']).sum(1))
        macro['Holding-wgt']=(mx['holdings_musd'].apply(lambda x: x/x.sum(),axis=1)
                              .multiply(mx['ah_stats']).sum(1))
        macro.columns.name='field'
        return macro

    def get_breakdown(self,whats,by,how_group,how_ts,
                      start=um.today_date()-24*pd.tseries.offsets.BDay(),
                      end=um.today_date()-3*pd.tseries.offsets.BDay(),
                      focus=[False,'focus group name']):
        '''
        whats: (needs to be a list!) the field(s) to show, such as holdings_musd/flow_musd/impact_daily etc
        by: how to group, such as sector/industry/size/stake_marcap_q/stake_ff_q
        how_group: can ONLY BE sum/mean/median/count
        how_ts: can ONLY BE sum/mean/first/last/nan. If nan then returns timeseries (e.g. qunitiles)
        start/end: time series range
        focus: show single stock for the given focus group name.
              Once this is chosen, "by" argument is effectively 'ticker' and how_group is effectively last
        '''
        if type(whats) is not list:
            whats=[whats]
        db=self.db.copy()
        db=db.set_index(['date','ticker']).unstack().loc[start:end].stack()
        def _how_func(x,how):
            if how=='mean':
                return x.mean()
            elif how=='median':
                return x.median()
            elif how=='last':
                return x.iloc[-1]
            elif how=='first':
                return x.iloc[0]
            elif how=='sum':
                return x.sum()
            elif how=='count':
                return x.count()
            elif how=='nan':
                return x
            else:
                print ('invalid how method')
                return None
        if not focus[0]:
            res=(
                db.groupby(['date',by])[whats].apply(_how_func,how=how_group).unstack()
                .loc[start:end].apply(_how_func,how=how_ts)
                )
        else:
            res=(
                db[db[by]==focus[1]].groupby(['date','ticker'])[whats].apply(_how_func,how='last').unstack()
                .loc[start:end].apply(_how_func,how=how_ts)
                )
        return res.unstack().T if how_ts !='nan' else res

    def get_sankey(self,direction='buy', #or sell,
                   look_back=21,look_back_longer=42,ss_top=1,flow_acceleration_level=0.5,
                   chart_para={
                            'title': 'From %s to %s\n(vs. %s to %s)',
                            'height': 800, 'width': 1200,
                            'thickness': 15, 'font_size': 11, 'font_color': 'black', 'pad': 12,
                            'margin':dict(l=20, r=20, t=20, b=20,pad=4),
                            },
                   ):
        '''
        This function can only be used in Jupyter Notebook
        import warnings to supress warnings
        '''
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_dict,start_l,end_l,start,end=self._get_sankey_df(look_back,look_back_longer,ss_top,flow_acceleration_level)
            chart_para['title']=chart_para['title'] % (start.strftime('%Y-%m-%d'),
                                                      end.strftime('%Y-%m-%d'),
                                                      start_l.strftime('%Y-%m-%d'),
                                                      end_l.strftime('%Y-%m-%d'))
            self._display_sankey(df_dict[direction],chart_para)
    def _display_sankey(self,df,chart_para):
        '''
        df is the well-organized sankey df with the following columns:
        source, target, value, color_node, color_value, node_label
        chart_para={
            'title':'chart title',
            'height':772,
            'width':950,
            'thickness':10,
            'font_size':11,
            'font_color':'white'
            },
        '''
        sankey_input=go.Sankey(
            #node
            node = dict(
              pad = chart_para['pad'],   #pad controls the distance between each group
              thickness = chart_para['thickness'],  #pad controls the width of each bar
              line = dict(
                color = "black",   #color of the bar edge
                width = 0.5  # width of the bar edge
              ),
              label =  df['node_label'].dropna(axis=0, how='any'),
              color = df['color_node'] #color of each bar, checking out examples: https://plot.ly/~alishobeiri/1591/plotly-sankey-diagrams/#/
              ),
            #link
            link = dict(
              source = df['source'].dropna(axis=0, how='any'),
              target = df['target'].dropna(axis=0, how='any'),
              value = df['value'].dropna(axis=0, how='any'), # value controls the width the link
                color=df['color_value']  # here we set the link color
                )
            )
        fig=go.Figure(data=[sankey_input])
        fig.update_layout(title=chart_para['title'],
                        font=dict(size=chart_para['font_size'],color=chart_para['font_color']),
                        height=chart_para['height'],
                        width=chart_para['width'],
                        margin=go.layout.Margin(
                                    l=chart_para['margin']['l'],
                                    r=chart_para['margin']['r'],
                                    b=chart_para['margin']['b'],
                                    t=chart_para['margin']['t'],
                                    pad=chart_para['margin']['pad']
                     ))
        fig.show()

    def _get_sankey_df(self,look_back,look_back_longer,ss_top,flow_acceleration_level):
        '''
        We drop 2013-HK from display
        '''
        to_drop=list(self.names_to_drop)
        db=self.db.copy()
        db=db[~db['ticker'].isin(to_drop)]
        db['ticker']=db['ticker'].map(lambda x: x.replace('-CN',' CH'))
        effective_last=db.groupby('date')['flow_musd'].sum().map(lambda x: np.nan if x==0 else x).dropna().index[-1]
        end=effective_last
        start=db.groupby('date')['flow_musd'].sum().loc[:end].iloc[-1*look_back:].index[0]
        start_l=db.groupby('date')['flow_musd'].sum().loc[:end].iloc[-1*look_back_longer:].index[0]
        flow_m=db.groupby(['date','ticker'])['flow_musd'].sum().unstack().loc[start:end]
        window=len(flow_m)
        #add historical flow for color coding, no overlapping
        flow_m_l=db.groupby(['date','ticker'])['flow_musd'].sum().unstack().loc[start_l:start].iloc[:-1]
        end_l=flow_m_l.index[-1]
        window_l=len(flow_m_l)
        flow_m=flow_m.sum()
        flow_m_l=flow_m_l.sum()
        data=pd.concat([
                        db.groupby(['ticker'])['short_name'].last().drop(self._bbg_tickers,errors='ignore').rename('name'),
                        db.groupby(['ticker'])['sector'].last().drop(self._bbg_tickers,errors='ignore'),
                        db.groupby(['ticker'])['industry'].last().drop(self._bbg_tickers,errors='ignore'),
                        flow_m.rename('flow'),
                        flow_m_l.rename('flow_l'),
                        ],axis=1,sort=True)
        data.index.name='ticker'
        data=data.reset_index()
        data=data[data['name'].fillna(0).map(lambda x: True if x!=0 else False)]
        directions=['buy','sell']
        #ss_top=1
        mm_flow=0 #setting mm_flow may mess up the chart so don't do it
        #flow_acceleration_level=0.5
        def _get_color(x):
            if x['flow']*x['flow_l']<0:
                return 2
            elif abs(x['flow'])/abs(x['flow_l'])-1>=flow_acceleration_level:
                return 1
            else:
                return 0
        mentions=[]
        collectors={}
        for direction in directions:
            to_plot=data[data['flow']>0 if direction=='buy' else data['flow']<=0]
            to_plot_color_code=data[data['flow']>0 if direction=='buy' else data['flow']<=0]
            node_color=uc.alt_colors[6 if direction=='sell' else 4]
            flow_color=uc.alt_colors[15 if direction=='sell' else 15]
            highlight_color=uc.alt_colors[7 if direction=='sell' else 5]
            highlight_color_dir=uc.alt_colors[16]
            highlight_color_dict={2:highlight_color_dir,
                                  1:highlight_color,}
            if direction=='sell':
                #to_plot[['flow','impact']]=to_plot[['flow','impact']].apply(lambda x: abs(x),axis=1)
                to_plot[['flow']]=to_plot[['flow']].apply(lambda x: abs(x),axis=1)
            sankey=pd.DataFrame(index=[0],columns=['source', 'target', 'value', 'color_node', 'color_value', 'node_label'])
            '''
            There may be cases where no value is recorded (i.e. no inflow to dual real estate).
            We need to fill value by 0
            '''
            #do the sector breakdown
            value=to_plot.groupby('sector').sum()['flow']
            #count=to_plot.groupby('sector').count()['flow']
            value_avg=to_plot_color_code.groupby('sector').sum()['flow']/window
            value_avg_l=to_plot_color_code.groupby('sector').sum()['flow_l']/window_l
            highlighter=pd.concat([value_avg,value_avg_l],axis=1)
            highlighter['highlight']=highlighter.apply(lambda x: _get_color(x),axis=1)
            for i,col in enumerate(value.index):
                sankey.at[i,'source']=i
                sankey.at[i,'value']=value[col]
                sankey.at[i,'color_node']='rgb%s' % (str(node_color))
                sankey.at[i,'color_value']='rgb%s' % (str(flow_color if highlighter['highlight'][col]==0 else highlight_color_dict[highlighter['highlight'][col]]))
                sankey.at[i,'node_label']='%s (US$ %sm)' % (col,'{:.0f}'.format(-1*value[col] if direction=='sell' else 1*value[col]),)
                sankey.at[i,'temp_index']=col
            #do the industry breakdown
            base=len(sankey)
            value=to_plot.groupby(['sector','industry']).sum()['flow']
            #count=to_plot.groupby(['sector','industry']).count()['flow']
            value_avg=to_plot_color_code.groupby(['sector','industry']).sum()['flow']/window
            value_avg_l=to_plot_color_code.groupby(['sector','industry']).sum()['flow_l']/window_l
            highlighter=pd.concat([value_avg,value_avg_l],axis=1)
            highlighter['highlight']=highlighter.apply(lambda x: _get_color(x),axis=1)
            for i,col in enumerate(value.index):
                sankey.at[base+i,'source']=base+i
                sankey.at[base+i,'value']=value[col]
                sankey.at[base+i,'color_node']='rgb%s' % (str(node_color))
                sankey.at[base+i,'color_value']='rgb%s' % (str(flow_color if highlighter['highlight'][col]==0 else highlight_color_dict[highlighter['highlight'][col]]))
                sankey.at[base+i,'node_label']='%s (US$ %sm)' % (col[1],'{:.0f}'.format(-1*value[col] if direction=='sell' else 1*value[col]),)
                sankey.at[base+i,'temp_index']='%s-%s' % (col[0],col[1])
                sankey.at[base+i,'target']=sankey.set_index('temp_index').loc[col[0]]['source']
            #stock highlight: Top x flows, show impact and stake as well
            ss=to_plot.set_index(['sector','industry','ticker'])['flow'].unstack().sort_index()
            value_avg=to_plot_color_code.groupby(['sector','industry','ticker']).sum()['flow']/window
            value_avg_l=to_plot_color_code.groupby(['sector','industry','ticker']).sum()['flow_l']/window_l
            highlighter=pd.concat([value_avg,value_avg_l],axis=1)
            highlighter['highlight']=highlighter.apply(lambda x: _get_color(x),axis=1)
            for i,col in enumerate(ss.index):
                base=len(sankey)
                names=ss.iloc[i].dropna().sort_values(ascending=False)[:ss_top]
                names=names[names>=mm_flow]
                if len(names)!=0:
                    for j, ticker in enumerate(names.index):
                        mentions.append(ticker)
                        sankey.at[base+j,'source']=base+j
                        sankey.at[base+j,'node_label']='%s-%s (US$ %sm)' % (to_plot.set_index('ticker').loc[ticker]['name'],
                                                                                  ticker.replace(' HK','.HK'),
                                                                                  '{:.0f}'.format(-1*to_plot.set_index('ticker').loc[ticker]['flow'] if direction=='sell' else to_plot.set_index('ticker').loc[ticker]['flow']))
                        sankey.at[base+j,'temp_index']='na'
                        sankey.at[base+j,'color_node']='rgb%s' % (str(node_color))
                        sankey.at[base+j,'color_value']='rgb%s' % (str(flow_color if highlighter['highlight'][(col[0],col[1],ticker)]==0 else  highlight_color_dict[highlighter['highlight'][(col[0],col[1],ticker)]]))
                        sankey.at[base+j,'value']=to_plot.set_index('ticker').loc[ticker]['flow']
                        sankey.at[base+j,'target']=sankey.set_index('temp_index').loc['%s-%s' % (col[0],col[1])]['source']
            #calculate the total
            value_avg=to_plot_color_code.sum()['flow']/window
            value_avg_l=to_plot_color_code.sum()['flow_l']/window_l
            need_highlight=1 if (value_avg*value_avg_l<0 or value_avg/value_avg_l-1>=flow_acceleration_level) else 0
            last_row=len(sankey.index)
            sankey.at[last_row,'value']=to_plot['flow'].sum()
            sankey.at[last_row,'source']=last_row
            sankey.at[last_row,'target']=last_row+1
            sankey.at[last_row,'color_node']='rgb%s' % (str(node_color))
            sankey.at[last_row,'color_value']='rgb%s' % (str(flow_color if need_highlight==0 else highlight_color))
            sankey.at[last_row,'node_label']='Total (US$ %sm)' % ('{:.0f}'.format(-1*to_plot.sum()['flow'] if direction=='sell' else to_plot.sum()['flow']) )
            sankey.at[last_row,'temp_index']='na'
            #seal the data
            sankey['target']=sankey['target'].fillna(last_row)
            #nan fill the last point
            sankey.loc[len(sankey)-1,'value']=np.nan
            collectors[direction]=sankey
        return collectors,start_l,end_l,start,end


    def update_connect_data(self,top_up=True):

        path=self.path
        file_name=self.SC_SHARES_FILE
        nice_col={
                #for sb
                'Last Holding Change Date':'date','BBG Ticker':'ticker','Holding':'sc_hld_shares',
                #for nb
                'date':'date','bbg_ticker':'ticker','holdings_shares':'sc_hld_shares',
                  }
        keep_col=['ticker','sc_hld_shares','date']
        tags=['sh','sz']
        all_files_dict={}
        #check for existing dump
        if not os.path.isfile(self.path+file_name):
            print ('No existing dump found, force refresh')
            top_up=False
        #decide the csv list to collect
        for tag in tags:
            path=self.path+tag+'\\'
            if not top_up:
                print ('re-load all csv dumps for %s' % (path))
                all_files=um.iterate_csv(path)
            else:
                print ('checking csv dump date for %s' % (path))
                all_files=um.iterate_file_time(path)
                existing_file=feather.read_dataframe(self.path+file_name)
                last_update_date=existing_file['date'].max()
                all_files.index=all_files.index.map(lambda x: x.replace('.csv','')).map(pd.to_datetime)
                all_files=all_files.sort_index()
                all_files=all_files.loc[last_update_date:]
                all_files=all_files.index.map(lambda x: x.strftime('%Y-%m-%d')).tolist()
            all_files_dict[tag]=all_files
        #load the csv list
        collector_all=[]
        for tag in tags:
            collector=[]
            path=self.path+tag+'\\'
            for file in all_files_dict[tag]:
                    df=(pd.read_csv(path+file+'.csv',
                        parse_dates=['Last Holding Change Date'] if self.direction=='sb' else ['date'])
                        .rename(columns=nice_col)[keep_col])
                    #force date to be downloading date.
                    #Webbsite last time change date is wrong sometimes
                    df['date']=file
                    collector.append(df)
                    print ('finish loading %s - %s' % (tag,file))
            sc_hld_shares_tag=pd.concat(collector,axis=0)
            sc_hld_shares_tag['date']=sc_hld_shares_tag['date'].map(pd.to_datetime)
            #drop the duplicated for the given tag (non-trading day)
            sc_hld_shares_tag=sc_hld_shares_tag.groupby(['date','ticker']).last().reset_index()
            collector_all.append(sc_hld_shares_tag)
        sc_hld_shares=pd.concat(collector_all,axis=0)
        sc_hld_shares=sc_hld_shares.groupby(['date','ticker']).sum().reset_index()

        #merge with existing dump if needed, and dump
        if top_up:
            sc_hld_shares_old=feather.read_dataframe(self.path+file_name)
            sc_hld_shares=pd.concat([sc_hld_shares_old,sc_hld_shares],axis=0)
            sc_hld_shares=sc_hld_shares.groupby(['date','ticker']).last().reset_index()
        #extra steps for NB: merging earlier holding data
        if self.direction=='nb':
            nb_shares_old=pd.read_csv(self.path+'nb_shares_old.csv',
                                      parse_dates=['Last Holding Change Date'])
            nb_shares_old=nb_shares_old.rename(columns=nice_col).set_index('date')
            nb_shares_old=nb_shares_old.stack().rename('sc_hld_shares').reset_index()
            nb_shares_old=nb_shares_old.rename(columns={'level_1':'ticker'})
            sc_hld_shares=pd.concat([nb_shares_old,sc_hld_shares],axis=0)
            sc_hld_shares=sc_hld_shares.groupby(['date','ticker']).last().reset_index()
        feather.write_dataframe(sc_hld_shares,self.path+file_name)
        print ('Connect data updated and dumped for %s' % self.direction)
        return None
    
    def update_mkt_data(self):
        '''
        Refresh all using fql for the below
        - ts needed: px_last, vwap, turnover, volume, shout_sec, marcap_sec,
                    pe,pb,div_yield,
                    roe
        - may be free float as well
        additional stuff from bbg for the below
        - GICS sector: gics_sector_name, gics_industry_group_name, short_name
        - mkt index trading dates: HSI Index or SHSZ300 Index
        - fx: CNY and HKD
        - stock connect macro:
            sb: H1DBTO, D1DSTO, H2DBTO, H2DSTO, HKSEVALU
            nb: C1DBTO, C1DSTO, C2DBTO, C2DSTO, VUSHCOMP,VUSZCOMP
        
        for fs related download, we enable topup up update
        '''
        import time
        start_time=time.time()
        fq=self.fq
        file_name=self.SC_SHARES_FILE
        sc_hld_shares=feather.read_dataframe(self.path+file_name)

        #temp fix for HKEx Error for NB
        sc_hld_shares['ticker']=sc_hld_shares['ticker'].map(lambda x: '000572 CH Equity' if x=='00572 CH Equity' else x)
        tickers_bbg=sc_hld_shares.groupby('ticker').last().index.tolist()
        tickers_fs=bbg_to_fs(tickers_bbg)
        
        #remove the ticker change
        tickers_fs=[x for x in tickers_fs if x not in self.TICKERS_TO_IGNORE_FOR_TOPUP]
        
        #get the factset data
        start_fs=self.start.strftime('%m/%d/%Y')
        
        
        fields_topup=['turnover','marcap_sec','pe','pb','div_yield' ]
        fields_full=['px_last', 'vwap', 'volume', 'shout_sec',]
        
        top_up_path=self.path+'mkt_data_topup_update_record\\'
        has_topup=os.path.isfile(top_up_path+'fs_mkt_data_topup.feather')
        has_unadj=os.path.isfile(top_up_path+'fs_mkt_data_unadj.feather')
        
        if has_unadj:
            universe_old=feather.read_dataframe(top_up_path+'universe_last_run.feather')
            universe_chg_check=pd.Series(index=tickers_fs,data=1).rename('new').to_frame().join(universe_old.set_index('ticker')['date'].rename('old'),how='outer').count()
            

            if universe_chg_check['new']>universe_chg_check['old']:
                universe_chg=True
                print ('universe change detected. Need to re-run all the old topup record')
                # Also sending a notificaiton email to myself about this
                # If we constanly trigger refresh then it's likely due to ticker change, use the below 2 lines to find out the relevant tickers and drop them from self.TICKERS_TO_IGNORE_FOR_TOPUP
                check=pd.Series(index=tickers_fs,data=1).rename('new').to_frame().join(universe_old.set_index('ticker')['date'].map(lambda x: 1).rename('old'),how='outer')
                new_tickers_to_validate=check[(check['new']==1) & (check['old']!=1)].index.tolist()
                um.quick_auto_notice('Connect (%s) new tickers found (%s)' % (self.direction,';'.join(new_tickers_to_validate)))
                
            elif universe_chg_check['new']==universe_chg_check['old']:
                universe_chg=False
            else:
                print ('not sure if new universe can be smaller then old universe, stop to check')
                pdb.set_trace()
                
        else:
            universe_chg=False
            
        if (not has_topup) or (not has_unadj) or universe_chg:
            print ('No existing dump found for topup eligible fields, OR universe change has detected, start downloading')
        
            fs_mkt_data_topup=fq.get_ts(tickers_fs,fields_topup,start=start_fs,adj=False)
            fs_mkt_data_topup=fs_mkt_data_topup.stack().reset_index()
            feather.write_dataframe(fs_mkt_data_topup, top_up_path+'fs_mkt_data_topup.feather')
        
            fs_mkt_data_unadj=fq.get_ts(tickers_fs,['vwap','px_last'],start=start_fs,adj=False)
            fs_mkt_data_unadj=fs_mkt_data_unadj.stack().reset_index()
            feather.write_dataframe(fs_mkt_data_unadj, top_up_path+'fs_mkt_data_unadj.feather')
           
        else:
            print ('Get ready for top up update')
            
            fs_mkt_data_topup_old=feather.read_dataframe(top_up_path+'fs_mkt_data_topup.feather')
            fs_mkt_data_unadj_old=feather.read_dataframe(top_up_path+'fs_mkt_data_unadj.feather')
            
            start_topup=fs_mkt_data_topup_old['date'].max()
            print ('Getting top up update since %s' % (start_topup.strftime('%Y-%m-%d')))
            fs_mkt_data_topup_new=fq.get_ts(tickers_fs,fields_topup,start=fql_date(start_topup),adj=False)
            fs_mkt_data_topup_new=fs_mkt_data_topup_new.stack().reset_index()
            fs_mkt_data_unadj_new=fq.get_ts(tickers_fs,['vwap','px_last'],start=fql_date(start_topup),adj=False)
            fs_mkt_data_unadj_new=fs_mkt_data_unadj_new.stack().reset_index()
            fs_mkt_data_topup=pd.concat([fs_mkt_data_topup_old,fs_mkt_data_topup_new],axis=0).groupby(['date','ticker']).last().reset_index()
            fs_mkt_data_unadj=pd.concat([fs_mkt_data_unadj_old,fs_mkt_data_unadj_new],axis=0).groupby(['date','ticker']).last().reset_index()
            feather.write_dataframe(fs_mkt_data_topup, top_up_path+'fs_mkt_data_topup.feather')
            feather.write_dataframe(fs_mkt_data_unadj, top_up_path+'fs_mkt_data_unadj.feather')
            
        # Dump the last universe, if the tickers_fs is different from last dump (new stocks entered) we will refresh all the data again
        feather.write_dataframe(fs_mkt_data_unadj.groupby('ticker').last()['date'].reset_index(), 
                        top_up_path+'universe_last_run.feather')
           
        
        fs_mkt_data_full=fq.get_ts(tickers_fs,fields_full,start=start_fs)
        
        fs_mkt_data_topup=fs_mkt_data_topup.set_index(['date','ticker']).unstack()
        fs_mkt_data=pd.concat([fs_mkt_data_full,fs_mkt_data_topup],axis=1)
        fs_mkt_data_unadj=fs_mkt_data_unadj.set_index(['date','ticker']).unstack()
        fs_rpt_data=fq.get_ts_reported_fundamental(tickers_fs,['roe'],
                    start=fql_date(self.START_SB if self.direction=='sb' else self.START_NB),
                    rbasis='ANN')
        # no longer use fs ff data
#        fs_ff_data=fq.get_ts_float(tickers_fs,
#                    start=(self.start-pd.tseries.offsets.DateOffset(years=2)).strftime('%m/%d/%Y'),)
        fs_snap=fq.get_snap(tickers_fs,['beta']) #we only get the latest beta to speed up
        #get the bbg data
        # not sure why but bbg sometimes fails for unknown reason
        start=self.start
        fields=['short_name','gics_sector_name','gics_industry_group_name']
        bbg_snap=bdp(tickers_bbg,fields)
        tickers=self._bbg_tickers
        bbg_hist=bdh(tickers,['px_last'],start,um.today_date())
        #get separate bbg free float data
        bbg_ff_data=bdh(tickers_bbg,['eqy_float','eqy_free_float_pct'],start,um.today_date())
        bbg_ff_data=bbg_ff_data.rename(columns={'eqy_float':'ff_shout_sec','eqy_free_float_pct':'ff_pct_sec'})
        bbg_ff_data['ff_pct_sec']=bbg_ff_data['ff_pct_sec']/100
        bbg_ff_data=bbg_ff_data.unstack().T
        bbg_ff_data.columns=bbg_to_fs(bbg_ff_data.columns)
        bbg_ff_data=bbg_ff_data.swaplevel(1,0,0).unstack().swaplevel(1,0,1).sort_index(1)

        #get the ah data using fs with fx locked to be USD.
        #use self.direction to determine index col
        tickers_ah=self.ah_map['HK'].tolist()+self.ah_map['CN'].tolist()
        px_ah=fq.get_ts(tickers_ah,['px_last'],start=start_fs,fx='USD')
        ah_map_dict=self.ah_map.set_index('CN')['HK'].to_dict() if self.direction=='sb' else self.ah_map.set_index('HK')['CN'].to_dict()
        ah_stats=(px_ah['px_last'][self.ah_map['HK' if self.direction=='sb' else 'CN'].tolist()]
            .divide(px_ah['px_last'][self.ah_map['CN'if self.direction=='sb' else 'HK'].tolist()]
                    .rename(columns=ah_map_dict))-1) #discount or premium
        end_time=time.time()
        print ('%s mins used for downloading all data' % ((end_time-start_time)/60))
        #tidy up the data before dump
        fs_mkt_data_unadj=fs_mkt_data_unadj.rename(columns={'vwap':'vwap_unadj','px_last':'px_last_unadj'},level=0)
        fs_rpt_data=fs_rpt_data.fillna(method='ffill')
        fs_data=pd.concat([fs_mkt_data,fs_mkt_data_unadj,fs_rpt_data,
                           bbg_ff_data
                           #fs_ff_data
                           ]
                        ,axis=1)
        fs_data=fs_data.stack()
        fs_data['ff_marcap_sec']=fs_data['px_last']*fs_data['ff_shout_sec']
        fs_data=fs_data.unstack()
        # if any na occurs in bbg_snap it's the ticker's problem so it's safe to just drop
        bbg_snap=bbg_snap.dropna()
        bbg_snap['gics_sector_name']=bbg_snap['gics_sector_name'].map(lambda x: uc.short_sector_name[x])
        bbg_snap['gics_industry_group_name']=bbg_snap['gics_industry_group_name'].map(lambda x: uc.short_industry_name[x])
        bbg_snap.index=bbg_to_fs(bbg_snap.index)
        bbg_snap['beta']=fs_snap['beta']
        #clean index
        clean_index=bbg_hist.unstack().T[self.mkt_index]['px_last'].dropna().index
        fs_data=fs_data.reindex(clean_index)
        bbg_hist=bbg_hist.swaplevel(1,0,0).unstack().reindex(clean_index)
        ah_stats=ah_stats.reindex(clean_index).stack().rename('ah_stats').to_frame().unstack()
        data=pd.concat([fs_data,bbg_hist,ah_stats],axis=1).stack().reset_index().set_index('ticker')
        data=data.merge(bbg_snap,left_index=True,right_index=True,how='left')
        data.index.name='ticker'

        data=data.reset_index()
        feather.write_dataframe(data,self.path+self.MKT_FILE)
        return None

    def update_db(self):
        mkt_data=feather.read_dataframe(self.path+self.MKT_FILE)
        sc_shares_hld=feather.read_dataframe(self.path+self.SC_SHARES_FILE)
        sc_shares_hld['ticker']=bbg_to_fs(sc_shares_hld['ticker'])
        clean_index=mkt_data.groupby('date').last().index
        #tidy up sc_shares and do the shift
        sc_shares_hld=(sc_shares_hld.set_index(['date','ticker']).sort_index().unstack()
                        .reindex(clean_index).fillna(method='ffill').fillna(0))

        if self.direction=='sb':
            sc_shares_hld=sc_shares_hld.shift(-2).fillna(method='ffill')
        #mask the non-connect trading day with 0 flow (or ffill hlds)
        connect_flow_day=mkt_data.set_index(['date','ticker'])['px_last'].unstack()[self.buy_col+self.sell_col].fillna(0).sum(1)
        connect_flow_day[datetime(2014,11,14)]=1 # make sure we have flow on the 1st day of stock connect
        connect_flow_day_mask=connect_flow_day.map(lambda x: True if x==0 else False)
        sc_shares_hld=sc_shares_hld.apply(lambda x: x.mask(connect_flow_day_mask),axis=0)
        sc_shares_hld=sc_shares_hld.fillna(method='ffill')
        sc_shares_chg=sc_shares_hld.diff()
        mkt_data=mkt_data.set_index(['date','ticker']).sort_index()
        db=pd.concat([mkt_data,
                      sc_shares_hld.stack(),
                      sc_shares_chg.stack().rename(columns={'sc_hld_shares':'sc_hld_chg'})],
                    axis=1
                      )
        '''
        Add more columns:
        fx, size (based on last), $flow, $holding, daily_impact (for dropping >100% day),
        marcap_stake, ff_stake,
        stake quintile (marcap and ff)
        '''
        #fx
        fx_field='HKDUSD Curncy' if self.direction=='sb' else 'CNYUSD Curncy'
        fx=db.swaplevel(1,0,0).loc[fx_field]['px_last']
        db=db.reset_index().set_index('date')
        db['fx']=fx
        #size
        size=(db.reset_index().set_index(['date','ticker'])
            .unstack()['marcap_sec'].iloc[-1].dropna().map(lambda x: group_marcap(x/1000/7.8,return_number=False)))
        db=db.reset_index().set_index('ticker')
        db['size']=size
        db['size']=pd.Categorical(db['size'].fillna('nil'),
                  categories=['nil','Micro Cap','Small Cap','Mid Cap','Large Cap','Mega Cap'],
                  ordered=False)
        size_code=(db.reset_index().set_index(['date','ticker'])
            .unstack()['marcap_sec'].iloc[-1].dropna().map(lambda x: group_marcap(x/1000/7.8,return_number=True)))
        db['size_code']=size_code
        # $flow, $holdings, daily_impact
        db['flow_musd']=db['vwap_unadj']*db['sc_hld_chg']*db['fx']/1000000

        #we do holdings musd after dropping the extreme day
        db['impact_daily']=(db['vwap_unadj']*db['sc_hld_chg']/
                            (db['turnover'].map(lambda x: np.nan if x==0 else x)*1000)
                            )
        #do not drop all. only drop certain columns
        cols_to_drop_and_ffill=['sc_hld_shares']
        cols_to_drop_and_zfill=['sc_hld_chg','flow_musd','impact_daily']
        cols_to_fix=cols_to_drop_and_ffill+cols_to_drop_and_zfill
        db['impact_daily_filter']=db['impact_daily'].fillna(0).map(lambda x: True if abs(x)<self.MAX_DAILY_IMPACT else False)
        db=db.reset_index().set_index(['date','ticker'])
        #using mask
        mask=~db['impact_daily_filter']
        db[cols_to_fix]=db[cols_to_fix].mask(mask)
        db[cols_to_drop_and_ffill]=db[cols_to_drop_and_ffill].unstack().fillna(method='ffill').stack()
        db[cols_to_drop_and_zfill]=db[cols_to_drop_and_zfill].unstack().fillna(0).stack()
        db['holdings_musd']=db['px_last_unadj']*db['sc_hld_shares']*db['fx']/1000000
        # stake (marcap and ff)
        db['stake_marcap']=db['px_last_unadj']*db['sc_hld_shares']/(db['marcap_sec']*1000000)
        db['stake_ff']=db['px_last_unadj']*db['sc_hld_shares']/(db['ff_marcap_sec']*1000000)
        # stake quintile (marcap and ff)
        temp_start=db['stake_marcap'].unstack().sum(1).map(lambda x: np.nan if x ==0 else x).dropna().index[1]
        db['stake_marcap_q']=(db['stake_marcap'].unstack().applymap(lambda x: np.nan if x==0 else x)
            .loc[temp_start:]
            .apply(lambda x: pd.qcut(x,5,labels=['Q1','Q2','Q3','Q4','Q5']),axis=1)
            .stack().rename('stake_marcap_q')
            )

        db['stake_ff_q']=(db['stake_ff'].unstack().applymap(lambda x: np.nan if x==0 else x)
            .fillna(method='ffill')
            .loc[temp_start:]
            .apply(lambda x: pd.qcut(x,5,labels=['Q1','Q2','Q3','Q4','Q5']),axis=1)
            .stack().rename('stake_ff_q')
            )
        #stake for AH only
        db['has_ah']=db['ah_stats'].map(lambda x: False if np.isnan(x) else True)
        temp_start=db[db['has_ah']]['stake_marcap'].unstack().sum(1).map(lambda x: np.nan if x ==0 else x).dropna().index[1]
        db['stake_marcap_q_ah']=(db[db['has_ah']]['stake_marcap'].unstack().applymap(lambda x: np.nan if x==0 else x)
            .loc[temp_start:]
            .apply(lambda x: pd.qcut(x,5,labels=['Q1','Q2','Q3','Q4','Q5']),axis=1)
            .stack().rename('stake_marcap_q')
            )
        db['stake_ff_q_ah']=(db[db['has_ah']]['stake_ff'].unstack().applymap(lambda x: np.nan if x==0 else x)
            .fillna(method='ffill')
            .loc[temp_start:]
            .apply(lambda x: pd.qcut(x,5,labels=['Q1','Q2','Q3','Q4','Q5']),axis=1)
            .stack().rename('stake_ff_q')
            )
        db=db.reset_index()
        #do some rename of columns
        db=db.rename(columns={'gics_sector_name':'sector','gics_industry_group_name':'industry'})
        #do some fix
        db=db.set_index(['date','ticker'])
        db['roe']=db['roe'].unstack().fillna(method='ffill').stack()
        db=db.reset_index()
        #output
        feather.write_dataframe(db,self.path+self.DB_FILES)
        return None

    def ml_get_attribute(self,input_matrix,method='sma',window=63,mask_natural_zero=True):
        db=self.db_clean
        holdings_matrix=db['holdings_musd']
        to_matrix=((db.stack()['turnover']*db.stack()['fx']/1000).unstack()
                    .applymap(lambda x: np.nan if x==0 else x))
        mask_holdings=(holdings_matrix.applymap(lambda x: np.nan if x==0 else x)
                        .applymap(lambda x: True if np.isnan(x) else False))
        mask_to=to_matrix.applymap(lambda x: True if np.isnan(x) else False)

        if method=='sma':
            if mask_natural_zero:
                res=(input_matrix
                    .mask(mask_holdings)
                    .applymap(lambda x: np.nan if x==0 else x)
                    .rolling(window,min_periods=1).mean()
                    .mask(mask_to).stack())
            else:
                res=(input_matrix
                    .mask(mask_holdings)
                    #.applymap(lambda x: np.nan if x==0 else x)
                    .rolling(window,min_periods=1).mean()
                    .mask(mask_to).stack())
        elif method=='ema':
            if mask_natural_zero:
                res=(input_matrix
                    .mask(mask_holdings)
                    .applymap(lambda x: np.nan if x==0 else x)
                    .ewm(span=window,min_periods=1).mean()
                    .mask(mask_to).stack())
            else:
                res=(input_matrix
                    .mask(mask_holdings)
                    #.applymap(lambda x: np.nan if x==0 else x)
                    .ewm(span=window,min_periods=1).mean()
                    .mask(mask_to).stack())
        elif method=='first':
            if mask_natural_zero:
                res=(input_matrix
                    .mask(mask_holdings)
                    .applymap(lambda x: np.nan if x==0 else x)
                    .shift(window)
                    .fillna(method='bfill',limit=window) # we do this bfill as shift alone cannot deal with min_periods
                    .mask(mask_to).stack())
            else:
                res=(input_matrix
                    .mask(mask_holdings)
                    #.applymap(lambda x: np.nan if x==0 else x)
                    .shift(window)
                    .fillna(method='bfill',limit=window)
                    .mask(mask_to).stack())
        elif method=='last':
            if mask_natural_zero:
                res=(input_matrix
                    .mask(mask_holdings)
                    .applymap(lambda x: np.nan if x==0 else x)
                    #.shift(window)
                    .mask(mask_to).stack())
            else:
                res=(input_matrix
                    .mask(mask_holdings)
                    #.applymap(lambda x: np.nan if x==0 else x)
                    #.shift(window)
                    .mask(mask_to).stack())
        elif method=='count': # note that shift cannot deal with min_periods
            if mask_natural_zero:
                res=(input_matrix
                    .mask(mask_holdings)
                    .applymap(lambda x: np.nan if x==0 else x)
                    .rolling(window).count()
                    .mask(mask_to).stack())
            else:
                res=(input_matrix
                    .mask(mask_holdings)
                    #.applymap(lambda x: np.nan if x==0 else x)
                    .rolling(window).count()
                    .mask(mask_to).stack())
        return res




    def notebook_display_1_aggregate_color(self):
        macro=self.get_macro()
        data=macro.copy()
        look_back_dict={5:'1w',21:'1m',63:'3m'}
        look_backs=[5,21,63]
        res=pd.DataFrame(index=look_backs,columns=['flow','since'])
        for look_back in look_backs:
            data_i=data['Net flow'].iloc[-look_back:]
            res.at[look_back,'since']=data_i.index[0].strftime('%Y/%m/%d')
            res.at[look_back,'flow']=round(data_i.sum(),1)
        res=res.rename(index=look_back_dict)
        res.index.name='period'
        res=res.reset_index()
        res['idx']=res[['period','since']].apply(lambda x: '%s-(%s)' % (x['period'],x['since']),axis=1)
        res=res.set_index('idx')
        res.at['YTD','flow']=data['Net flow'].resample('A').sum().iloc[-1]
        to_plot=res['flow']
        fig,ax=ud.easy_plot_quick_subplots((1,1),'Southbound net flow (mUSD)' if self.direction=='sb' else 'Northbound net flow (mUSD)')
        to_plot.plot(ax=ax,kind='bar',color=uc.alt_colors_quick_pd_plot[0])
        ud.easy_plot_tidy_up_ax_ticks([ax],dimension='both')
        ud.easy_plot_tick_label_twist(ax,rotation=0,size=12)
        ax.set_xlabel('')
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Holdings (US$ bn)", "Stake (%)", "Flow (US$ mn, 1mma)", "Impact (%)"),specs=[[{}, {}],[{}, {}]],horizontal_spacing = 0.05,vertical_spacing = 0.1,shared_xaxes=True)
        fig.append_trace(go.Scatter(x=macro.index,y=macro['Holdings cost'],name='Holding cost'),row=1, col=1); fig.append_trace(go.Scatter(x=macro.index,y=macro['Holdings MtM'],name='Holding MtM'),row=1, col=1)
        fig.append_trace(go.Scatter(x=macro.index,y=macro['Stake marcap']*100,name='Stake marcap'),row=1, col=2); fig.append_trace(go.Scatter(x=macro.index,y=macro['Stake float']*100,name='Stake float'),row=1, col=2)
        fig.append_trace(go.Scatter(x=macro.index,y=macro['Gross flow'],name='Gross flow'),row=2, col=1); fig.append_trace(go.Scatter(x=macro.index,y=macro['Net flow'],name='Net flow'),row=2, col=1)
        fig.append_trace(go.Scatter(x=macro.index,y=macro['Gross impact']*100,name='Gross impact'),row=2, col=2); fig.append_trace(go.Scatter(x=macro.index,y=macro['Net impact']*100,name='Net impact'),row=2, col=2)
        fig.update_layout(height=700, width=1000,  title={'text': "<b>Aggregate %s color" % (self.direction.upper()),'x':0.5,'y':0.95,'xanchor': 'center','yanchor': 'top'}); fig.show()
    def notebook_display_2_flow_hld_by_sector_industry(self,start,end):
        fig = make_subplots(rows=2, cols=3, subplot_titles=("Sector holdings (US$ bn)", "Sector flow (1m US$m)", "Sector Flow vs holdings (%)", "Industry holdings (US$ bn)", "Industry flow (1m US$m)", "Industry Flow vs holdings (%)", ),
                            specs=[[{'type':'bar'}, {'type':'bar'},{'type':'bar'}],[{'type':'bar'}, {'type':'bar'},{'type':'bar'}]],horizontal_spacing = 0.03,vertical_spacing = 0.16)
        hlds_last=self.get_breakdown(['holdings_musd'],'sector','sum','last',start=start,end=end)['holdings_musd'].sort_values().map(lambda x: int(x/1000));hlds_first=self.get_breakdown(['holdings_musd'],'sector','sum','first',start=start,end=end)['holdings_musd']; flow=self.get_breakdown(['flow_musd'],'sector','sum','sum',start=start,end=end)['flow_musd'].sort_values().map(lambda x: int(x));flow_pct=(flow/hlds_first).map(lambda x: round(x*100,1)).sort_values()
        fig.add_trace(go.Bar(x=hlds_last.index.values,y=hlds_last.values,text=hlds_last.values,textposition='auto'),row=1, col=1);fig.add_trace(go.Bar(x=flow.index.values,y=flow.values,text=flow.values,textposition='auto'),row=1, col=2);fig.add_trace(go.Bar(x=flow_pct.index.values,y=flow_pct.values,text=flow_pct.values,textposition='auto'),row=1, col=3)
        hlds_last=self.get_breakdown(['holdings_musd'],'industry','sum','last',start=start,end=end)['holdings_musd'].sort_values().map(lambda x: int(x/1000));hlds_first=self.get_breakdown(['holdings_musd'],'industry','sum','first',start=start,end=end)['holdings_musd']; flow=self.get_breakdown(['flow_musd'],'industry','sum','sum',start=start,end=end)['flow_musd'].sort_values().map(lambda x: int(x));flow_pct=(flow/hlds_first).map(lambda x: round(x*100,1)).sort_values()
        fig.add_trace(go.Bar(x=hlds_last.index.values,y=hlds_last.values,text=hlds_last.values,textposition='auto'),row=2, col=1); fig.add_trace(go.Bar(x=flow.index.values,y=flow.values,text=flow.values,textposition='auto'),row=2, col=2); fig.add_trace(go.Bar(x=flow_pct.index.values,y=flow_pct.values,text=flow_pct.values,textposition='auto'),row=2, col=3)
        fig.update_layout(height=700, width=1400, showlegend=False, title={'text': "<b>Flow & holdings by sector/industry (from %s to %s)" % (start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d')),'x':0.5,'y':0.95,'xanchor': 'center','yanchor': 'top'}); fig.show()
        
    def notebook_display_3_sankey(self,look_back=21,look_back_longer=42):
        self.get_sankey(direction='buy',look_back=look_back,look_back_longer=look_back_longer,ss_top=1,flow_acceleration_level=0.6,chart_para={'title': '<i><b>From %s to %s</i></b><br>(vs. %s to %s)','height': 600, 'width': 1000, 'thickness': 15, 'font_size': 11, 'font_color': 'black', 'pad': 12,'margin':dict(l=30, r=20, t=40, b=20,pad=4),})
        self.get_sankey(direction='sell',look_back=look_back,look_back_longer=look_back_longer,ss_top=1,flow_acceleration_level=0.6,chart_para={'title': '<i><b>From %s to %s</i></b><br>(vs. %s to %s)','height': 600, 'width': 1000, 'thickness': 15, 'font_size': 11, 'font_color': 'black', 'pad': 12,'margin':dict(l=30, r=20, t=40, b=20,pad=4),})

    def notebook_display_4_sector_flow_monthly(self):
        # 2x5 subplot
        to_plot=self.get_breakdown(['flow_musd'],'sector','sum','nan',start=datetime(2014,11,1),end=um.today_date()).resample('BM').sum()['flow_musd']
        cols=to_plot.columns.values
        cols=np.append(cols,'')
        cols=cols.reshape(3,4)
        fig = make_subplots(rows=3, cols=4, subplot_titles=(tuple(to_plot.columns)),specs=[[{},{},{},{}],[{},{},{},{}],[{},{},{},{}]],horizontal_spacing = 0.05,vertical_spacing = 0.08,shared_xaxes=True)
        for i in np.arange(0,3):
            for j in np.arange(0,4):
                col=cols[i][j]
                if col!='':
                    fig.add_trace(go.Bar(x=to_plot[col].index.map(lambda x: x.strftime('%Y-%m')).values,y=to_plot[col].values),row=int(i+1), col=int(j+1))
        fig.update_layout(height=700, width=1200, showlegend=False,  title={'text': "<b>Monthly flow (US$ mn) history by sector",'x':0.5,'y':0.95,'xanchor': 'center','yanchor': 'top'}); fig.show()

    def notebook_display_5_industry_flow_monthly(self):
        # 4x6 subplot
        to_plot=self.get_breakdown(['flow_musd'],'industry','sum','nan',start=datetime(2014,11,1),end=um.today_date()).resample('BM').sum()['flow_musd']
        cols=to_plot.columns.values
        cols=cols.reshape(4,6)
        fig = make_subplots(rows=4, cols=6, subplot_titles=(tuple(to_plot.columns)),horizontal_spacing = 0.05,vertical_spacing = 0.08,shared_xaxes=True)
        for i in np.arange(0,4):
            for j in np.arange(0,6):
                col=cols[i][j]
                if col!='':
                    fig.add_trace(go.Bar(x=to_plot[col].index.map(lambda x: x.strftime('%Y-%m')).values,y=to_plot[col].values),row=int(i+1), col=int(j+1))
        fig.update_layout(height=800, width=1500, showlegend=False,  title={'text': "<b>Monthly flow (US$ mn) history by sector",'x':0.5,'y':0.95,'xanchor': 'center','yanchor': 'top'}); fig.show()


    def notebook_display_6_prepare_table(self,start,end):
        # get block
        flow=self.db_clean['flow_musd'].loc[start:end].sum().rename('Flow')
        to=(self.db_clean['turnover'].multiply(self.db_clean['fx'])/1000).loc[start:end].sum()
        hlds_first=self.db_clean['holdings_musd'].loc[start:end].iloc[0]
        hlds_last=self.db_clean['holdings_musd'].loc[start:end].iloc[-1].rename('Holdings')

        marcap_stake=self.db_clean['stake_marcap'].loc[start:end].iloc[-1].rename('Marcap stake')
        ff_stake=self.db_clean['stake_ff'].loc[start:end].iloc[-1].rename('FF stake')
        marcap_stake_first=self.db_clean['stake_marcap'].loc[start:end].iloc[0]
        ff_stake_first=self.db_clean['stake_ff'].loc[start:end].iloc[0]
        impact=(flow/to).rename('Impact')
        hlds_chg=(flow/hlds_first).rename('Flow vs. Hlds')
        marcap_stake_chg=(marcap_stake-marcap_stake_first).rename('Marcap stake chg')
        ff_stake_chg=(ff_stake-ff_stake_first).rename('FF stake chg')
        ah_stats=self.db_clean['ah_stats'].loc[start:end].iloc[-1].rename('HA discount')
        df=pd.concat([hlds_last,marcap_stake,ff_stake,flow,impact,hlds_chg,marcap_stake_chg,ff_stake_chg,ah_stats],axis=1)
        df.index.name='Ticker Factset'
        df=df.reset_index()
        df['Ticker BBG']=fs_to_bbg(df['Ticker Factset'])
        df=df.set_index('Ticker BBG')
        bbg_usual=get_bbg_usual_col(df.index.tolist(),add_short_industry=True)
        df=pd.concat([bbg_usual,df],axis=1)
        df=df.drop('Ticker Factset',1)
        df['Flow vs. Hlds']=df['Flow vs. Hlds'].map(lambda x: 999 if np.isinf(x) else x)
        df=df[df['Holdings']!=0]
        self.notebook_table=df.copy()
        self.notebook_table.index.name='Ticker BBG'
        # define the display style
        self._display_style={'MarketCap (US$ bn)':'{:.1f}',
               '3m ADV (US$ mn)':'{:.1f}',
               'Holdings':'{:.1f}',
               'Marcap stake':'{:.2%}',
               'FF stake':'{:.2%}',
               'Flow':'{:.1f}',
               'Impact':'{:.1%}',
               'Flow vs. Hlds':'{:.1%}',
               'Marcap stake chg':'{:.1%}',
               'FF stake chg':'{:.1%}',
               'HA discount':'{:.1%}',
              }
        def black(x):
            return 'color: black'
        self._display_font_color=black
    def notebook_display_7_display_table(self,N=5,by='',sector_focus=[False,'sector name'],
                                         industry_focus=[False,'industry name']):
        df=self.notebook_table.copy()
        if sector_focus[0]:
            print ('Sector focus mode for %s' % (sector_focus[1]))
            df=df[df['Sector']==sector_focus[1]]
        if industry_focus[0]:
            print ('Industry focus mode for %s' % (industry_focus[1]))
            df=df[df['Industry']==industry_focus[1]]
        top=df.sort_values(by=by,ascending=False).iloc[:N]
        top=top.reset_index()
        top.index=top.index+1
        bottom=df.sort_values(by=by,ascending=True).iloc[:N]
        bottom=bottom.reset_index()
        tb=pd.concat([top,bottom],axis=0)
        tb=tb.set_index('Ticker BBG')
        tb=tb[~tb.index.duplicated()].sort_values(by=by,ascending=False)

        return (
        tb.style.format(self._display_style).background_gradient('RdYlGn',subset=[by])
        .applymap(self._display_font_color,subset=[by])
        .set_properties(**{'font-size': '10pt','border-color': 'white','border-style' :'solid','border-width':'1px','text-align':'center' })
        .set_table_styles([{'selector': 'th', 'props': [('font-size', '9pt'),('border-style','solid'),('border-width','1px'),('text-align', 'center')]}])
        )

    def notebook_display_8_single_stock_highlight(self,stocks,start,end,drop_ff=False):
        '''
        Stocks is a list of stock
        '''
        if type(stocks) is not list:
            stocks=[stocks]
        collector=[]
        for stock in stocks:
            to_plot=self.db_clean.swaplevel(1,0,1)[stock][['px_last','stake_marcap','stake_ff','impact_daily','short_name','turnover','fx']].loc[start:end].copy()
            to_plot['turnover_musd']=to_plot['turnover']*to_plot['fx']/1000
            adv=round(to_plot['turnover_musd'].iloc[-63:].mean(),1)
            short_name=to_plot.iloc[0]['short_name']
            to_plot['Impact (1mma)']=to_plot['impact_daily'].rolling(21,min_periods=1).mean()
            nice_name={'px_last':'Price','stake_marcap':'Stake (MarCap)','stake_ff':'Stake (FF)','impact_daily':'Daily turnover impact'}
            fig,axes=ud.easy_plot_quick_subplots((1,2),'Single stock highlight: %s (%s, 3mADV: US$ %s mn)' % (short_name,stock,adv))
            if not drop_ff:
                ud.quick_plot_lineplot(axes[0],to_plot.rename(columns=nice_name),'price vs. stake',['Stake (MarCap)','Stake (FF)'],['Price'])
            else:
                ud.quick_plot_lineplot(axes[0],to_plot.rename(columns=nice_name),'price vs. stake',['Stake (MarCap)'],['Price'])
            ud.easy_plot_pct_tick_label(axes[0],direction='y',pct_format='{:.1%}')
            to_plot.rename(columns=nice_name)[['Daily turnover impact','Impact (1mma)']].plot(ax=axes[1],title='Impact over time',color=[uc.alt_colors_quick_pd_plot[19],uc.alt_colors_quick_pd_plot[20]])
            ud.easy_plot_pct_tick_label(axes[1],direction='y',pct_format='{:.0%}')
            ud.easy_plot_tidy_up_ax_ticks(axes)
            axes[1].set_xlabel('')
            to_plot['stock']=stock
            collector.append(to_plot)
        return pd.concat(collector,axis=0)
    
    # ---- just to reorganize things here
    def get_scraper(self):
        if self.direction=='sb':
            func=download_SB
        else:
            func=download_NB
        return func
    
    def get_nb_universe(self):
        return get_nb_universe
    
    def update_bbg_ccass_id_map(self):
        return update_bbg_ccass_id_map
        

if __name__=='__main__':
    #---- test
    print ('ok')
    sc=STOCK_CONNECT(direction='sb')
    # sc.update_connect_data()
    # sc.update_mkt_data()
    # sc.update_db()
    sc.load_db(add_momentum=True)
    
    
    
#    sc=STOCK_CONNECT(direction='nb',skip_fql=True)
#    sc.load_db()
#    sc=STOCK_CONNECT(direction='sb',skip_fql=True)
#    sc.load_db()
#
#
#    # load the bottom-up holdings and eastmoney holdings, compare
#    hlds_mf=feather.read_dataframe("Z:\\dave\\data\\eastmoney\\hlds.feather")
#    hlds_bbg_path="Z:\\dave\\data\\connect\\southbound\\bottom_up_hlds\\"
#    files=um.iterate_csv(hlds_bbg_path,iterate_others=[True,'.feather'])
#    collector=[]
#    for file in files:
#        hlds_bbg_i=feather.read_dataframe(hlds_bbg_path+'%s.feather' % (file))
#        collector.append(hlds_bbg_i)
#    hlds_bbg=pd.concat(collector,axis=0)
#    hlds_bbg['ticker']=bbg_to_fs(hlds_bbg['ticker'])
#    hlds_bbg['Filing Date']=pd.to_datetime(hlds_bbg['Filing Date'])
#
#    fund_type_to_drop=['qdii','qdii_etf','qdii_index']
#    hlds_mf_last=(hlds_mf[(hlds_mf['location']=='H') & (~hlds_mf['fund_type'].isin(fund_type_to_drop))]
#                .groupby(['asof','ticker'])['shares'].sum().unstack().fillna(method='ffill').iloc[-1])
#    # just use the last data point
#    hlds_mf_last=hlds_mf_last.rename('mutual_fund').to_frame()
#
#    hlds_bbg_last=(hlds_bbg[hlds_bbg['Country']=='China']
#                .groupby(['Filing Date','Institution Type','ticker'])['Amount Held'].sum()
#                .unstack().unstack().fillna(method='ffill').resample('A').last().iloc[-1].unstack())
#
#    hlds_bottom_up=pd.concat([hlds_bbg_last,hlds_mf_last],axis=1)
#
#    hlds_bottom_up['sb']=sc.db_clean['sc_hld_shares'].iloc[-1]
#    hlds_bottom_up['shout']=sc.db_clean['shout_sec'].iloc[-1]*1000000
#
#    hlds_pct=hlds_bottom_up.divide(hlds_bottom_up['shout'],axis='index').drop('shout',1)
#
#
#    to_include=['mutual_fund']#['Investment Advisor','Insurance Company','mutual_fund']
#    to_check=pd.concat([hlds_pct['sb'], hlds_pct[to_include].sum(1).rename('bottom_up')],axis=1)
#
#    fig,ax=ud.easy_plot_quick_subplots((1,1),'scatter')
#    ud.quick_plot_scatterplot(ax,to_check.fillna(0),'','sb',['bottom_up'])
#
#
#
#    sc.notebook_display_8_single_stock_highlight(['700-HK'],datetime(2014,11,1),um.yesterday_date())
#
#    what='roe' # refer to above for available field
#    by='sector' # sector, industry, size, stake_marcap_q, stake_ff_q
#    how_group='median' # sum, mean, median, count
#    how_ts='last' # sum, mean, first, last
#    start=datetime(2019,12,1)
#    end=datetime(2020,1,15)
#    focus=[False,'focus group name']
#
#    test=sc.get_breakdown(what,by,how_group,how_ts,start=start,end=end,focus=focus)
#
#    ### manage database
#    sc.update_connect_data(top_up=True)
#    sc.update_mkt_data()
#    db=sc.update_db()
#    sc.load_db()
#    sc.get_sankey()
#    ### some quick stats
#    macro=sc.get_macro()
#    whats=['roe','pe','pb','ah_stats','div_yield','beta','impact_daily']#['holdings_musd','flow_musd']
#    by='stake_marcap_q'
#    how_group='median'
#    how_ts='nan'
#    start=datetime(2014,1,1)#um.today_date()-24*pd.tseries.offsets.BDay()
#    end=um.today_date()-4*pd.tseries.offsets.BDay()
#
#    res=self.get_breakdown(whats,by,how_group,how_ts,
#                     start=start,end=end,
#                     )
































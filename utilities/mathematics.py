# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import utilities.misc as um
import pdb
from sklearn import datasets, linear_model


import statsmodels.formula.api as smf
import statsmodels.api as sm


from pyfinance import ols as pf_ols

from pykalman import KalmanFilter

from datetime import datetime


#---- ML dataset
def get_iris():
    from utilities.constants import root_path_data
    df= pd.read_excel(root_path_data+'misc\\iris_data.xlsx')
    speices_map={'setosa':0,'versicolor':1,'virginica':2}
    df['y']=df['y'].map(lambda x: speices_map[x])
    return df


def get_ebm_initialized(func_type='regressor',interactions=0,n_jobs=-1,
                        # binning='uniform'
                       ):
    '''
    * if using the latest version, the binning strategy is hidden and will no longer be available for control; so always transform the data
    * maybe able to do feature_types
    
    func_type can be regressor or classifier
    
    interactions: Interactions to be trained on.
                Either a list of lists of feature indices, or an integer for number of automatically detected interactions.
                Interactions are forcefully set to 0 for multiclass problems.
                
    binning: Method to bin values for pre-processing. 
    Choose "uniform" (old version default), "quantile" (new version default), or "quantile_humanized".
    
    quantile:
    https://www.i2tutorials.com/what-do-you-mean-by-binning-in-machine-learning-what-are-the-differences-between-fixed-width-binning-and-adoptive-binning/
    
    uniform:
    Since we transform the inputs by ranking, using uniform bining can better maintain the interpretability
    However if inputs are not uniformzied, it is worthwhile considering quantile binning
    
    BUG!:
    not sure why and how it can happen, but when at spyder environment or cmd environment (so not notebook enrivonment),
    setting n_job!=1 can leads to code using up too much memory
    this maybe different from differnt PC setups
    '''
    from interpret.glassbox import ExplainableBoostingRegressor
    from interpret.glassbox import ExplainableBoostingClassifier
    func_to_use=ExplainableBoostingRegressor if func_type=='regressor' else ExplainableBoostingClassifier
    # Do not use this one if you want to play around with model parameters.
    # however the default setting should be good enough for most of the cases
    # the argument may change in the new release
    ebm_model=func_to_use(
        interactions=interactions,
        n_jobs=n_jobs,
        # binning=binning,
        )

    
    

    return ebm_model

def get_ebm_coef_rank_and_shape(ebm):
    '''
    this applies to both regressor and classifier
    ebm can only handle 2 outcome fitting for now
    '''
    ebm_global=ebm.explain_global()
    shape_collector=[]
    coefs_rank=pd.Series(index=ebm_global.data()['names'],data=ebm_global.data()['scores']).rename('importance').to_frame()
    for predictor in coefs_rank.index:
        i=ebm_global.feature_names.index(predictor)
        # note that names will always have one more data point than score as it's the bins
        shape=pd.Series(index=ebm_global.data(i)['names'][1:],data=ebm_global.data(i)['scores']).rename('shape').to_frame()
        shape['predictor']=predictor
        shape.index.name='x_range_actual'
        shape=shape.reset_index()
        shape.index.name='x_range'
        shape_collector.append(shape)
    shape_all=pd.concat(shape_collector,axis=0)
    return coefs_rank,shape_all


def rolling_max_dd(data, window_size, min_periods=1,method='pct',updown='down'):
    """Compute the rolling maximum drawdown of `x`.
    Dave copied the main part from the site below:
    http://stackoverflow.com/questions/21058333/compute-rolling-maximum-drawdown-of-pandas-series
    data should be in data frame form
    `min_periods` should satisfy `1 <= min_periods <= window_size`.

    Returns an 1d array with length `len(x) - min_periods + 1`.
    to get the full period mdd, set the window_size to be euqal to (or larger than) the length of the data, and get the last point of the output
    """
    #input validation
    if method not in ['abs','pct']:
        print ('invalid method input')
        return None
    if updown not in ['up','down']:
        print ('invalid updown input')
        return None
    def windowed_view(x, window_size):
        from numpy.lib.stride_tricks import as_strided
        """Creat a 2d windowed view of a 1d array.
        `x` must be a 1d numpy array.
        `numpy.lib.stride_tricks.as_strided` is used to create the view.
        The data is not copied.
        Example:
        >>> x = np.array([1, 2, 3, 4, 5, 6])
        >>> windowed_view(x, 3)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6]])
        """
        y = as_strided(x, shape=(x.size - window_size + 1, window_size),
                       strides=(x.strides[0], x.strides[0]))
        return y
    output=pd.DataFrame(index=data.index,columns=data.columns)
    for column in data.columns:
        x=data[column].values
        #this min period is designed so that we can have mdd at the begining of the period.
        if min_periods < window_size:
            pad = np.empty(window_size - min_periods)
            pad.fill(x[0])
            x = np.concatenate((pad, x))
            y = windowed_view(x, window_size)
            if updown=='down':
                running_max_y = np.maximum.accumulate(y, axis=1) #key is the accumulate function
                if method=='abs':
                    dd = y - running_max_y
                elif method=='pct':
                    dd = (y - running_max_y)/running_max_y
                else:
                    print ('unknown method')
                    return None
                output.at[output.index,column]=dd.min(axis=1)
            elif updown=='up':
                running_min_y = np.minimum.accumulate(y, axis=1)
                if method=='abs':
                    dd = y - running_min_y
                elif method=='pct':
                    dd = (y - running_min_y)/running_min_y
                else:
                    print ('unknown method')
                    return None
                output.at[output.index,column]=dd.max(axis=1)
            else:
                print ('unknown updown')
                return None
    return output



def quick_polynomial_formula_builder(x,order):
    '''
    This is for single variable polynominal of order n
    Can pass the resulting string to quick_linear_regression_fit
    '''
    xs=[]
    if order>=2:
        for o in np.arange(order,1,-1):
            xs.append('np.power(%s , %s)' % (x,o))
    xs.append(x)
    return xs


def quick_linear_regression_fit(data,y,xs=['X1','X2','np.power(X1,2)'],
                        sort_exog_input=False
                        #ONLY use True for single variable regression.
                        #This is for plotting of fitted higher degree polynominal
                        ):
    '''
    We output the regression results:
        model specification Y=X1+X2 etc
        Y-hat, beta, intercept, p-value, R-squared
    We can add control for interception later
    '''

    formula='%s ~' % (y)
    for x in xs:
        formula=formula+' %s +' % (x)
    formula=formula[:-2]
    res= smf.ols(formula=formula, data=data).fit()
    exog_input={}
    for x in xs:
        if x[:3]!='np.':
            exog_input[x]=data[x].values if not sort_exog_input else data[x].sort_values(ascending=False).values
    y_hat=res.predict(exog=exog_input)
    anova=sm.stats.anova_lm(res,typ=2)
    rsquared=res.rsquared
    paras=res.params
    pvalues=res.pvalues
    reg_res=pd.concat([paras.rename('paras'),
                       pvalues.rename('pvalues')],axis=1)
    result_dict={'reg_res':reg_res,
                 'y_hat':y_hat,
                 'rsquared':rsquared,
                 'anova':anova,
                 }
    return result_dict


def rolling_regression(df,y,xs,window,expanding_starting_mode=[False,'minimum data point']):
    '''
    Using pyfinance for fast and efficient rolling regression.
    Return object which contains all the regression info for use
    *Cannot handle nan, need fillna
    If turnning off expanding_starting_mode then it cannot handle number of data points <= window length
    If turning on expanding_starting_mode, the function will do expanding regression until reaching the window length
        - For now if we turn on expanding_starting_mode the output will be beta not the rolling regression object
    
    commonly used output
    .alpha
    .beta
    .predicted --> multiindex df, needs reset_index and groupby
    .std_err   --> standard error of estimate, degree of freedom taken care of already
    .pvalue_alpha
    .pvalue_beta
    .rsq_adj
    '''
    if not expanding_starting_mode[0]:
        if len(df)<=window:
            print ('number of data points need to be larger than window length, returning False')
            return False
        else:
            return pf_ols.PandasRollingOLS(y=df[y],x=df[xs],window=window)
    else:
        min_point=expanding_starting_mode[1]
        if len(df)<=min_point:
            print ('number of data points need to be larger than minimum data point length, returning False')
            return False
        else:
            # use a loop to go through the rows with expansion window until we reach the window length, where we start the rolling
            collector=[]
            for i,dt_i in enumerate(df.index):
                df_i=df.loc[:dt_i]
                if len(df_i)<min_point:
                    pass
                elif len(df_i)>=min_point and len(df_i)<=window:
                    beta_i=pf_ols.PandasRollingOLS(y=df_i[y],x=df_i[xs],window=len(df_i)-1).beta
                    collector.append(beta_i.iloc[[-1]])
                else:
                    beta_rest=pf_ols.PandasRollingOLS(y=df[y],x=df[xs],window=window).beta
                    collector.append(beta_rest)
                    break
                    
            beta=pd.concat(collector).reset_index().groupby('date').last()
            return beta
            
            


def rolling_avg_pariwise_correlation(data,window_size,min_period=1):
    '''
    This func outputs the rolling average pairwise correlation of time series
    It will drop the diagonal
    Input requires pct return
    '''
    df=data.copy()
    df.index.name=None
    df.columns.name=None
    df=df.rolling(window_size,min_periods=min_period).corr().dropna().stack().rename('corr').reset_index()
    corr=df[df['level_1']!=df['level_2']].reset_index().groupby('level_0').mean()['corr']
    corr.index.name=None
    return corr

def dollar_return_to_pct_return(pnl_dollar_cumu,notional=1):
    '''
    input can be df or series
    We calculate the cumulative return of constant notional LS portfolio
    Assuming capital is 1 for long 1 and short 1
    Pnl (not mtm value so start with 0 or -bps) needs to be in dollars
    Can handle front loaded transaction cost
    '''
    pnl=pnl_dollar_cumu/notional
    initial_dt=pnl.index[0]-pd.tseries.offsets.BDay()
    pnl.loc[initial_dt]=0
    pnl=pnl.sort_index()
    mtm=pnl.diff().fillna(0)
    pct_daily=mtm/((pnl+1).shift(1))
    return pct_daily.loc[pnl_dollar_cumu.index[0]:]

def get_sharpe(pct_daily):
    return pct_daily.mean()/pct_daily.std()*np.sqrt(252)


def get_stats(cumu_perf,include_summary_text=False):
    stats=pd.concat([
                get_sharpe(cumu_perf.pct_change()).rename('sharpe'),
                (rolling_max_dd(cumu_perf,len(cumu_perf)+1).iloc[-1]).rename('MDD'),
                (cumu_perf.iloc[-1].map(lambda x: x**(252/len(cumu_perf)))-1).rename('CAGR'),
                ],axis=1)
    if include_summary_text:
        stats['text']=stats.apply(lambda x:
            'Sharpe:%s;CAGR:%s;MDD:%s' % (round(x['sharpe'],2),'{:.1%}'.format(x['CAGR']),'{:.1%}'.format(x['MDD'])),axis=1)
    stats['MDD']=pd.to_numeric(stats['MDD'])
    return stats



def diag(x):
    # input is df or series
    # get diagonal of a square matrix x,
    # or construct diagonal matrix from vector x
    # maintain the df or series labeling
    if type(x) is pd.Series:
        D=pd.DataFrame(data=np.diag(x),index=list(x.index),columns=list(x.index))
        D.index.name='index';D.columns.name='column'
    elif type(x) is pd.DataFrame:
        D=pd.Series(data=np.diag(x),index=list(x.index))
    else:
        print ('Input needs to be either series or dataframe')
        return False
    return D

def inv(x):
    # use numpy to inverse matrix but keep the pandas labels
    # x needs to be an invertable matrix
    # note that we use pinv instead of inv. Google the math for difference.
    res= pd.DataFrame(data=np.linalg.pinv(x.values),
                      index=list(x.columns),columns=list(x.index))
    res.index.name='index';res.columns.name='column'
    return res


def grouped_regression(df,by,y,xs_input,res_type):
    '''
    Can add other res_type if needed, currently we have residual and fitted
    '''
    if res_type not in ['residual','fitted']:
        print ('res_type needs to be in %s' % ( ','.join(['residual','fitted'])))
    res=df.copy()
    if res_type=='residual':
        output=(res.groupby(by)
                .apply(lambda x: x[y]-quick_linear_regression_fit(x,y,xs_input)['y_hat'].values)
                .reset_index().set_index('level_1')[y])
    elif res_type=='fitted':
        output=(res.groupby(by)
                .apply(lambda x: x[y]*0+quick_linear_regression_fit(x,y,xs_input)['y_hat'].values)
                .reset_index().set_index('level_1')[y])
    return output

def KF_beta(df,y,x,add_ols=[False,252]):
    '''
    This one do univariate regression. Expand later if needed
    Note that the hyper-parameters below is suitable for return/log-price.
    Can't guarantee they still make sense if the input scale is hugely different
    useful links:
        http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
        https://arxiv.org/ftp/arxiv/papers/1204/1204.0375.pdf (how does the prediction->update cycle work)
        https://www.quantstart.com/articles/Dynamic-Hedge-Ratio-Between-ETF-Pairs-Using-the-Kalman-Filter/ (how to formulate in regression context)
    '''
    print ('The function KF_beta is not recommended, please use KF_betas')
    
    data_to_use=df.copy()
    obs_input=data_to_use[[x]].copy()
    obs_input['intercept']=1
    obs_input=[x.reshape(1,2) for x in obs_input.values]

    kf = KalmanFilter(

        initial_state_mean=np.array([1,0]), # 2nd items is for intercept which is assumed to be 0
        initial_state_covariance=1e-5*np.eye(2),
        transition_matrices=np.eye(2),
        transition_covariance=1e-5* np.eye(2),
        observation_matrices=obs_input,
        observation_covariance=1e-3,# smaller --> more responsive, very large --> flat line. the larger the oversevation error is, the more cautious the model is to update state
    )

    state_means, state_covs = kf.filter(data_to_use[y].values)

    res=pd.DataFrame(index=data_to_use.index,columns=['beta','intercept'],data=state_means)

    if add_ols[0]:
        rolling_beta=rolling_regression(data_to_use,y,[x],add_ols[1]).beta
        res['rolling_beta']=rolling_beta[x]
    return res


def KF_betas(df,y,xs,add_ols=[True,504],
            hp_isc=1e-5, # initial_state_covariance
            hp_tc=1e-5, # transition_covariance
            hp_oc=1e-3, # observation_covariance. Smaller --> more responsive, very large --> flat line
            auto_estimate_initial_beta=[False,0.1] # use a given proportion of the initial data to estimate initial beta using regression. 
                                                     # Not recommended unless the initial estimate happens to be close to the long time mean
            ):
    '''
    Multivariate KF-based beta estimation
    The parameter auto_estimate_initial_beta is for fun, better not use it
    For initial level different from ols, the KF will adjust automatically based on the hyper parameters. If it looks too bad we can include longer history
    
    The default hyper parameter levels seems to work better only for daily return series
    '''
    if auto_estimate_initial_beta[0]:
        proportion=auto_estimate_initial_beta[1]
        df=df.iloc[int(len(df)*proportion)+1:]
        df_est=df.iloc[:int(len(df)*proportion)]
        reg_est=rolling_regression(df_est,y,xs,int(len(df)*proportion)-1).beta.iloc[-1][xs]
    
    data_to_use=df.copy()
    obs_input=data_to_use[xs].copy()
    obs_input['intercept']=1 # indication of intercept, not level of intercept
    obs_input=[x.reshape(1,len(xs)+1) for x in obs_input.values]

    kf = KalmanFilter(

        initial_state_mean=np.array([1]* len(xs)+[0]) if not auto_estimate_initial_beta[0] else np.array(reg_est.values.tolist()+[0]), # initial betas are 1 and initial intercept is 0
        initial_state_covariance=hp_isc*np.eye(len(xs)+1),
        transition_matrices=np.eye(len(xs)+1),
        transition_covariance=hp_tc* np.eye(len(xs)+1),
        observation_matrices=obs_input,
        observation_covariance=hp_oc,# The larger the oversevation error is, the more cautious the model is to update state
    )

    state_means, state_covs = kf.filter(data_to_use[y].values)
    res=pd.DataFrame(index=data_to_use.index,columns=xs+['intercept'],data=state_means)
    res.columns.name='ticker'

    if add_ols[0]:
        rolling_reg_res=rolling_regression(data_to_use,y,xs,add_ols[1])
        rolling_beta=rolling_reg_res.beta
        rolling_beta['intercept']=rolling_reg_res.alpha
        # we create multil columns for easier comparison
        rolling_beta=rolling_beta.stack().rename('value').to_frame()
        rolling_beta['type']='RR'

        res=res.stack().rename('value').to_frame()
        res['type']='KF'
        res=pd.concat([res,rolling_beta],axis=0).reset_index().set_index(['date','ticker','type'])['value'].unstack().unstack()
        
    return res




def get_cutoff_level_for_cumu_coverage(input_s,cutoff_level_pct,ascending=False):
    '''
    Default is to sort value from high to low
    return cut_off_level,series_details
    '''
    series_details=input_s.sort_values(ascending=ascending).dropna().rename('value').to_frame()
    series_details['cumu_value']=series_details['value'].cumsum()
    series_details['cumu_value_norm']=series_details['cumu_value']/series_details['cumu_value'].iloc[-1]
    cut_off_level=series_details[series_details['cumu_value_norm']<=cutoff_level_pct]['value'].iloc[-1]
    return cut_off_level,series_details


def cap_existing_wgt(x,cap=0.1,iteration=1000):
    '''
    Apply this func to rows of a dataframe of weight
    '''
    wgt_i=x.copy()

    for i in np.arange(0,iteration,1):

        wgt_i=wgt_i.map(lambda x: min(x,cap))

        res_tot=1-wgt_i.sum()
        if res_tot!=0:
            res_i=wgt_i[wgt_i<cap]
            wgt_i=pd.concat([wgt_i,(res_i/res_i.sum())*res_tot],axis=1,sort=True).fillna(0).sum(1)
        else:
            break
    return wgt_i.map(lambda x: np.nan if x==0 else x)


def get_distance(df,x,y,method='euclidean',rebase=True):
    '''
    method can be dtw or euclidean
    '''
    if rebase:
        df=df/df.iloc[0]
    x_s=df[x]
    y_s=df[y]
    if method=='euclidean':
        return np.linalg.norm(x_s-y_s,ord=2)
    elif method=='dtw':
        from dtw import dtw
        return dtw(x_s, y_s,distance_only=True).distance
    else:
        print ('%s not supported!' % (method))
        return np.nan
def get_rolling_distance(df,x,y,lookback,method='euclidean',rebase=True):
    '''
    we need to write loop to get rolling distance
    '''
    i_s=np.arange(lookback,len(df),1)
    res=pd.DataFrame(index=df.index,columns=['distance'])
    for i in i_s:
        df_i=df.iloc[i-lookback:i]
        res.at[df.index[i],'distance']=get_distance(df_i,x,y,method=method,rebase=rebase)
    res['method']=method
    return df.join(res)


def get_expanding_distance(df,x,y,method='euclidean',rebase=True):
    '''
    we need to write loop
    '''
    i_s=np.arange(1,len(df),1)
    res=pd.DataFrame(index=df.index,columns=['distance'])
    for i in i_s:
        df_i=df.iloc[0:i+1]
        res.at[df.index[i],'distance']=get_distance(df_i,x,y,method=method,rebase=rebase)
    res['method']=method
    return df.join(res)


def mean_reversion_half_life(series):
    '''
    we use method here:
    https://victor-bernal.weebly.com/uploads/5/3/6/9/53696137/projectcalibration.pdf

    Note that don't use it with pandas rolling, too slow
    Better using rolling_regression func
    May lead to negative hl, can process (e.g. drop) outside of the function
    '''
    series=series.rename('rt').to_frame()
    series['rt_next']=series['rt'].shift(-1)
    reg_res=quick_linear_regression_fit(series.dropna(),'rt_next',['rt'],)
    theta=(1-reg_res['reg_res']['paras']['rt'])
    hl=np.log(2)/theta
    return hl

def rolling_mean_reversion_half_life(series,window):
    series=series.rename('rt').to_frame()
    series['rt_next']=series['rt'].shift(-1)
    series=series.dropna()
    res=rolling_regression(series,'rt_next',['rt'],min(window,len(series)-1))
    return res.beta['rt'].map(lambda x: np.log(2)/(1-x))

def normalize_ls_port(port_input):
    '''
    input is concated port matrix (axis=1) with +ve and -ve weight, with potential overlapping names
    output will be the net and re-normalized matrix
    '''
    port_input.index.name='date'
    port_input.columns.name='ticker'
    port=port_input.stack().rename('wgt').reset_index().groupby(['date','ticker']).sum()['wgt'].unstack()
    res=pd.concat([
        port.applymap(lambda x: x if x>=0 else np.nan).apply(lambda x: x/x.sum(),axis=1).stack(),
        port.applymap(lambda x: x if x<0 else np.nan).apply(lambda x: x/abs(x.sum()),axis=1).stack(),
        ],axis=0).unstack()
    return res


def get_RSI(data,window,method='sma'):
    '''    
    https://www.macroption.com/rsi-calculation/
    https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi
    
    we can choose between 'sma' and 'ema'
    input data should be a dataframe with datetime index
    input data should already be in return or change
    '''
    current_gain=data[data>0].fillna(0)
    current_loss=data[data<0].fillna(0).abs()

    avg_gain=current_gain.rolling(window,min_periods=1).mean().fillna(0)
    avg_loss=current_loss.rolling(window,min_periods=1).mean().fillna(0)

    if method=='ema':
        wgt=2/(window+1)
        # need to use a loop
        for i,dt in enumerate(avg_gain.index):
            if i>=1:
                avg_gain.iloc[i]=wgt*current_gain.iloc[i]+(1-wgt)*avg_gain.iloc[i-1]
                avg_loss.iloc[i]=wgt*current_loss.iloc[i]+(1-wgt)*avg_loss.iloc[i-1]


    rsi=100-100/(avg_gain.divide(avg_loss)+1)

    return rsi
    
    
    
    
    



if __name__ == "__main__":
    print ('ok')
    
    import feather
    path="C:\\Users\\davehanzhang\\python_data\\misc\\test.feather"
    to_fit_i=feather.read_dataframe(path)
    
    ebm=get_ebm_initialized()
    ebm.fit(to_fit_i[['score']],to_fit_i['alpha'])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
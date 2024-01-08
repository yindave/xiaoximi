# -*- coding: utf-8 -*-

from matplotlib.lines import Line2D
import os

#--- root data path
root_path_data="C:\\Users\\davehanzhang\\python_data\\"

#---- path
compo_path=root_path_data+"cache\\index_compo\\%s.feather"
compo_path_daily=root_path_data+"cache\\index_compo_daily\\%s.feather"


python_dump=root_path_data+"%s\\%s_%s.csv"
python_dump_png=root_path_data+"%s\\%s_%s.png"


#---- proxy
use_proxy=False
proxy_to_use={'https':'hk-proxy.ap.hedani.net:8090',
              'http':'hk-proxy.ap.hedani.net:8090'}

#---- email smtp
smtp='smtp-amg-ap.hedani.net'
gmail_login='no longer needed!'


#---- distribution list
dl_csv_path='do this later'
dl={
    'self':['davehanzhang@gmail.com'],
    }


#---- nice BBG sector industry name
short_sector_name={'Materials':'Matls',
                   'Industrials':'Indust',
                   'Information Technology':'IT',
                   'Real Estate':'R Estate',
                   'Consumer Discretionary':'Cons Disc',
                   'Financials':'Finls',
                   'Consumer Staples':'Cons Stpl',
                   'Utilities':'Utilities',
                   'Health Care':'H Care',
                   'Energy':'Energy',
                   'Telecommunication Services':'Telecom',
                   'Communication Services':'Comm Svc',}

short_industry_name={'Capital Goods':'Cap Goods',
                    'Real Estate':'R Estate',
                    'Consumer Durables & Apparel':'Cons Durables',
                    'Diversified Financials':'Diversified Fins',
                    'Energy':'Energy',
                    'Utilities':'Utilities',
                    'Household & Personal Products':'Personal Prods',
                    'Transportation':'Transportation',
                    'Materials':'Materials',
                    'Automobiles & Components':'Auto & Compo',
                    'Media & Entertainment':'Media & Entertain',
                    'Commercial & Professional Serv':'Prof Serv',
                    'Health Care Equipment & Servic':'Health Care Equip',
                    'Food, Beverage & Tobacco':'Food Bev Toba',
                    'Consumer Services':'Cons Serv',
                    'Pharmaceuticals, Biotechnology':'Pharma & Bio',
                    'Banks':'Banks',
                    'Retailing':'Retailings',
                    'Semiconductors & Semiconductor':'Semiconductors',
                    'Software & Services':'Software',
                    'Technology Hardware & Equipmen':'Hardware',
                    'Insurance':'Insurance',
                    'Telecommunication Services':'Telecom Serv',
                    'Food & Staples Retailing':'Food Stpls Retail'
                    }

#---- BBG price index vs. total return index
total_ret_index_map={
                    'HSCI Index':'HSCI Index', # can't find total return index for HSCEI
                    'SHCOMP Index':'SHCOMP Index', # history is too short for the total ret index
                    'SZCOMP Index':'SZCOMP Index', # history is too short for the total ret index
                    'SHCOMP_L Index':'SHCOMP Index', # history is too short
                    'SZCOMP_L Index':'SZCOMP Index', # history is too short
                    'TPX Index':'TPXDDVD Index',
                    'TPX_L Index':'TPXDDVD Index',
                    'HSCEI Index':'HSI 21 Index',
                    'HSI Index':'HSI 1 Index',
                    'XIN9I Index':'TXIN9IC Index',
                    'SHSZ300 Index':'CSIR0300 Index',
                    'SH000905 Index':'CSIR0905 Index',
                     }
total_ret_index_map_r={}
for k,v in total_ret_index_map.items():
    total_ret_index_map_r[v]=k


#---- short  bbg ticker
def nice_ticker_name(x):
    return x[:x.find(' Equity')]

#---- Plot color, style, legend

linestyles=['-','--','-.',':']
markers=['o','d','v','s','p','+']
alt_colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
            (0,0,175), (0,135,255),
             (0,60,120), (0,30,60),
            (13,19,4), (70,70,70),
            ]

alt_colors_quick=[]
for i in range(len(alt_colors)):
    r, g, b = alt_colors[i]
    alt_colors_quick.append( (r / 255., g / 255., b / 255.) )

alt_colors_quick_pd_plot=[]
for i in range(len(alt_colors)):
    r, g, b = alt_colors[i]
    alt_colors_quick_pd_plot.append( (r / 255., g / 255., b / 255. ,1) )


#legend
quick_boxplot_color_map={'mean':alt_colors_quick[0],'median':alt_colors_quick[2],
           'now':alt_colors_quick[4],'previous':alt_colors_quick[0],'flier':'r'}
quick_boxplot_marker_map={'mean':'o','median':'s',
           'now':'d','previous':'d','flier':'o'}


def get_quick_boxplot_legend(mode,previous_label='x-days ago',has_outlier=False):
    if mode=='mean_vs_median':
        if not has_outlier:
            quick_box_plot_legend=[
                        Line2D([0], [0],
                               markerfacecolor=quick_boxplot_color_map['mean'],
                               marker=quick_boxplot_marker_map['mean'],
                               markersize=10,
                               markeredgecolor=quick_boxplot_color_map['mean'],alpha=0.8,
                               linewidth =0,
                               label='Mean'),
                        Line2D([0], [0],
                               markerfacecolor='none',
                               marker=quick_boxplot_marker_map['median'],
                               markersize=10,
                               markeredgecolor=quick_boxplot_color_map['median'],alpha=0.8,
                               linewidth =0,
                               label='Median'),
                          ]
        else:
            quick_box_plot_legend=[
                    Line2D([0], [0],
                           markerfacecolor=quick_boxplot_color_map['mean'],
                           marker=quick_boxplot_marker_map['mean'],
                           markersize=10,
                           markeredgecolor=quick_boxplot_color_map['mean'],alpha=0.8,
                           linewidth =0,
                           label='Mean'),
                    Line2D([0], [0],
                           markerfacecolor='none',
                           marker=quick_boxplot_marker_map['median'],
                           markersize=10,
                           markeredgecolor=quick_boxplot_color_map['median'],alpha=0.8,
                           linewidth =0,
                           label='Median'),
                    Line2D([0], [0],
                           markerfacecolor='none',
                           marker=quick_boxplot_marker_map['flier'],
                           markersize=5,
                           markeredgecolor=quick_boxplot_color_map['flier'],alpha=0.8,
                           linewidth =0,
                           label='Outlier'),
                      ]
    elif mode=='now_vs_previous':
        if not has_outlier:
            quick_box_plot_legend=[
                        Line2D([0], [0],
                               markerfacecolor=quick_boxplot_color_map['now'],
                               marker=quick_boxplot_marker_map['now'],
                               markersize=10,
                               markeredgecolor=quick_boxplot_color_map['now'],alpha=0.8,
                               linewidth =0,
                               label='Now'),
                        Line2D([0], [0],
                               markerfacecolor='none',
                               marker=quick_boxplot_marker_map['previous'],
                               markersize=10,
                               markeredgecolor=quick_boxplot_color_map['previous'],alpha=0.8,
                               linewidth =0,
                               label=previous_label),
                          ]
        else:
            quick_box_plot_legend=[
                        Line2D([0], [0],
                               markerfacecolor=quick_boxplot_color_map['now'],
                               marker=quick_boxplot_marker_map['now'],
                               markersize=10,
                               markeredgecolor=quick_boxplot_color_map['now'],alpha=0.8,
                               linewidth =0,
                               label='Now'),
                        Line2D([0], [0],
                               markerfacecolor='none',
                               marker=quick_boxplot_marker_map['previous'],
                               markersize=10,
                               markeredgecolor=quick_boxplot_color_map['previous'],alpha=0.8,
                               linewidth =0,
                               label=previous_label),
                        Line2D([0], [0],
                               markerfacecolor='none',
                               marker=quick_boxplot_marker_map['flier'],
                               markersize=5,
                               markeredgecolor=quick_boxplot_color_map['flier'],alpha=0.8,
                               linewidth =0,
                               label='Outlier'),
                          ]
    elif mode=='now_vs_median':
        if not has_outlier:
            quick_box_plot_legend=[
                        Line2D([0], [0],
                               markerfacecolor=quick_boxplot_color_map['now'],
                               marker=quick_boxplot_marker_map['now'],
                               markersize=10,
                               markeredgecolor=quick_boxplot_color_map['now'],alpha=0.8,
                               linewidth =0,
                               label='Now'),
                        Line2D([0], [0],
                               markerfacecolor='none',
                               marker=quick_boxplot_marker_map['median'],
                               markersize=10,
                               markeredgecolor=quick_boxplot_color_map['median'],alpha=0.8,
                               linewidth =0,
                               label='Median'),
                          ]
        else:
            quick_box_plot_legend=[
                    Line2D([0], [0],
                           markerfacecolor=quick_boxplot_color_map['now'],
                           marker=quick_boxplot_marker_map['now'],
                           markersize=10,
                           markeredgecolor=quick_boxplot_color_map['now'],alpha=0.8,
                           linewidth =0,
                           label='Now'),
                    Line2D([0], [0],
                           markerfacecolor='none',
                           marker=quick_boxplot_marker_map['median'],
                           markersize=10,
                           markeredgecolor=quick_boxplot_color_map['median'],alpha=0.8,
                           linewidth =0,
                           label='Median'),
                    Line2D([0], [0],
                           markerfacecolor='none',
                           marker=quick_boxplot_marker_map['flier'],
                           markersize=5,
                           markeredgecolor=quick_boxplot_color_map['flier'],alpha=0.8,
                           linewidth =0,
                           label='Outlier'),
                      ]
    return quick_box_plot_legend


def get_quick_scatter_legend(label,color_i,show_fit=[True,'label for fit']):

    legend_output=[
                Line2D([0], [0],
                       color=alt_colors_quick[color_i*2],
                       markerfacecolor=alt_colors_quick[color_i*2],
                       marker='o',
                       markersize=10,
                       markeredgecolor=alt_colors_quick[color_i*2],
                       alpha=1.0,
                       linewidth =2 if show_fit[0] else 0,
                       label=label),]


    return legend_output


#---- Nice month and quarter
nice_month={1:'Jan',
            2:'Feb',
            3:'Mar',
            4:'Apr',
            5:'May',
            6:'Jun',
            7:'Jul',
            8:'Aug',
            9:'Sep',
            10:'Oct',
            11:'Nov',
            12:'Dec',
            }
nice_month_rev={'Jan':1,
            'Feb':2,
            'Mar':3,
            'Apr':4,
            'May':5,
            'Jun':6,
            'Jul':7,
            'Aug':8,
            'Sep':9,
            'Oct':10,
            'Nov':11,
            'Dec':12,
            }
nice_quarter={1:'Q1',
              2:'Q2',
              3:'Q3',
              4:'Q4',}


#---- get python path
def get_python_path():
    try:
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        user_paths = []
    python_path=user_paths[0]+'\\'
    return python_path








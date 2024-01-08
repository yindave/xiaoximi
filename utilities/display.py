# -*- coding: utf-8 -*-


import utilities.constants as uc
import utilities.mathematics as umath
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
import utilities.misc as um

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import pdb



'''
Frequently used plt function in case you forget

tick rotation, size etc (shouldn't be used very often)
    ax.tick_params(axis="x", labelsize=8, rotation=20)

tick label alignment etc
    use easy_tick_label

add vertical/horizontal line
    ax.axvline()
    ax.axhline()

annotation
    use easy_annotate function

turnoff ax
    ax.axis('off')

ax label
    ax.set_xlabel()
    ax.set_ylabel()


filling between
    ax.fill_between(x,y1,y2) fill between 2 horizonal lines
    ax.fill_betweenx
    or axvspan


Using pandas plot for the following types:
bar/column, area, distribution, heatmap, parallel coordinate

'''




class HTML_Builder():
    '''
    In case of pd.df, no need to convert it to html
    to enable precise formatter, set precise_formatter =[('column name',"{:.1f}")]
    highlight_col only supported when quick_formatter or precise_formatter are turned on
    '''        
    
    def __init__(self,customize_font_style=[False,'"font-family: Calibri, sans-serif;"']):
        self.body=''
        self.pictures={}
        self.picture_id=1
        self.body_with_png_path='' #this one is for pdf
        
        self.font_style='<font style="'+customize_font_style[1]+'">' if customize_font_style[0] else ''
        self.font_style_content_only=customize_font_style[1] if customize_font_style[0] else ''
        
    def __repr__(self):
        return 'HTML String Builder'
    
    def insert_row_for_pdf(self,row_number=1):
        for i in range(0 , row_number+1):
            self.body_with_png_path += '<br>'
            
    def insert_content_for_pdf(self,html_string,bold=False):
        html_string=self.font_style+html_string
        if not bold:        
            self.body_with_png_path += html_string
        else:
            self.body_with_png_path += '<br><big><b>%s</b></big>' % (html_string)
    
    def insert_title(self,title,bold=True,ignore_body_for_pdf=False):
        title=self.font_style+title
        if bold:
            self.body +='<br><big><big><big><b>%s</b></big></big></big>' % (title)
            if not ignore_body_for_pdf:
                self.body_with_png_path +='<br><big><big><big><b>%s</b></big></big></big>' % (title)
        else:
            self.body +='<br><big>%s</big>' % (title)
            if not ignore_body_for_pdf:
                self.body_with_png_path +='<br><big><big><big>%s</big></big></big>' % (title)
    def insert_body(self,body,bold=True,ignore_body_for_pdf=False):
        body=self.font_style+body
        if bold:
            self.body +='<br><big><big><b>%s</b></big></big>' % (body)
            if not ignore_body_for_pdf:
                self.body_with_png_path +='<br><big><big><b>%s</b></big></big>' % (body)
        else:
            self.body +='<br>%s' % (body)
            if not ignore_body_for_pdf:
                self.body_with_png_path +='<br><big>%s</big>' % (body)
        
    def insert_table(self,table,table_name,quick_formatter=False,precise_formatter_col=[],highlight_col=[],use_nice_name={},highlight_negative=[]):
        '''
        Note that if you want to highlight the cols with nice name you need to input the nice name
        can apply to both styler or the df
        
        
        highlight_col input type is [(col name,color)], one column at a time
        (clumsy) highlight_col needs precise_formatter_col to contain the same col name
        
        '''
        def _local_quick_formatter(x):
            try:
                if abs(x)<1:
                    return "{:.1%}".format(x)
                elif abs(x)>1000:
                    return "{:,}".format(round(x,0))
                else:
                    return "{:.1f}".format(x)
            except TypeError:
                return x
        def _local_precise_formatter(x,col_formatter):
            col_collection=[col[0] for col in col_formatter]
            col_formatter_dict={col[0]:col[1] for col in col_formatter}
            if x.name in col_collection:
                formatter=col_formatter_dict[x.name]
                if formatter =="{:,}":
                    return x.map(lambda y: y if np.isnan(y) else formatter.format(round(y,1)))
                else:
                    return x.map(lambda y: y if np.isnan(y) else formatter.format(y))
            else:
                return x
        def _local_highligh_col(x,highlight_col):
            col_collection=[col[0] for col in highlight_col]
            col_color_dict={col[0]:col[1] for col in highlight_col}
            if x.name in col_collection:
                color=col_color_dict[x.name]
                return x.map(lambda y: 'background-color: %s' % (color))
            else:
                return x.map(lambda y: '')

        table_name=self.font_style+table_name
        
        self.body +='<p><b><big><big>%s</big></big></b>' % (table_name)
        
        self.body_with_png_path +='<p><b><big><big>%s</big></big></b>' % (table_name)
        
        
        if not quick_formatter:
            if len(precise_formatter_col)==0:
                self.body +=table.to_html()
                return None
            else:
                if len(highlight_col)==0:
                    styler=(table.apply(lambda x: _local_precise_formatter(x,precise_formatter_col))
                            .rename(columns=use_nice_name)
                            .style.set_table_attributes('border="1" class="dataframe" style="%s float:left"' % (self.font_style_content_only)))
                    styler_for_pdf=(table.apply(lambda x: _local_precise_formatter(x,precise_formatter_col))
                            .rename(columns=use_nice_name))
                           
                else:
                    styler=(table.apply(lambda x: _local_precise_formatter(x,precise_formatter_col))
                            .rename(columns=use_nice_name)
                            .style.set_table_attributes('border="1" class="dataframe" style="%s float:left"' % (self.font_style_content_only))
                            .apply(lambda x: _local_highligh_col(x,highlight_col)))
                    styler_for_pdf=(table.apply(lambda x: _local_precise_formatter(x,precise_formatter_col))
                            .rename(columns=use_nice_name))
                    
        else:
            styler=(table.applymap(_local_quick_formatter)
                        .rename(columns=use_nice_name)
                        .style.set_table_attributes('border="1" class="dataframe" style="%s float:left"' % (self.font_style_content_only)))
            styler_for_pdf=(table.applymap(_local_quick_formatter)
                        .rename(columns=use_nice_name))
                        
        
        
#        if len(highlight_negative)!=0:
#            styler.applymap(subset=[highlight_negative])
        
        self.body +=styler.render()
        self.body_with_png_path +=styler_for_pdf.to_html()

    
    def insert_picture(self,pic_path):
        '''
        This formula inserts pictures for both email and pdf
        '''        
        self.body += """<br><img src="cid:image%s"><br/>""" % (self.picture_id)
        self.pictures['image%s' % (self.picture_id)]=pic_path
        self.picture_id+=1
        
        self.body_with_png_path += """<br><img src="%s"><br/>""" % (pic_path)

    def insert_picture_for_pdf_only(self,pic_path):
        
        self.body_with_png_path += """<br><img src="%s"><br/>""" % (pic_path)


        
def display_pretty_color_map():
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    to_plot=uc.alt_colors_quick
    #dummy_df=pd.DataFrame(index=np.arange(0,len(to_plot)),columns=np.arange(0,len(to_plot)))
    for i, color_code in enumerate(to_plot):
        ax.axhline(i,color=color_code,label=str(i),linewidth=10)
    ax.legend(bbox_to_anchor=(1.15,1))
    return None


def display_corr_table(corr):
    styler=corr.applymap(lambda x: round(x,2)).style#
    styler.set_table_styles([dict(selector="th", props=[("text-align", "right"),("width","80px")])]   )
    styler.set_properties(**{'width': '80px','height': '50px','text-align': 'right'}).background_gradient(cmap='coolwarm')
    return styler



def easy_plot_quick_subplots(size_tuple,suptitle,
                    sharex=False,sharey=False,
                    customize_size=[False,['tuple','pad','hpad','wpad']]):
    '''
    size_tuple supported includes:
        (1,1),(1,2),(2,1),(1,3),(3,1)
        (2,2),
        (3,2),
        (1,6)
    '''
    size_map={(1,1):[(8,5),2.5,0,0],
              (1,2):[(15,5),4.5,0,3.5],
              (1,6):[(30,5),4.5,0,3.5],
              (2,1):[(8,10),4.5,3.5,0],
              (3,1):[(8,12),4.5,2.5,0],
              (1,3):[(18,5),4,0,2.5],
              (1,4):[(20,4.5),4,0,2],
              (2,2):[(15,10),4.5,3,2.5],
              (3,2):[(12,10),4.5,3,2.5],
              }
    try:
        size_para=size_map[size_tuple]
    except KeyError:
        for k in size_map.keys():
            print ('Currently only support %s-%s' % (k[0],k[1]))
        return None,None
    if customize_size[0]:
        size_para=customize_size[1]

    fig,axes=plt.subplots(size_tuple[0],size_tuple[1],figsize=size_para[0],
                          sharex=sharex,sharey=sharey
                          )
    plt.tight_layout(pad=size_para[1],h_pad=size_para[2],w_pad=size_para[3])
    plt.suptitle(suptitle,fontsize=16,fontweight='bold')

    return fig,axes

def easy_plot_tidy_up_ax_ticks(axes,dimension=1):
    if dimension==2:
        for AX in axes:
            for ax in AX:
                ax.tick_params(axis="both", which="both", bottom=False, top=False,left=False,right=False,)
    else:
        for ax in axes:
            ax.tick_params(axis="both", which="both", bottom=False, top=False,left=False,right=False,)

def easy_plot_line_legend(axes_list,ncol=1,loc='best',
                                use_bbox_to_anchor=[False,(1,0.5)],font=12):
    all_lines=[]
    for ax in axes_list:
        all_lines=all_lines+ax.get_lines()
    labs=[line.get_label() for line in all_lines]
    if not use_bbox_to_anchor[0]:
        axes_list[0].legend(all_lines,labs,loc=loc,ncol=ncol ,framealpha=0.5,fancybox=True,fontsize=font).get_frame().set_edgecolor('k')
    else:
        axes_list[0].legend(all_lines,labs,loc=loc,ncol=ncol,bbox_to_anchor=(use_bbox_to_anchor[1]) ,framealpha=0.5,fancybox=True,fontsize=font).get_frame().set_edgecolor('k')


def easy_plot_pct_tick_label(ax,direction='x',pct_format='{:.0%}'):
    vals=ax.get_xticks() if direction=='x' else ax.get_yticks()
    if direction=='x':
        ax.set_xticklabels([pct_format.format(x) for x in vals])
    elif direction=='y':
        ax.set_yticklabels([pct_format.format(x) for x in vals])
    elif direction=='both':
        vals=ax.get_yticks()
        ax.set_yticklabels([pct_format.format(x) for x in vals])
        vals=ax.get_xticks()
        ax.set_xticklabels([pct_format.format(x) for x in vals])
    else:
        print ('unknown direction')

def easy_plot_tick_label_twist(ax,direction='x',ha='center',va='center',color='k',size=10,rotation=90):
    plt.setp(ax.xaxis.get_majorticklabels() if direction=='x' else ax.yaxis.get_majorticklabels(),
             ha=ha,va=va, rotation=rotation,size=size,color=color)

def easy_plot_annotate(ax,text,xy,xytext,
                       rad=0.2, #this can be negative
                       color=uc.alt_colors_quick[6],
                       text_color='w',
                       lw=2,
                       fontsize=12,
                       ha='center',
                       va='center'
                       ):

    '''
    both xy and xy_text are in data point
    rad can be negative to control the shape of the arrow
    keep the other kwargs unless it's necessary to change
    '''
    arrowprops=dict(arrowstyle='->', color=color,lw=lw,connectionstyle='arc3,rad=%s' % (rad))
    bbox=dict(boxstyle='round,pad=0.2', facecolor=color,edgecolor='none', alpha=0.9)
    ax.annotate(text,xy=xy,xytext=xytext,arrowprops=arrowprops,bbox=bbox,
                fontsize=fontsize,ha=ha, va=va,color=text_color)


def quick_plot_lineplot(ax,to_plot,title,primary_cols,secondary_cols,
                          color_counter=0):
    #for timeseries plot
    #pass [] to secondary cols if it's empty
    #use color_counter to start from other alternative quick color
    #return the as_sec if we need to change the format
    has_scondary=True if len(secondary_cols)!=0 else False
    if has_scondary:
        ax_sec=ax.twinx()
    ax.set_title(title)
    for i,col in enumerate(primary_cols):
        ax.plot(to_plot.index,to_plot[col],c=uc.alt_colors_quick[i*2+color_counter],label=col)
        last_i=i*2+color_counter
    if has_scondary:
        for i,col in enumerate(secondary_cols):
            ax_sec.plot(to_plot.index,to_plot[col],c=uc.alt_colors_quick[last_i+(i+1)*2],label='%s (RHS)' % (col))
    ax.tick_params(axis='x',rotation=30)
    easy_plot_tidy_up_ax_ticks([ax,ax_sec] if has_scondary else [ax])
    easy_plot_line_legend([ax,ax_sec] if has_scondary else [ax])


    if has_scondary:
        ax_sec.grid(False)
        return ax_sec


def quick_plot_boxplot(ax,to_plot,
                       style='mean_vs_median', # or 'now_vs_previous' or 'now_vs_median'
                       title='title',
                       whis=np.inf, #inputs can be np.inf, 1.5, or [5,95]
                       auto_labels=[True,'list of user defined labels if False'],
                       vert=True,
                       previous_point_input=[5,'previous label'],
                       scatter_size=100
                       ):
    '''
    whis=np.inf, #inputs can also be np.inf, 1.5, or [5,95]
    '''
    all_styles=['mean_vs_median','now_vs_previous','now_vs_median']
    if style not in all_styles:
        print ('Unsupported sytle input %s. Use the following instead' % (style))
        for s in all_styles:
            print (s)


    color_map=uc.quick_boxplot_color_map
    marker_map=uc.quick_boxplot_marker_map
    labels=to_plot.columns.tolist() if auto_labels[0] else auto_labels[1]
    flierprops=dict(markerfacecolor='none', marker='o', markersize=5,
                      linestyle='-', markeredgecolor='r',)
    medianprops = dict(linestyle='-', linewidth=0, color='k') #hide the default median
    previous_point=previous_point_input[0]
    previous_point_label=previous_point_input[1]
    scatter_size=scatter_size
    means=to_plot.mean()
    medians=to_plot.median()
    lasts=to_plot.iloc[-1]


    ax.set_title(title)
    filtered_data =to_plot.values.T.tolist()
    filtered_data_to_use=[]
    for i_row in filtered_data:
        clean_row=[i for i in i_row if str(i)!='nan']
        filtered_data_to_use.append(clean_row)

    ax.boxplot(filtered_data_to_use,
                       whis=whis,labels=labels,vert=vert,
                        flierprops=flierprops, medianprops=medianprops
                        )
    if style=='mean_vs_median':
        ax.scatter(np.arange(1,len(labels)+1),means.values,
                    facecolor=color_map['mean'], marker=marker_map['mean'], s=scatter_size,
                    edgecolor=color_map['mean'],alpha=0.8,
                    linewidth =2)
        ax.scatter(np.arange(1,len(labels)+1),medians.values,
                    facecolor='none', marker=marker_map['median'], s=scatter_size,
                    edgecolor=color_map['median'],alpha=0.8,
                    linewidth =2)
        #do the legend
        legend_elements = uc.get_quick_boxplot_legend(style,has_outlier=False if whis is np.inf else True)
        ax.legend(handles=legend_elements, loc='best',ncol=1 ,
                framealpha=0.5,fancybox=True,fontsize=11).get_frame().set_edgecolor('k')
    elif style=='now_vs_previous':
        previous=to_plot.iloc[-1*previous_point]
        ax.scatter(np.arange(1,len(labels)+1),lasts.values,
                    facecolor=color_map['now'], marker=marker_map['now'], s=scatter_size,
                    edgecolor=color_map['now'],alpha=1.0,
                    linewidth =2)
        ax.scatter(np.arange(1,len(labels)+1),previous.values,
                    facecolor='none', marker=marker_map['previous'], s=scatter_size,
                    edgecolor=color_map['previous'],alpha=0.8,
                    linewidth =2)
        #do the legend
        legend_elements = uc.get_quick_boxplot_legend(style,previous_label=previous_point_label,has_outlier=False if whis is np.inf else True)
        ax.legend(handles=legend_elements, loc='best',ncol=1 ,
                framealpha=0.5,fancybox=True,fontsize=11).get_frame().set_edgecolor('k')
    elif style=='now_vs_median':

        ax.scatter(np.arange(1,len(labels)+1),lasts.values,
                    facecolor=color_map['now'], marker=marker_map['now'], s=scatter_size,
                    edgecolor=color_map['now'],alpha=1.0,
                    linewidth =2)
        ax.scatter(np.arange(1,len(labels)+1),medians.values,
                    facecolor='none', marker=marker_map['median'], s=scatter_size,
                    edgecolor=color_map['median'],alpha=0.8,
                    linewidth =2)
        #do the legend
        legend_elements = uc.get_quick_boxplot_legend(style,previous_label=previous_point_label,has_outlier=False if whis is np.inf else True)
        ax.legend(handles=legend_elements, loc='best',ncol=1 ,
                framealpha=0.5,fancybox=True,fontsize=11).get_frame().set_edgecolor('k')

    easy_plot_tidy_up_ax_ticks([ax])

    return None


def quick_plot_scatterplot(ax,to_plot,title,y,all_x,
                       show_regression=[True,1,2.5],#whether fit/order/lw
                       return_reg_results=False, #if True return a nested dict else empty dict
                       turnoff_legend=False,
                       add_vhline=[False,'x_level','y_level']
                       ):

    '''
    all_x will be a list
    Can return the regression stats: beta, intercept, pvalue and r2, in one df
    Do individual scatter highlight separately
    The function will tolerate nan input
    '''
    ax.set_title(title)

    fit_line=show_regression[0]
    order=show_regression[1]
    lw=show_regression[2]

    reg_results={}
    legend_elements=[]
    for i, x in enumerate(all_x):
        ax.scatter(to_plot[x],to_plot[y],marker='o',color=uc.alt_colors_quick[i*2],
                        s=8,alpha=0.4)
        if fit_line:
            xs=umath.quick_polynomial_formula_builder(x,order)
            reg_dict=umath.quick_linear_regression_fit(to_plot[[y,x]].dropna(),y,xs=xs,
                                                       sort_exog_input=True)
            ax.plot(to_plot[x].sort_values(ascending=False).dropna(),reg_dict['y_hat']
            ,color=uc.alt_colors_quick[i*2],lw=lw)
            reg_results[x]=reg_dict
        #legend
        legend_elements_i=uc.get_quick_scatter_legend(x,i,show_fit=[fit_line,'Fitted' if order==1 else 'Fitted (%s)' % (order)])
        legend_elements=legend_elements+legend_elements_i


        ax.legend(handles=legend_elements, loc='best',ncol=1 ,
                framealpha=0.5,fancybox=True,fontsize=11).get_frame().set_edgecolor('k')
        if turnoff_legend:
            ax.get_legend().remove()
    easy_plot_tidy_up_ax_ticks([ax])
    if add_vhline[0]:
        ax.axvline(add_vhline[1],c='k',lw=1,ls='--')
        ax.axhline(add_vhline[2],c='k',lw=1,ls='--')
    if return_reg_results:
        return reg_results
    else:
        return None







if __name__ == "__main__":
    print('OK')
    # easy_plot_quick_subplots((1,6),'test')

#----# test Sankey digaram
#    path="M:\\DerivativesStrategy\\Team_Members\\Dave Yin\\Data\\sankey_test\\"
#    df=pd.DataFrame.from_csv(path+'sb_test_2.csv').reset_index()
#    sankey_plot(df,
#                chart_para={'path':path+'sb_test_2.png',
#                            'title':'sbss',
#                            'height':900,
#                            'width':1600,
#                            'pad':10,
#                            'thickness':20,
#                            'font_size':11,
#                            'font_color':'red'},
#                account='gmail')





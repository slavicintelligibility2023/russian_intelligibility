import pandas as pd
import numpy as np

from math import sqrt
from scipy.stats import kstest
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.stats import ttest_ind

from sklearn.linear_model import LinearRegression
from statsmodels.stats.weightstats import ztest
from scipy.stats import pearsonr

import matplotlib as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text as plText
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import seaborn as sns
import matplotlib.transforms as transforms

from ipywidgets import interact, interactive_output, Select, HBox, FloatSlider
from IPython.display import display

import copy

def drawCompleteness(test_id, users_answers):
    dat = users_answers[(~users_answers["answer"].isna())&(users_answers["test_id"]==test_id)]
    dat = dat[["id_user", "answer"]].groupby("id_user").count()
    axes = dat.hist()[0, 0]
    axes.set_xlabel('Number of Non-Empty answers')
    axes.set_ylabel('Number Participants')

def calcCIFrame(data2: pd.DataFrame, field: str) -> pd.DataFrame:
    data = data2.groupby([field]).agg(['count', 'mean', 'std'])
    data['ci'] = 1.96 * data[('mean_mark', 'std')] / np.sqrt(data[('mean_mark', 'count')])
    data['ci_min'] = data[('mean_mark', 'mean')] - data['ci']
    data['ci_max'] = data[('mean_mark', 'mean')] + data['ci']
    data = data.drop([('mean_mark', 'std'), 'ci'], axis=1)
    data.index = [i for i in data.index]
    data.columns = ['count', 'mean', 'ci_min', 'ci_max']
    return data

def moveCollection(ax, coll_no, size, direction):
    offset = transforms.ScaledTranslation(size, 0, ax.figure.dpi_scale_trans)
    trans = ax.collections[coll_no].get_transform()
    if direction == 'left':
        ax.collections[coll_no].set_transform(trans - offset)
    elif direction == 'right':
        ax.collections[coll_no].set_transform(trans + offset)
    return offset

def moveLines(ax, offset, direction='left'):
    for line in ax.lines:
        trans = line.get_transform()
        if direction == 'left':
            line.set_transform(trans - offset)
        elif direction == 'right':
            line.set_transform(trans + offset)

def orderIndex(data: pd.DataFrame, order: list) -> pd.DataFrame:
    o2 = []
    for ind in order:
        if ind in data.index:
            o2.append(ind)
            
    data4 = pd.DataFrame()
    for ind in o2:
        for col in data.columns:
            data4.loc[ind, col] = data.loc[ind, col]
            
    return data4

# Drawing data on the number of partisipants.
def show_langs(test: str, user_data, axes):
    if test == 'All':
        data2 = user_data[['parallel_lang', 'mean_mark']]
    else:
        data2 = user_data[['parallel_lang', 'mean_mark']][user_data.test_id == int(test)]
    
    order = ['No Parallel Text', 'Ukranian', 'Belarussian', 'Bulgarian', 'Serbian', 'Slovene', 
             'Polish', 'Slovak', 'Czech']
    order2= ['Control', 'Ukrainian', 'Belarus.', 'Bulgarian', 'Serbian', 'Slovene', 
             'Polish', 'Slovak', 'Czech']
    
    data = calcCIFrame(data2, 'parallel_lang')
    data4 = orderIndex(data, order)
    display(data4.T)
    
    axes.clear()

    axes.set_ylim(0, 1)
    sns.pointplot(x='parallel_lang', y="mean_mark", data=data2, 
                  palette="Dark2", hue='parallel_lang', 
                  markers="x", err_kws={'linewidth': 2.0}, capsize=0.1, order=order,
                  linestyle='none', ax=axes
                 ) 
#    sns.pointplot(x='parallel_lang', y="mean_mark", data=data2, palette="Dark2",# color="#000000", 
#                  markers="x", order=order, errwidth=2., capsize=0.1, 
#                  join=False, ax=axes) 

    data3 = data2.groupby('parallel_lang').median()
    data3_2 = data2.groupby('parallel_lang').mean()
    data3['lang'] = data3.index
    sns.pointplot(x='lang', y="mean_mark", data=data3, 
                  palette="Dark2", hue='parallel_lang', 
                  markers="*", order=order, err_kws={'linewidth': 2.0}, 
                  capsize=0.1, linestyle='none', ax=axes
                 ) 
#   sns.pointplot(x='lang', y="mean_mark", data=data3, palette="Dark2",
#                  markers="*", order=order, errwidth=2., capsize=0.1, join=False, ax=axes) 
    
    sns.swarmplot(x='parallel_lang', y="mean_mark", hue='parallel_lang', 
                  data=data2, order=order, 
                  size=3, palette="Set2", 
                  linewidth=0, ax=axes, alpha=0.5
                 ) 
#    sns.swarmplot(x='parallel_lang', y="mean_mark", data=data2, order=order, 
#                  size=3, palette="Set2", #color="#444444", 
#                  linewidth=0, ax=axes, alpha=0.5) 
    
    offset = moveCollection(axes, 0, 12/72., "left")
    moveLines(axes, offset)

    patches = [Rectangle((-0.5, data4.iloc[0, 2]), 10, data4.iloc[0, 3] - data4.iloc[0, 2], alpha = 0.3, edgecolor='#0011DD'),]
    pc = PatchCollection(patches, alpha = 0.1, facecolor='#0011DD')
    axes.add_collection(pc)  

    patches = [Rectangle((0.5, 0), 2, 1, alpha = 0.1, edgecolor='#FF00FF')]
    pc = PatchCollection(patches, alpha = 0.02, facecolor='#FF00FF')
    axes.add_collection(pc)
    patches = [Rectangle((2.5, 0), 3, 1, alpha = 0.1, edgecolor='#00FF00')]
    pc = PatchCollection(patches, alpha = 0.02, facecolor='#00FF00')
    axes.add_collection(pc)
    patches = [Rectangle((5.5, 0), 3, 1, alpha = 0.1, edgecolor='#FF0000')]
    pc = PatchCollection(patches, alpha = 0.02, facecolor='#FF0000')
    axes.add_collection(pc)
    
    for i in range(1, 9):
        axes.text(i+0.1, data3_2.loc[order[i], 'mean_mark']-0.005, 
                  f"+{data3_2.loc[order[i], 'mean_mark'] - data3_2.loc[order[0], 'mean_mark']:5.2}", 
                  color=sns.palettes.get_colormap('Dark2').colors[i%8], fontsize=12)
    
    axes.text(1.3, 0.05, 'East', color='#777777', fontsize=14)
    axes.text(3.75, 0.05, 'South', color='#777777', fontsize=14)
    axes.text(6.75, 0.05, 'West', color='#777777', fontsize=14)

    for item in [axes.title, axes.xaxis.label, axes.yaxis.label] + \
                 axes.get_xticklabels() + axes.get_yticklabels():
        item.set_fontsize(12)
    
    plt.ylabel("Correctness of Answers")
    axes.set_xticks(axes.get_xticks(), order2)
    plt.xlabel('Parallel Language')

def show_langs_by_tests(user_data, axes):
    data2 = user_data[['parallel_lang', 'mean_mark', 'test_id']]
    data = data2.groupby(['parallel_lang', 'test_id']).agg(['count', 'mean', 'std'])
    data['ci'] = 1.96 * data[('mean_mark', 'std')] / np.sqrt(data[('mean_mark', 'count')])
    data['ci_min'] = data[('mean_mark', 'mean')] - data['ci']
    data['ci_max'] = data[('mean_mark', 'mean')] + data['ci']
    data = data.drop([('mean_mark', 'std'), 'ci'], axis=1)
    data.columns = ['count', 'mean', 'ci_min', 'ci_max']
    data['test_id'] = [i[1] for i in data.index]
    data['language'] = [i[0] for i in data.index]
    data.index = range(len(data.index))

    order = ['No Parallel Text', 'Ukranian', 'Belarussian', 'Bulgarian', 'Serbian', 'Slovene', 
             'Polish', 'Slovak', 'Czech']
    order2= ['Cortrol', 'Ukrainian', 'Belarus.', 'Bulgarian', 'Serbian', 'Slovene', 
             'Polish', 'Slovak', 'Czech']

    fig2, axes = plt.subplots(1, 1, figsize=(12,9), num=101)

    markers = ['o', 'X', 'd', '^', '+', 's', 'p', '*', 'v']
    colors = ['#000000', '#111111', '#222222', '#333333', '#444444', '#555555', '#666666', '#777777', '#888888']

    for i, lang in enumerate(order[::-1]):
        sns.pointplot(x='test_id', y="mean_mark", hue='test_id', 
                      data=data2[data2.parallel_lang==lang], 
                      palette="Dark2",# color=colors[i], 
                      markers=markers[i], err_kws={'linewidth': 1.5}, 
                      capsize=0.1, order=range(1,7), 
                      linestyle='none', ax=axes
                     ) 
#        sns.pointplot(x='test_id', y="mean_mark", data=data2[data2.parallel_lang==lang], palette="Dark2",# color=colors[i], 
#                      markers=markers[i], errwidth=1.5, capsize=0.1, order=range(1,7), 
#                      join=False, ax=axes) 
#        for j in range(i+1):
#            offset = moveCollection(axes, j, 6/70., "right")
        offset = transforms.ScaledTranslation(6/70., 0, axes.figure.dpi_scale_trans)
        moveLines(axes, offset, "right")
    axes.set_ylim(0, 1)
    axes.set_xticks(axes.get_xticks(), 
                    [' '*20+str(i) for i in range(1, 7)]
                   )
    axes.set_xlim(0, 6)
    plt.ylabel('Avg. Correctness among All Tests')
    plt.xlabel('Test No')

    axes.grid()

    _ = axes.legend(handles=[Line2D([], [], marker=m, color='black') for m in markers[::-1]], labels=order2,
                    loc = 'lower right', ncol=2, title='Languages')

    fig2.savefig('img_res/Fig_2_distribution_users_tests.png', dpi = 600)
    return fig2

def calc_p_data(data2, test, order, pwrs):
    data = data2.groupby(['parallel_lang']).agg(['count', 'mean', 'std'])
    data['se'] = data[('mean_mark', 'std')] / np.sqrt(data[('mean_mark', 'count')])
    data['ci_min'] = data[('mean_mark', 'mean')] - data['se']
    data['ci_max'] = data[('mean_mark', 'mean')] + data['se']
    data.columns = ['count', 'mean', 'std', 'se', 'ci_min', 'ci_max']
    
    for lang in order:
        if not (test == 6 and lang == 'Slovak'):
            pwrs.append(('pval', test, lang, 
                         ttest_ind(data2[data2.parallel_lang==lang]['mean_mark'], 
                                   data2[data2.parallel_lang=='No Parallel Text']['mean_mark'])[1]))
            pwrs.append(('power', test, lang, 
                         norm.cdf(1.645-abs(data['mean'][lang]-data['mean']['No Parallel Text'])/(data['se'][lang]))))

def draw_p_values(ax, data, subset_name, order, order2, y_label):
    ax.set_ylim(0, 1)
    colors_light = [(2/3+c[0]/3, 2/3+c[1]/3, 2/3+c[2]/3) for c in sns.palettes.get_colormap('Pastel2').colors]
    sns.barplot(x='language', y="one", hue='test', 
                data=data[data['kind']==subset_name], 
                palette=colors_light[1:],
                order=order, ax=ax)
    sns.barplot(x='language', y="value", hue='test', 
                data=data[data['kind']==subset_name], 
                palette="Dark2",
                order=order, ax=ax)
#    sns.barplot(x='language', y="one", data=data[data['kind']==subset_name], palette=colors_light,
#                hue='test', order=order, ax=ax)
#    sns.barplot(x='language', y="value", data=data[data['kind']==subset_name], palette="Dark2",
#                hue='test', order=order, ax=ax)
    numl = len(order2)
    ax.set_xlim(-0.5, numl-0.5)
    ax.set_xticks(ax.get_xticks(), 
                  [' '*0+str(i) for i in order2]
                 )
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[7:], labels[7:]
    ax.axhline(y = 0.05)
    ax.legend(handles, labels, loc = 'center left')
    ax.set_ylabel(y_label)
    ax.set_xlabel('Parallel Language')

def show_tests_by_langs(user_data, axes):
    data2 = user_data[['parallel_lang', 'mean_mark', 'test_id']]
    data = data2.groupby(['parallel_lang', 'test_id']).agg(['count', 'mean', 'std'])
    data['ci'] = 1.96 * data[('mean_mark', 'std')] / np.sqrt(data[('mean_mark', 'count')])
    data['ci_min'] = data[('mean_mark', 'mean')] - data['ci']
    data['ci_max'] = data[('mean_mark', 'mean')] + data['ci']
    data = data.drop([('mean_mark', 'std'), 'ci'], axis=1)
    data.columns = ['count', 'mean', 'ci_min', 'ci_max']
    data['test_id'] = [i[1] for i in data.index]
    data['language'] = [i[0] for i in data.index]
    data.index = range(len(data.index))

    order = ['No Parallel Text', 'Ukranian', 'Belarussian', 'Bulgarian', 'Serbian', 'Slovene', 
             'Polish', 'Slovak', 'Czech']
    order2= ['Control', 'Ukrainian', 'Belarus.', 'Bulgarian', 'Serbian', 'Slovene', 
             'Polish', 'Slovak', 'Czech']
    markers = ['o', 'X', 'd', '^', 's', '*']
    colors = ['#000000', '#222222', '#333333', '#444444', '#555555', '#777777']

    fig, axes = plt.subplots(3, 1, figsize=(12,12), gridspec_kw={'height_ratios': [2, 1, 1]}, num=102)

    for i in range(1, 7):
#        sns.pointplot(x='parallel_lang', y="mean_mark", data=data2[data2.test_id==6-i+1], palette="Dark2",# color=colors[i], 
#                      markers=markers[i-1], errwidth=1.5, capsize=0.1, order=order, #facecolor=colors[i],
#                      join=False, ax=axes[0])
        sns.pointplot(x='parallel_lang', y="mean_mark", hue='parallel_lang',
                      data=data2[data2.test_id==6-i+1], 
                      palette="Dark2",# color=colors[i], 
                      markers=markers[i-1], err_kws={'linewidth': 1.5}, 
                      capsize=0.1, order=order, #facecolor=colors[i],
                      linestyle='none', ax=axes[0]
                     )
#       for j in range(i):
#            offset = moveCollection(axes[0], j, 6/70., "right")
        offset = transforms.ScaledTranslation(6/70., 0, axes[0].figure.dpi_scale_trans)
        moveLines(axes[0], offset, "right")
    axes[0].set_ylabel('Avg. Correctness of a test')
    axes[0].set_xlabel('Parallel Language')
    axes[0].set_ylim(0, 1)
    numl = len(order2)
    axes[0].set_xlim(0, numl)
    axes[0].set_xticks(axes[0].get_xticks(), 
                       [' '*17+str(i) for i in order2]
                      )
    _ = axes[0].legend(handles=[Line2D([], [], marker=m, color='black') for m in markers[::-1]], 
                    labels=range(1, 7), loc = 'lower left', ncol=2, title='Test Number')

    data3_2 = data2.groupby(['parallel_lang', 'test_id']).mean()

    for i in range(9):
        mn = data3_2.loc[order[i], :].min()['mean_mark']
        mx = data3_2.loc[order[i], :].max()['mean_mark']
        sns.lineplot(x=[i+0.62, i+0.62], y=[mn, mx], 
                     color = sns.palettes.get_colormap('Dark2').colors[i%8], 
                     ax = axes[0]) 
        axes[0].text(i+0.64, (mn+mx)/2, f"{mx-mn:5.2}", 
                     color=sns.palettes.get_colormap('Dark2').colors[i%8])

    axes[0].grid()
    
    order0 = order[1:]
    pwrs = []

    for test in range(1, 7):
        calc_p_data(user_data[['parallel_lang', 'mean_mark']][user_data.test_id==test], 
                    test, order0, pwrs)
    calc_p_data(user_data[['parallel_lang', 'mean_mark']], 
                'AllTests', order0, pwrs)

    powers2 = pd.DataFrame(data=pwrs, columns=['kind', 'test', 'language', 'value'])
    powers2['one'] = 1
    colors_light = [(2/3+c[0]/3, 2/3+c[1]/3, 2/3+c[2]/3) for c in sns.palettes.get_colormap('Pastel2').colors]

    draw_p_values(axes[1], powers2, 'pval', order, order2, 'p-value for a test')
    draw_p_values(axes[2], powers2, 'power', order, order2, 'power for a test')
    for ax, t, x in zip(axes, ['a)', 'b)', 'c)'], [-0.7, -1.2, -1.2]):
        ax.text(x, 0.5, t)

    for ax in axes:
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + \
                    ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(12)
        
        
    fig.savefig('img_res/Fig_3_distribution_users_languagess.png', dpi = 600)
    return fig

def show_age(field: str, test: str, user_data, axes):
    if test == 'All':
        data2 = user_data[[field, 'mean_mark']]
        data3 = user_data[[field, 'mean_mark', 'parallel_lang']]
    else:
        data2 = user_data[[field, 'mean_mark']][all_users.test_id == int(test)]
        data3 = user_data[[field, 'mean_mark', 'parallel_lang']][all_users.test_id == int(test)]

    if field == 'age':
        order = ['sch', 'bak', 'mag', 'asp', 'fin', 'unk', 'unk2']
    elif field == 'speciality':
        order = ['yes', 'no', 'unk', 'unk2']
        
    data = calcCIFrame(data2, field)
    data = orderIndex(data, order)
    display(data.T)
  
    data3['no parallel text'] = data3['parallel_lang'] == 'No Parallel Text'
    
    axes.clear()

    axes.set_ylim(0, 1)
    sns.pointplot(x=field, y="mean_mark", 
                  data=data3, color="#FF0000", 
                  markers="x", order=order, 
                  err_kws={'linewidth': 1.5}, capsize=0.1, 
                  linestyle='none', ax=axes
                 ) 

    #sns.boxplot(x="speciality", y="uavg", data=datas[(datas.ln1==lang)], order=["yes", "no"], notch=True, ax=axes[0])
    sns.swarmplot(x=field, y="mean_mark", hue='no parallel text',
                  data=data3, order=order, 
                  palette="Set2", 
                  size=4, color=".3", linewidth=0, 
                  ax=axes, alpha=0.7
                 )
    #axes.set_title(lang)
    offset = moveCollection(axes, 0, 6/72., "left")
    moveLines(axes, offset)

    axes.set_xticklabels(axes.get_xticklabels(), rotation=15, ha='right')

    
    # If there is any difference between linguists and non-linguists?
    # It is for pupils and bachelors, but not for masters and alumni.
    if field == 'age':
        if test == 'All':
            datal = user_data[[field, 'mean_mark', 'parallel_lang']][user_data.speciality=='yes']
            sns.pointplot(x=field, y="mean_mark", data=datal, color="#00FF00", 
                          markers="x", order=order, err_kws={'linewidth': 1.5}, capsize=0.1, 
                          linestyle='none', ax=axes) 
            datan = user_data[[field, 'mean_mark', 'parallel_lang']][user_data.speciality=='no']
            sns.pointplot(x=field, y="mean_mark", data=datan, color="#0000FF", 
                          markers="x", order=order, err_kws={'linewidth': 1.5}, capsize=0.1, 
                          linestyle='none', ax=axes) 
        else:
            datal = user_data[[field, 'mean_mark', 'parallel_lang']] \
                             [(all_users.test_id == int(test)) & (user_data.speciality=='yes')]
            sns.pointplot(x=field, y="mean_mark", data=datal, color="#00FF00", 
                          markers="x", order=order, err_kws={'linewidth': 1.5}, capsize=0.1, 
                          linestyle='none', legend=None, ax=axes) 
            datan = user_data[[field, 'mean_mark', 'parallel_lang']] \
                             [(all_users.test_id == int(test)) & (user_data.speciality=='no')]
            sns.pointplot(x=field, y="mean_mark", data=datan, color="#0000FF", 
                          markers="x", order=order, err_kws={'linewidth': 1.5}, capsize=0.1, 
                          linestyle='none', ax=axes) 
    
    offset = moveCollection(axes, 0, 12/72., "left")
    moveLines(axes, offset)
    _ = moveCollection(axes, -1, 12/72., "left")
    _ = moveCollection(axes, -2, 12/72., "left")
    axes.set_ylabel('Intelligibility')
    
def drawForeign(test_no, user_data, axes):
    if test_no == 'All':
        dat = user_data
    else:
        dat = user_data[user_data.test_id==int(test_no)]
    flangs = dat.groupby('known_langs').count()
#    flangs = list(flangs[flangs['native_lang']>2].index)
    flangs = list(flangs.index)
    dat = dat[dat['known_langs'].map(lambda x: x in flangs)].copy()
    #display(dat)
    axes.clear()
    
    sns.pointplot(x='known_langs', y='mean_mark', 
                  data = dat, markers='x', color='r', 
                  err_kws={'linewidth': 1.5}, capsize=0.2, 
                  linestyle='none', ax=axes
                 )
    sns.swarmplot(x='known_langs', y='mean_mark', 
                  data = dat, ax=axes
                 )

# These data are for statistical image generation.
sim_rus1={#    0        10        20        30        40
"Belarussian":"111!121122211111112110111121021111011011100",
"Bulgarian"  :"200!!120001020111102201122010!01!12112211!0",
"Czech"      :"010!00020020010011222!01002202!2!2221021000",
"Polish"     :"021!011020201021102020!101000!!000!01001002",
"Serbian"    :"01000100!011210111020!1122010001!120122100!",
"Slovak"     :"220!00000020010101220!1100222202102110200!2",
"Ukranian"   :"1211121020121111012110111021010111001210100",   
"Slovene"    :"000!0122!020000011021!0122200!!2!020102112!",
}

# 0 - nothing similar, 1 - same word root, 2 - has word with a similar sence, ! - false translator's friend.
sim_rus2={#    0        10        20        30        40
"Belarussian":"11001111211000111021220000!2111121201020",
"Bulgarian"  :"!10111200021!121100100010!12111!0121!120",
"Czech"      :"!002010201000!21000100200!200110!121!0!0",
"Polish"     :"10000001!120!0210001012020101112!021!020",
"Serbian"    :"!0011010!020012110!100000!10111!11210120",
"Slovak"     :"!00201022100022100010000!!20111!0121!120",
"Ukranian"   :"21011110!11020111001!10020!21111!1201220",   
"Slovene"    :"!001100000!0202102!1000220!2111!!021!1!0",
}

# 0 - nothing similar, 1 - same word root, 2 - has word with a similar sence, ! - false translator's friend.
sim_rus3={#    0        10        20        30
"Belarussian":"100120101222!12!111!1!121210!121121",
"Bulgarian"  :"1210011110002100112!1!101211!111101",
"Czech"      :"110120!002000!00!1002!0!!2!2!!1102!",
"Polish"     :"100122!01202!!00!12!2!12!2!2!!11121",
"Ukranian"   :"111122!012!20!0!!1211!12!2!2!!11121",
"Slovak"     :"111120!0220!0!!0!10!2!22!2!2!!2111!",
"Slovene"    :"110222!200020!!0!1122212!1!2!!2110!",
"Serbian"    :"11!0211!012001101102201!1!11!101122",
}

# 0 - nothing similar, 1 - same word root, 2 - has word with a similar sence, ! - false translator's friend.
sim_rus4={#    0        10        20        30        40
"Ukranian"   :"11011!1101212!21!122!01!10001!2000011001100!",
"Belarussian":"01!1212000212222!222!01212001!21000210012000",
"Polish"     :"20!12!200!222022!222!01010!01!21200201121000",
"Bulgarian"  :"02!1221201!2!121!!22111021111101!!020121110!",
"Czech"      :"!2!11!!20!200011!2220!1010!20!01202022211!!0",
"Slovak"     :"!2201!220!210011!2200!1010!200012010121!1!!2",
"Slovene"    :"22002!0002020221!2122200!2202220!0!210121110",
"Serbian"    :"!2011200012222110022!10022121!02!00!0001100!",
}

# 0 - nothing similar, 1 - same word root, 2 - has word with a similar sence, ! - false translator's friend.
sim_rus5={#    0        10        20        30        40
"Ukranian"   :"221101111021110111!11110210020102!010",
"Belarussian":"021001010001100!11021110!100201022!10",
"Polish"     :"22100101100210201102111000!!101111010",
"Bulgarian"  :"222!00!!10221!2!11!01111!211122112101",
"Czech"      :"221!!0!110221!201002111!!000101111010",
"Slovak"     :"220!1!!110221!!111021110!000111111010",
"Slovene"    :"2!20!20!!02200!1!102111!02111!!010012",
"Serbian"    :"2!2!0!!!00221!!011!!11100011122000002",
}

# 0 - nothing similar, 1 - same word root, 2 - has word with a similar sence, ! - false translator's friend.
sim_rus6={#    0        10        20        30        40
"Ukranian"   :"!1110011210012210001001111210111101",
"Belarussian":"!11112110101111!0011010111120111101",
"Polish"     :"!11122110100102!0022001121100122200",
"Bulgarian"  :"111110011101111112!10!0111000121112",
"Czech"      :"!11102200000100!00010!0!0!1!100001!",
"Serbian"    :"11111201211!1!!1!!!10!010102!211112",
"Slovene"    :"!12120211100122!2202002!21!!2!22120",
}

sim_sim = [sim_rus1, sim_rus2, sim_rus3, sim_rus4, sim_rus5, sim_rus6]
    
def makeWords(langs, texts, set_no):
    words = pd.DataFrame()
    for lang in langs:
        words0 = []
        for i, sent in enumerate(texts['sent'][(texts['id_set']==set_no) & (texts['lang_name']==lang)]):
            for j, s in enumerate(sent.split('_')):
                if j%2 == 1:
                    if lang != 'Russian':
                        t = f'{s}({sim_sim[int(set_no)-1][lang][len(words0)]})'
                    else: 
                        t = s
                    words0.append(t)
        words[lang] = words0
    return words

def makeStatistics(words, wo_frame, rf, sim, set_no):
    tmpres=[]
    langs3=[l for l in rf.columns if l in words.columns]
    count_pos = list(wo_frame.columns).index('mean_mark')
    for j, word in enumerate(words['Russian']):
        for lang in langs3:
            a=rf[lang][set_no, j]
            b=wo_frame[(wo_frame.answer_no == j) & (wo_frame['test_id'] == set_no)].iloc[0, count_pos]
            if b!=0:
                tmpres.append({'orig':word, 'lang':lang, 'fword':words[lang][j], 'type':sim[lang][j], 'wo_par':b, 'w_par':a, 'rel':a/b})
            else:
                tmpres.append({'orig':word, 'lang':lang, 'fword':words[lang][j], 'type':sim[lang][j], 'wo_par':b, 'w_par':a, 'rel':0})

    return pd.DataFrame(data=tmpres, columns=['orig', 'lang', 'fword', 'type', 'wo_par', 'w_par', 'rel'])

def showWords(test_no: str, text_frame):
    if test_no == '1' or test_no == '2':
        langs = ['Russian', 'Ukranian', 'Belarussian', 'Bulgarian', 'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
        display(makeWords(langs, text_frame, test_no))
    elif test_no == '3':
        langs3 = ['Russian', 'Ukranian', 'Belarussian', 'Bulgarian', 'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
        display(makeWords(langs3, text_frame, '3'))
    elif test_no == '4' or test_no == '5':
        langs4 = ['Russian', 'Ukranian', 'Belarussian', 'Polish', 'Bulgarian', 'Czech', 'Slovak', 'Serbian', 'Slovene']
        display(makeWords(langs4, text_frame, test_no))
    else:
        langs4 = ['Russian', 'Ukranian', 'Belarussian', 'Polish', 'Bulgarian', 'Czech', 'Serbian', 'Slovene']
        display(makeWords(langs4, text_frame, test_no))
        

# Generates an image for statistical analysis of dependency 
# between phonetical similarity and intelligibility.
def processIntelligibility(data, texts):
    all_frame = data[data['parallel_lang'] != 'No Parallel Text'].copy()
    wo_frame = data[data['parallel_lang'] == 'No Parallel Text'].copy()
    all_frame['test_id'] = all_frame['test_id'].astype("str")
    wo_frame['test_id'] = wo_frame['test_id'].astype("str")
    
    langs  = ['Russian', 'Ukranian', 'Belarussian', 'Bulgarian', 'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
    langs3 = ['Russian', 'Ukranian', 'Belarussian', 'Bulgarian', 'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
    langs4 = ['Russian', 'Ukranian', 'Belarussian', 'Bulgarian', 'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
    langs5 = ['Russian', 'Ukranian', 'Belarussian', 'Bulgarian', 'Polish', 'Czech', 'Serbian', 'Slovene']

    rf=all_frame.pivot_table(values="mean_mark", index=['test_id', 'answer_no'], columns='parallel_lang')

    # !!! If you will not use these wordsN, it will be faster. But if you suddenly will need the words list, you will reimplement this.
    words1=makeWords(langs, texts, '1')
    words2=makeWords(langs, texts, '2')
    words3=makeWords(langs3, texts, '3')  
    words4=makeWords(langs4, texts, '4')  
    words5=makeWords(langs4, texts, '5')  
    words6=makeWords(langs5, texts, '6')  
    changes={"Polish":[(16, 17), (23, 24), (36, 37)], "Czech":[(16, 17), (26, 27)], "Slovak":[(26, 27)], "Bulgarian":[(16, 17), (23, 24)], "Belarussian":[(40, 41)]}

    # In the first text some words was swapped during the translation.
    for c_lang, chang in changes.items():
        for ch in chang:
            words1[c_lang][ch[0]], words1[c_lang][ch[1]] = words1[c_lang][ch[1]], words1[c_lang][ch[0]]
            rf[c_lang]['1',ch[0]], rf[c_lang]['1',ch[1]] = rf[c_lang]['1',ch[1]], rf[c_lang]['1',ch[0]]

    all_resn = []
    all_resn.append(makeStatistics(words1, wo_frame, rf, sim_rus1, '1'))
    all_resn.append(makeStatistics(words2, wo_frame, rf, sim_rus2, '2'))
    all_resn.append(makeStatistics(words3, wo_frame, rf, sim_rus3, '3'))
    all_resn.append(makeStatistics(words4, wo_frame, rf, sim_rus4, '4'))
    all_resn.append(makeStatistics(words5, wo_frame, rf, sim_rus5, '5'))
    all_resn.append(makeStatistics(words6, wo_frame, rf, sim_rus6, '6'))
    all_tests=['test 1,', 'test 2,', 'test 3,', 'test 4,', 'test 5,', 'test 6,']
    return all_resn, all_tests

draw_shift = 2./72.

def drawIntelligibility2(dats, nams, axes, fig):
    # Join data into one DataFrame.
    replacement=["same root", "similar word", "no analogues", "false friend"]
    replacement2=["1", "2", "0", "!"]
    ddd2 = []
    for n, dat2 in enumerate(dats):
        for r1,r2 in zip(replacement, replacement2):
            dat=dat2[dat2['type']==r2].copy()
            dat['type']=dat['type'].replace(r2, nams[n]+" "+r1)
            ddd2.append(dat)
    ddd=pd.concat(ddd2)
    
    order=[n+" "+r1 for r1 in replacement for n in nams]

    axes[0].clear()
    axes[1].clear()

    sns.pointplot(x=ddd['type'], y=ddd['wo_par'], color="#FF0000", markers="x", order=order, errwidth=1.5, capsize=0.2, join=False, ax=axes[0]) 
    offset = moveCollection(axes[0], 0, draw_shift, "right")
    moveLines(axes[0], offset, "right")
    moveLines(axes[0], offset, "right")
    sns.pointplot(x=ddd['type'], y=ddd['w_par'], palette="Dark2", markers="x", order=order, errwidth=1.5, capsize=0.2, join=False, ax=axes[0]) 
    offset = moveCollection(axes[0], 1, draw_shift, "left")
    moveLines(axes[0], offset, "left")

    sns.swarmplot(x='type', y='w_par', data=ddd, order=order,# hue='lang',
                  palette="Set2", size=2, color="#999999", linewidth=0, ax=axes[0], alpha=0.7)
    
    sns.pointplot(x=ddd['type'], y=ddd['wo_par'], color="#FF0000", markers="x", order=order, errwidth=1.5, capsize=0.2, join=False, ax=axes[1]) 
    offset = moveCollection(axes[1], 0, draw_shift, "left")
    moveLines(axes[1], offset)

    sns.swarmplot(x='type', y='wo_par', data=ddd, order=order,# hue='lang',
                  palette="Set2", size=2, color=".3", linewidth=0, ax=axes[1], alpha=0.7)

    axes[0].set_xticklabels([i  for j in range(4) for i in range(1,7)])
    axes[1].set_xticklabels([i  for j in range(4) for i in range(1,7)])
    axes[0].set(xlabel="With Parallel Text", ylabel="", ylim=(0,1))
    axes[1].set(xlabel="Control Group", ylabel="", ylim=(0,1))
    axes[0].yaxis.grid(True)
    axes[1].yaxis.grid(True)
    

    patch_colors = ['#0011DD', '#FF0044', '#00FFDD', '#DD88DD']
    for i in range(4):
        patches = [Rectangle((-0.5+i*6, -0.5), 6, 2, alpha = 0.05, edgecolor=patch_colors[i])]
        pc = PatchCollection(patches, alpha = 0.05, facecolor=patch_colors[i])
        pc2 = copy.copy(pc)
        axes[0].add_collection(pc)  
        axes[1].add_collection(pc2)  
    
    plt.subplots_adjust(bottom = 0.3, wspace = 0.1)
    axes[0].set_ylabel('Words Intelligibility')
    
    axes[0].text(0.1, 0.02, "Same root")
    axes[0].text(5.6, 0.02, "Cognate synonyms")
    axes[0].text(11.5, 0.02, "Non-Cognate")
    axes[0].text(17.5, 0.02, "False Friends")
    axes[1].text(0.1, 0.02, "Same root")
    axes[1].text(5.6, 0.02, "Cognate synonyms")
    axes[1].text(11.5, 0.02, "Non-Cognate")
    axes[1].text(17.5, 0.02, "False Friends")
    
    fig.savefig('img_res/Fig_4_intelligibility_on_cognate.png', dpi = 600)    


def drawIntelligibility3(dats, nams, ax, fig, lang=None):
    # Join data into one DataFrame.
    replacement=["same root", "similar word", "no analogues", "false friend"]
    replacement2=["1", "2", "0", "!"]
    ddd2 = []
    for n, dat2 in enumerate(dats):
        for r1, r2 in zip(replacement, replacement2):
            if lang == None:
                dat = dat2[dat2['type']==r2].copy()
            else:
                dat = dat2[(dat2['type']==r2) & (dat2['lang']==lang)].copy()
            dat['type'] = dat['type'].replace(r2, nams[n]+" "+r1)
            ddd2.append(dat)
    ddd = pd.concat(ddd2)
    
    order=[n+" "+r1 for r1 in replacement for n in nams]
    palt = []
    for i in range(4):
        palt.extend(sns.palettes.color_palette("Set2")[:5])
        palt.append(sns.palettes.color_palette("Set2")[6])
    palt2 = []
    for i in range(4):
        palt2.extend(sns.palettes.color_palette("Dark2")[:5])
        palt2.append(sns.palettes.color_palette("Dark2")[6])

    ax.clear()

    sns.pointplot(x=ddd['type'], y=ddd['wo_par'], 
                  color="#FF0000", markers="x", order=order, 
                  errwidth=1.5, capsize=0.2, 
                  join=False, ax=ax
                 ) 
    #! offset = moveCollection(ax, 0, draw_shift, "right")
    # offset = moveCollection(ax, 0, draw_shift, "right")

    offset = transforms.ScaledTranslation(6/70., 0, ax.figure.dpi_scale_trans)
    moveLines(ax, offset, "right")
    moveLines(ax, offset, "right")
    sns.pointplot(x=ddd['type'], y=ddd['w_par'], palette=palt2, markers="x", order=order, errwidth=1.5, capsize=0.2, join=False, ax=ax) 
    # offset = moveCollection(ax, 1, draw_shift, "left")
    
    offset = transforms.ScaledTranslation(-6/70., 0, ax.figure.dpi_scale_trans)
    moveLines(ax, offset, "left")
    
    sns.swarmplot(x='type', y='w_par', hue='type',
                  data=ddd, order=order,
                  palette=palt, size=2, 
                  linewidth=0, ax=ax, alpha=0.7)
    
    ax.set_xticklabels([i  for j in range(4) for i in range(1,7)])
    ax.set(xlabel="Test No", 
           ylabel="Correctness of Answers", ylim=(0,1))
    ax.text(0, 0.15, 
            "Parallel texts are coloured, control group is red", 
            color='#220033',
            bbox=dict(boxstyle="round", 
                      ec=(0.8, 0., 0.), 
                      fc=(0.95, 1., 0.95),
                     )
           )
    ax.yaxis.grid(True)
    

    patch_colors = ['#0011DD', '#FF0044', '#00FFDD', '#DD88DD']
    for i in range(4):
        patches = [Rectangle((-0.5+i*6, -0.5), 
                             6, 2, alpha = 0.05, 
                             edgecolor=patch_colors[i]
                            )
                  ]
        pc = PatchCollection(patches, alpha = 0.05, 
                             facecolor=patch_colors[i]
                            )
        pc2 = copy.copy(pc)
        ax.add_collection(pc)  
    
    plt.subplots_adjust(bottom = 0.3, wspace = 0.1)
#     ax.set_ylabel()
    
    ax.text(0.1, 0.02, "Same root")
    ax.text(5.6, 0.02, "Cognate synonyms")
    ax.text(11.5, 0.02, "Non-Cognate")
    ax.text(17.5, 0.02, "False Friends")
    
    fig.savefig('img_res/Fig_4_intelligibility_on_cognate_.png', dpi = 600)    

def showIntelligibility(qu_data, text_frame, axes, fig, lang=None):
    all_resn, all_tests = processIntelligibility(qu_data, text_frame)
    drawIntelligibility3(all_resn, all_tests, axes, fig, lang)
    
markers = ['', 'o', 'v', '^', 'x', '+', '*']
edge_colors = ['#000000','#FF0000', '#00FF00', '#0000FF', '#FF8800', '#8800FF', '#00FF88']

def drawAnIntel(user_data, all_resn, intellig, sameness, test_no, test_len, axes=None):
    if sameness == 'Same':
        intellig['cnt'+str(test_no)] = all_resn[test_no-1][(all_resn[test_no-1]["type"]=='1')][['lang', 'w_par']].groupby('lang').count()['w_par']
    elif sameness == 'Same and Similar':
        intellig['cnt'+str(test_no)] = all_resn[test_no-1][(all_resn[test_no-1]["type"]=='1') | (all_resn[test_no-1]["type"]=='2')][['lang', 'w_par']].groupby('lang').count()['w_par']
    elif sameness == 'No analogues':
        intellig['cnt'+str(test_no)] = all_resn[test_no-1][(all_resn[test_no-1]["type"]=='0')][['lang', 'w_par']].groupby('lang').count()['w_par']
    elif sameness == 'False Friends':
        intellig['cnt'+str(test_no)] = all_resn[test_no-1][(all_resn[test_no-1]["type"]=='!')][['lang', 'w_par']].groupby('lang').count()['w_par']
    elif sameness == 'Similar':
        intellig['cnt'+str(test_no)] = all_resn[test_no-1][(all_resn[test_no-1]["type"]=='2')][['lang', 'w_par']].groupby('lang').count()['w_par']
    elif sameness == 'No analogues and Similar':
        intellig['cnt'+str(test_no)] = all_resn[test_no-1][(all_resn[test_no-1]["type"]=='0') | (all_resn[test_no-1]["type"]=='2')][['lang', 'w_par']].groupby('lang').count()['w_par']
    else:
        intellig['cnt'+str(test_no)] = all_resn[test_no-1][['lang', 'w_par']].groupby('lang').count()['w_par']
    intellig['intel'+str(test_no)] = all_resn[test_no-1][['lang', 'w_par']].groupby('lang').mean()
    intellig['cnt'+str(test_no)] = intellig['cnt'+str(test_no)] / test_len
    mean = user_data[['parallel_lang', 'mean_mark']] \
                     [(user_data.test_id == test_no) & \
                      (user_data.parallel_lang == 'No Parallel Text')]['mean_mark'].mean()
    if axes != None:
        sns.scatterplot(x='cnt'+str(test_no), y='intel'+str(test_no), data=intellig, marker=markers[test_no],
                        hue=intellig.index, ax = axes, alpha=0.7)
#         sns.scatterplot(x=[0 for n in intellig.index], y='wo_par', data=all_resn[test_no-1], marker='o',
#                         hue=intellig.index, ax = axes, alpha=0.7)
        # Горизонтальные линии с маркерами.
#         sns.lineplot(x=[0, 1], y=[mean, mean], alpha = 0.5, linewidth = 1, ax = axes, 
#                      color=sns.palettes.color_palette("tab10")[test_no], )
#         axes.lines[-1].set_linestyle("--")
        sns.scatterplot(x=[-0.05], y=[mean], marker=markers[test_no],
                        facecolor='r', #sns.palettes.color_palette("tab10")[test_no], 
                        ax=axes, alpha=0.7)
    return mean

def showIntel(sameness, user_data, qu_data, text_frame, axes, fig, 
              test_no = 'All tests', show_legend = None):
    
    all_resn, all_tests = processIntelligibility(qu_data, text_frame)
    intellig = pd.DataFrame()
    
    axes.clear()
    
    if test_no == 'All tests':
        mean1 = drawAnIntel(user_data, all_resn, intellig, sameness, 1, 43, axes)
        mean2 = drawAnIntel(user_data, all_resn, intellig, sameness, 2, 40, axes)
        mean3 = drawAnIntel(user_data, all_resn, intellig, sameness, 3, 35, axes)
        mean4 = drawAnIntel(user_data, all_resn, intellig, sameness, 4, 44, axes)
        mean5 = drawAnIntel(user_data, all_resn, intellig, sameness, 5, 37, axes)
        mean6 = drawAnIntel(user_data, all_resn, intellig, sameness, 6, 37, axes)
        intellig.loc["No Parallel Text"] = (None, mean1, None, mean2, None, mean3, None, mean4, None, mean5, None, mean6)
        intellig.columns = [f'%{sameness} Words, Test1', 'Avg.Results, Test1', 
                            f'%{sameness} .Words, Test2', 'Avg.Results, Test2',
                            f'%{sameness} .Words, Test3', 'Avg.Results, Test3',
                            f'%{sameness} .Words, Test4', 'Avg.Results, Test4',
                            f'%{sameness} .Words, Test5', 'Avg.Results, Test5',
                            f'%{sameness} .Words, Test6', 'Avg.Results, Test6']
    else:
        test_len = [43, 40, 35, 44, 37, 37]
        test_n = int(test_no)
        mean1 = drawAnIntel(user_data, all_resn, intellig, sameness, test_n, test_len[test_n-1], axes)
        intellig.loc["No Parallel Text"] = (None, mean1)
        intellig.columns = [f'%{sameness} Words, Test{test_no}', f'Avg.Results, Test{test_no}']
        
    sns.lineplot(x=[0, 1], y=[0, 1], color = '#888888', ax = axes, size=1, legend=None, style=True, dashes=[(4,4)])
#     axes.lines[-1].set_linestyle("--")
    axes.set_ylabel('Correctness of Answers')

    same2title = {"Same": "Same root", "Similar":"Cognate synonyms", 
                  "No analogues": "Non-cognates", "False Friends":"False Friends",
                  'No analogues and Similar': "Non-cognates and Cognate synonyms Together"
                 }
    same2xlabel = {"Same": "Percentage of words with Same root in a Test", 
                   "Similar":"Percentage of Cognate synonyms in a Test", 
                   "No analogues": "Percentage of Non-cognates in a Test", 
                   "False Friends":"Percentage of False Friends in a Test",
                   'No analogues and Similar': "Percentage of Non-cognates and Cognate synonyms in a Test"
                  }
    axes.set_title(same2title.get(sameness, "XXX"))
    axes.set_xlabel(same2xlabel.get(sameness, "XXX"))
    if show_legend == 1:
        handles, labels = axes.get_legend_handles_labels()
        handles, labels = handles[:8], labels[:8]
        labels.append('Control group')
        handles.append(copy.copy(handles[-1]))
        handles[-1].set_color('#FF0000')
        axes.legend(handles, labels)
    elif show_legend == 2:
#         markers = ['o', 'x', 'd', '^', 's', '*']
        _ = axes.legend(handles=[Line2D([], [], marker=m, color='black') for m in markers[1:]], 
                        labels=range(1, 7), loc = 'lower right', ncol=2, title='Test number')
    else:
        axes.get_legend().remove()
        
    display(fig)
    display(intellig)
    
#     same2fig = {"Same": "_1", "Similar":"_2", "No analogues": "_3", "False Friends":"_4"}
#     fig.savefig('img_res/Fig_5'+same2fig.get(sameness, "_X")+'_correlation_on_type.png', dpi = 600)
    
    
    print("Correlation by tests")
    t1, t2 = [], []
    if test_no == 'All tests':
        for i in range(6):
            t = intellig.iloc[0:-1, i*2:i*2+2].dropna()
            t1.extend(t.iloc[:, 0])
            t2.extend(t.iloc[:, 1])
            print(f"Test {i+1}", pearsonr(t.iloc[:,0], t.iloc[:,1])[0])
        print("All Tests", *pearsonr(t1, t2))
        print("\nCorrelation by language")
        for i in range(intellig.shape[0]-1):
            t = pd.DataFrame([[intellig.iloc[i, 2*j], intellig.iloc[i, 2*j+1]] 
                              for j in range(int(intellig.shape[1]/2))])
            t = t.dropna()
            print(intellig.index[i], pearsonr(t.iloc[:, 0].dropna(),
                                              t.iloc[:, 1].dropna())[0])
    else:
        t = intellig.iloc[0:-1, :].dropna()
        t1.extend(t.iloc[:, 0])
        t2.extend(t.iloc[:, 1])
        print(f"Test {test_n}", pearsonr(t.iloc[:,0], t.iloc[:,1])[0])


def showWords2(test_n, coef, word_type, thr, qu_data, text_frame):
    all_resn, all_tests = processIntelligibility(qu_data, text_frame)
    test_n = int(test_n) - 1
    coef = {'<': 1, '>': -1}[coef]
    if word_type != 'All':
        word_type = {'Same':'1', 'Similar': '2', 'No analagues': '0', 'False friend': '!'}[word_type]
        dat = all_resn[test_n][(all_resn[test_n]["type"]==word_type) & (all_resn[test_n]["w_par"]*coef<thr*coef)].copy()
    else:
        dat = all_resn[test_n][(all_resn[test_n]["w_par"]*coef<thr*coef)].copy()
    dat["type"] = dat["type"].replace({'1': 'same', '2': 'similar', '0': 'no analogues', '!': 'false friend'})
    dat.columns = ["Russian Word", "Language", "Translation", "Word Class", "Control Group", "Parallel text", "Relation"]
    display(dat)

def draw_correlations_by_types(data, fig, axes):
    palt = [sns.palettes.color_palette("tab10")[2],
           sns.palettes.color_palette("tab10")[0],
           sns.palettes.color_palette("tab10")[3],
           sns.palettes.color_palette("tab10")[1]
          ]

    sns.scatterplot(data=data, x='wo_par', y='w_par', 
                    style='type', hue='type', size=1, ax=axes, palette=palt, alpha=0.5)

    # Trend lines.
    for t, s, c in zip(['1', '2', '0', '!'], ["solid", "dotted", "dashed", "dashdot"], palt):
        d1 = data[data['type']==t]
        reg = LinearRegression().fit(np.array([[d] for d in d1['wo_par']]), np.array([d for d in d1['w_par']]))
        r2_score = reg.score(np.array([[d] for d in d1['wo_par']]), np.array([d for d in d1['w_par']]))
        plt.plot(np.array([0, 1]), 
                 np.array([reg.intercept_, reg.intercept_+reg.coef_[0]]), 
                 linestyle=s, 
                 color=c)
        plt.text(0.05, reg.intercept_+0.05, f'{r2_score:5.3}')
    axes.set_xlim(0, 1)    
    axes.set_ylim(0, 1)
    # axes.set_title("Word's intelligibility on a class of cognate")
    axes.set_xlabel("Correctness of answers in control text")
    axes.set_ylabel("Correctness of answers with a parallel text")

    _ = axes.legend(handles=[Line2D([], [], marker=m, color=c, linestyle=s) 
                             for m,s,c in zip(['o', 'x', '+', 's'], 
                                              ["solid", "dotted", "dashed", "dashdot"], palt
                                             )
                            ], 
                    labels=['Same root', 'Cognate synonyms', 'Non-cognate', 'False friends'],
                    loc = 'lower right', ncol=2, title='')

    # Custom grid
    divx, divy = 10, 10
    figx, figy = 1, 1

    for i in range(divx):
        sns.lineplot(x=[0, figx], y=[i*figx/divx, i*figy/divy], color = '#888888', ax = axes, 
                     size=1, legend=None, style=True, dashes=[(4,4)])
        sns.lineplot(x=[i*figx/divx, i*figy/divy], y=[0, figy], color = '#888888', ax = axes, 
                     size=1, legend=None, style=True, linestyle='--')

    fig.savefig('img_res/Fig_6_control_test_vs_cognate_classes.png', dpi = 600)    

def draw_correlations_by_types_triples(data, fig, axis, figx, figy):

    divx, divy = 10, 10
    freqs = np.zeros((4, divx+1, divy+1))
    corr = {'1':0, '2':1, '0':2, '!':3}

    for i, row in data.iterrows():
    #     print(corr[row['type']], nt(row['wo_par']*10), int(row['w_par']*10))
        freqs[corr[row['type']]][int(row['wo_par']*divx)][int(row['w_par']*divy)] += 1

    for i in range(4):
        freqs[i] /= freqs[i].max()

    palt = [sns.palettes.color_palette("tab10")[2],
           sns.palettes.color_palette("tab10")[0],
           sns.palettes.color_palette("tab10")[3],
           sns.palettes.color_palette("tab10")[1]
          ]

    for i in range(divx):
        for j in range(divy):
            for k in range(4):
                patches = [Rectangle((figx/(divx*4)*(i*4+k), figy/divy*j), figx/(divx*4), figy/divy, edgecolor=palt[k])]
                pc = PatchCollection(patches, alpha = freqs[k][i][j], facecolor=palt[k])
                axis.add_collection(pc)  

    for k in range(4):
        patches = [Rectangle((figx/(divx*4)*(4*divx-4+k), figy/divy*0), figx/(divx*4), figy/(divy*4), edgecolor=palt[k])]
        pc = PatchCollection(patches, facecolor=palt[k])
        axis.add_collection(pc)  
    for i in range(divx):
        sns.lineplot(x=[0, figx], y=[i*figx/divx, i*figy/divy], color = '#888888', ax = axis, 
                     size=1, legend=None, style=True, dashes=[(4,4)])
        sns.lineplot(x=[i*figx/divx, i*figy/divy], y=[0, figy], color = '#888888', ax = axis, 
                     size=1, legend=None, style=True, linestyle='--')

    sns.lineplot(x=[0, figx], y=[0, figy], color = '#888888', ax = axis, 
                 size=1, legend=None)

    axis.set_xlim(0, figx);
    axis.set_ylim(0, figy);
    axis.set_xlabel("Correctness of answers in control text")
    axis.set_ylabel("Correctness of answers with a parallel text")
    labs = [f"{i/divx:4.2}" if i%2==0 else "" for i in range(divx+1)]
    labs[0], labs[-1] = "0", "1"
    axis.set_xticks([figx/divx*i for i in range(divx+1)])
    axis.set_xticklabels(labs)#, rotation=30);
    axis.set_yticks([figx/divy*i for i in range(divy+1)])
    axis.set_yticklabels(labs);

def draw_all_tests_and_languages(data, axes_all, fig):
    palt = [sns.palettes.color_palette("tab10")[2],
           sns.palettes.color_palette("tab10")[0],
           sns.palettes.color_palette("tab10")[3],
           sns.palettes.color_palette("tab10")[1]
          ]

    order = ['Ukranian', 'Belarussian', 'Bulgarian', 'Serbian', 'Slovene', 
             'Polish', 'Slovak', 'Czech']

    for test_no in range(1, 7):
        for lang_no, lang in enumerate(order):
            if lang == 'Slovak' and test_no == 6:
                continue

            axes = axes_all[test_no - 1][lang_no]
            if test_no == 1:
                axes.set_title(lang)

            palt2 = []
            for c, t in zip(palt, ['1', '2', '0', '!']):
                if data[(data['type']==t) & 
                        (data['test_no']==test_no) & 
                        (data['lang']==lang)].shape[0] != 0:
                    palt2.append(c)

            for t, s, c in zip(['1', '2', '0', '!'], ["solid", "dotted", "dashed", "dashdot"], palt):
                d1 = data[(data['type']==t) & (data['test_no']==test_no) & (data['lang']==lang)]
                sns.scatterplot(data=d1, 
                                x='wo_par', y='w_par', #hue='type', 
                                style='type', 
                                size=1, ax=axes, 
                                color=c, alpha=0.5)
                if d1.shape[0] < 3:
                    continue
                reg = LinearRegression().fit(np.array(object=[[d] for d in d1['wo_par']]), 
                                             np.array([d for d in d1['w_par']]))
                axes.plot(np.array([0, 1]), 
                          [reg.intercept_, reg.intercept_+reg.coef_[0]], 
                          linestyle=s, 
                          color=c
                         )
            axes.set_xlim(0, 1)    
            axes.set_ylim(0, 1)
            axes.set_xlabel = ''
            axes.set_ylabel = ''
            axes.get_legend().remove()

            divx, divy = 10, 10
            figx, figy = 1, 1

            for i in range(divx):
                sns.lineplot(x=[0, figx], y=[i*figx/divx, i*figy/divy], 
                             color = '#888888', 
                             size=1, legend=None, style=True, dashes=[(4,4)], 
                             ax = axes
                            )
                sns.lineplot(x=[i*figx/divx, i*figy/divy], y=[0, figy], 
                             color = '#888888', 
                             size=1, legend=None, style=True, linestyle='--', 
                             ax = axes
                            )

    plt.show()
    fig.savefig('img_res/Fig_6_control_test_vs_cognate_classes_all48.png', dpi = 200)

def draw_all_tests_and_languages(data, axes_all, fig):

    palt = [sns.palettes.color_palette("tab10")[2],
           sns.palettes.color_palette("tab10")[0],
           sns.palettes.color_palette("tab10")[3],
           sns.palettes.color_palette("tab10")[1]
          ]

    order = ['Czech', 'Slovak', 'Polish', 'Belarussian', 
             'Slovene', 'Serbian', 'Bulgarian', 'Ukranian']

    for lang_no, lang in enumerate(order):
        axes = axes_all[int(lang_no/4)][lang_no%4]
        axes.set_title(lang)

        palt2 = []
        for c, t in zip(palt, ['1', '2', '0', '!']):
            if data[(data['type']==t) & 
                    (data['lang']==lang)].shape[0] != 0:
                palt2.append(c)

        for t, s, c in zip(['1', '2', '0', '!'], ["solid", "dotted", "dashed", "dashdot"], palt):
            d1 = data[(data['type']==t) & (data['lang']==lang)]
            sns.scatterplot(data=d1, 
                            x='wo_par', y='w_par', style='type', #hue='type', 
                            size=1, ax=axes, color=c, alpha=0.5)
            if d1.shape[0] < 3:
                continue
            reg = LinearRegression().fit(np.array(object=[[d] for d in d1['wo_par']]), 
                                         np.array([d for d in d1['w_par']]))
            axes.plot(np.array([0, 1]), [reg.intercept_, reg.intercept_+reg.coef_[0]], 
                     linestyle=s, color=c)
            r2_score = reg.score(np.array([[d] for d in d1['wo_par']]), np.array([d for d in d1['w_par']]))
            axes.text(0.05, reg.intercept_+0.05, f'{r2_score:5.3}')

        axes.set_xlim(0, 1)    
        axes.set_ylim(0, 1)
        axes.set_xlabel("Correctness of answers in control text")
        axes.set_ylabel('')
        axes.get_legend().remove()

        divx, divy = 10, 10
        figx, figy = 1, 1

        for i in range(divx):
            sns.lineplot(x=[0, figx], y=[i*figx/divx, i*figy/divy], color = '#888888', ax = axes, 
                         size=1, legend=None, style=True, dashes=[(4,4)])
            sns.lineplot(x=[i*figx/divx, i*figy/divy], y=[0, figy], color = '#888888', ax = axes, 
                         size=1, legend=None, style=True, linestyle='--')


    axes_all[0][0].set_ylabel("Correctness of answers with a parallel text")
    axes_all[1][0].set_ylabel("Correctness of answers with a parallel text")

    plt.show()
    fig.savefig('img_res/Fig_6_control_test_vs_cognate_classes_all8_lang.png', dpi = 200)

def calc_powers(user_data, part_no):
    order = ['Ukranian', 'Belarussian', 'Bulgarian', 
             'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
    #powers = pd.DataFrame(columns=["Test "+str(i) for i in range(1, 7)], index=order)
    powers = pd.DataFrame(columns=[], index=order)
    for i in range(1, 7):
        powers["Test "+str(i)] = -1
    powers["All Tests"] = -1
    for i in range(1, 7):
        powers["Prtcp "+str(i)] = 0
    powers["Total Prtcp"] = -1

    for test in range(1, 7):
        data2 = user_data[['parallel_lang', 'mean_mark']][user_data.test_id==test]

        data = data2.groupby(['parallel_lang']).agg(['count', 'mean', 'std'])
        data['se'] = data[('mean_mark', 'std')] / np.sqrt(data[('mean_mark', 'count')])
        data['ci_min'] = data[('mean_mark', 'mean')] - data['se']
        data['ci_max'] = data[('mean_mark', 'mean')] + data['se']
        data.columns = ['count', 'mean', 'std', 'se', 'ci_min', 'ci_max']

        for i in data.index:
            powers.loc[i, "Test "+str(test)] = norm.cdf(1.645-abs(data['mean'][i]-data['mean']['No Parallel Text'])/(data['se'][i]))
            powers.loc[i, "Prtcp "+str(test)] = part_no[(part_no['test_id']==test)&(part_no['parallel_lang']==i)]['count'].iloc[0]

    data2 = user_data[['parallel_lang', 'mean_mark']]

    data = data2.groupby(['parallel_lang']).agg(['count', 'mean', 'std'])
    data['se'] = data[('mean_mark', 'std')] / np.sqrt(data[('mean_mark', 'count')])
    data['ci_min'] = data[('mean_mark', 'mean')] - data['se']
    data['ci_max'] = data[('mean_mark', 'mean')] + data['se']
    data.columns = ['count', 'mean', 'std', 'se', 'ci_min', 'ci_max']

    for i in data.index:
        powers.loc[i, "All Tests"] = norm.cdf(1.645-abs(data['mean'][i]-data['mean']['No Parallel Text'])/(data['se'][i]))
        powers.loc[i, "Total Prtcp"] = part_no[part_no['parallel_lang']==i]['count'].sum()

    powers = powers.drop("No Parallel Text")
    powers = powers.round(3)
    for test in range(1, 7):
        powers[f"Prtcp {test}"] = powers[f"Prtcp {test}"].astype("int")
    powers["Total Prtcp"] = powers["Total Prtcp"].astype("int")
    
    return powers


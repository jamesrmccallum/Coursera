# -*- coding: utf-8 -*-
import pandas as pd 
import numpy 
import scipy.stats as stats 
import itertools
import seaborn as sns 
import matplotlib.pyplot as plt 

print ('reading data file...')
data = pd.read_csv('nesarc_pds.csv', low_memory=False)
data.columns = map(str.upper, data.columns)

# bug fix for display formats to avoid run time errors - put after code for loading data above
pd.set_option('display.float_format', lambda x:'%f'%x)
pd.set_option('display.max_rows', None)
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 1000)
# Current drinkers(CONSUMER  -  DRINKING STATUS ) Either 1 (yes) or 2(no) to (S7Q31A -  EVER DRANK ALCOHOL TO AVOID SOCIAL PHOBIA)
drinkerstemp=data[(data['CONSUMER'] ==1) & ((data['S7Q31A']=='1') | (data['S7Q31A']=='2'))]

#Get rid of everything unneeded 
drinkers = drinkerstemp[['S7Q31A','S2AQ8B','S2AQ8C','S2AQ10','S2BQ1A2','S2BQ1A4','S2BQ1A7', 'S2BQ1A8','S2BQ3B']].copy()

del drinkerstemp 
del data

for col in drinkers: # Convert columns to numeric and replace 99's and nulls
    drinkers[col] = drinkers[col].convert_objects(convert_numeric=True)
    drinkers[col]=drinkers[col].replace(99 ,numpy.nan).fillna(numpy.nan)

for col in ['S2BQ1A2','S2BQ1A4','S2BQ1A7']: # Set missing values to Nan
    drinkers[col]=drinkers[col].replace(9 ,numpy.nan).fillna(numpy.nan)

drinkers['S7Q31A'] = drinkers['S7Q31A'].map({1:1, 2:0}) 

sns.factorplot(x='S2AQ10',y='S7Q31A',data=drinkers, kind="bar",ci=None)
plt.xlabel('Number of time drank until drunk')
plt.ylabel('Proportion with Social Anxiety')

#Chi Sq test
print('Crosstab for entire dataset\n')
ct= pd.crosstab(drinkers['S7Q31A'],drinkers['S2AQ10'])
print(ct)

print('\nColumn proprotional %s for dataset')
colpcts = ct/ct.sum(axis=0)
print(colpcts)

print('\nChi Sq for entire dataset\n')
cs = stats.chi2_contingency(ct)
print(cs)

bonferroni = 0.05 / 55
#Post Hoc Paired Comparisons 
i = 0
result= []
combos = itertools.combinations(pd.Series.unique(drinkers['S2AQ10'].dropna()),2) 
for key in combos:
    label = 'POSTHOC_' + str(i)    
    _map = {key[0]: key[0], key[1]:key[1]}
    drinkers[label] = drinkers['S2AQ10'].map(_map)   
    ct_ = pd.crosstab(drinkers['S7Q31A'],drinkers[label])
    colpct = ct_/ct_.sum(axis=0)
    cs_ = stats.chi2_contingency(ct_)
    if cs_[1] < bonferroni: 
        print('Post Hoc Analysis ' + str(i+1) + ' comparing ' + str(key[0]) + ' with ' + str(key[1]) + '________________________________________________')
        
        print('\nCrosstab for pair comparison \n')
        print(ct_)
        print('\nFrequency distribution as % values\n')
        print(colpct)
        print ('\nChi Square Contingency Table for post hoc ' + str(i))
        print(cs_)
        out = {'KEY': str(key), 'CS':cs_[0], 'P': cs_[1] }
        result.append(out)
        print('\n')
        i += 1
    del drinkers[label]
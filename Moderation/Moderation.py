import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
import scipy.stats as stats 
import itertools
import seaborn as sns

print ('reading data file...')
data = pd.read_csv('nesarc_pds.csv', low_memory=False)
data.columns = map(str.upper, data.columns)

# bug fix for display formats to avoid run time errors - put after code for loading data above
pd.set_option('display.float_format', lambda x:'%f'%x)
pd.set_option('display.max_rows', None)
# Current drinkers(CONSUMER  -  DRINKING STATUS ) Either 1 (yes) or 2(no) to (S7Q31A -  EVER DRANK ALCOHOL TO AVOID SOCIAL PHOBIA)
drinkerstemp=data[(data['CONSUMER'] ==1) & ((data['S7Q31A']=='1') | (data['S7Q31A']=='2'))]

#Get rid of everything unneeded 
drinkers = drinkerstemp[['SEX','S7Q31A','S2AQ8B','S2AQ8C','S2AQ10','S2BQ1A2','S2BQ1A4','S2BQ1A7', 'S2BQ1A8','S2BQ3B']].copy()

del drinkerstemp 
del data

for col in drinkers: # Convert columns to numeric and replace 99's and nulls
    drinkers[col] = drinkers[col].convert_objects(convert_numeric=True)
    drinkers[col]=drinkers[col].replace(99 ,np.nan).fillna(np.nan)

for col in ['S2BQ1A2','S2BQ1A4','S2BQ1A7']: # Set missing values to Nan
    drinkers[col]=drinkers[col].replace(9 ,np.nan).fillna(np.nan)
    
drinkers['S7Q31A'] = drinkers['S7Q31A'].map({1:1,2:0})
    
# Moderator - Correlation
# Does the presence of social anxiety moderate the relationship between the number of usual drinks and the amount of times drank to abuse?

SA = drinkers[(drinkers['S7Q31A']==1)].dropna()
NO_SA = drinkers[(drinkers['S7Q31A']==0)].dropna()

sns.regplot(x='S2AQ8C',y='S2BQ3B', data=SA)
stats.pearsonr(SA['S2AQ8C'],SA['S2BQ3B'])

sns.regplot(x='S2AQ8C',y='S2BQ3B', data=NO_SA)
stats.pearsonr(NO_SA['S2AQ8C'],NO_SA['S2BQ3B'])

# Moderator - ANOVA 
# Does gender moderate the relationship between social anxiety and the number of usual drinks consumed? 

sns.factorplot(x='S7Q31A',y='S2AQ8B',data=drinkers,kind='bar',ci=None)
plt.xlabel('Respondent has social anxiety')
plt.ylabel('Mean number of drinks ordinarily consumed')

MALE = drinkers[(drinkers['SEX']==1)]
FEMALE = drinkers[(drinkers['SEX']==2)]


MALEMEAN = MALE.groupby(['S7Q31A']).aggregate(np.mean) 
print(MALEMEAN['S2AQ8B'])
model1 = smf.ols(formula = 'S2AQ8B ~ C(S7Q31A)',data=MALE).fit()
print(model1.summary())

sns.factorplot(x='S7Q31A',y='S2AQ8B',data=MALE,kind='bar',ci=None)
plt.xlabel('Respondent has social anxiety')
plt.ylabel('Mean number of drinks ordinarily consumed')


FEMALEMEAN = FEMALE.groupby(['S7Q31A'],as_index=False).aggregate(np.mean)
print(FEMALEMEAN['S2AQ8B'])
model2 = smf.ols(formula = 'S2AQ8B ~ C(S7Q31A)',data=FEMALE).fit()
print(model2.summary())

sns.factorplot(x='S7Q31A',y='S2AQ8B',data=FEMALE,kind='bar',ci=None)
plt.xlabel('Respondent has social anxiety')
plt.ylabel('Mean number of drinks ordinarily consumed')
    
#print('#S2AQ10 - HOW OFTEN DRANK ENOUGH TO FEEL INTOXICATED IN LAST 12 MONTHS')
#['S2AQ8B'] NUMBER OF DRINKS OF ANY ALCOHOL USUALLY CONSUMED ON DAYS WHEN DRANK ALCOHOL IN LAST 12 MONTHS
#S2AQ8C LARGEST NUMBER OF DRINKS OF ANY ALCOHOL CONSUMED ON DAYS WHEN DRANK ALCOHOL IN LAST 12 MONTHS

#print('#S2BQ1A2 - EVER HAD TO DRINK MORE TO GET THE EFFECT WANTED')
#print('#S2BQ1A4 - EVER INCREASE DRINKING BECAUSE AMOUNT FORMERLY CONSUMED NO LONGER GAVE DESIRED EFFECT')
#print('#S2BQ1A7 -  EVER HAVE PERIOD WHEN ENDED UP DRINKING MORE THAN INTENDED')
#print('#S2BQ1A8 -  EVER HAVE PERIOD WHEN KEPT DRINKING LONGER THAN INTENDED') 
#   #S2BQ3B - NUMBER OF EPISODES OF ALCOHOL ABUSE)
#print('ABUSECNT_GRP #S2BQ3B BINNED INTO INTERVALS OF 10 ') 

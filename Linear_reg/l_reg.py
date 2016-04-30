import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
import scipy.stats as stats 
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
drinkers['DRUNK_CNT'] = drinkers['S2AQ10'].map({1:10,2:9,3:8,4:7,5:6,6:5,7:4,8:3,9:2,10:1})
    
#Testing a linear regression model
    
print(drinkers['S7Q31A'].value_counts())
    
lin_reg = smf.ols('S2AQ8B ~ S7Q31A',data=drinkers).fit()
print(lin_reg.summary())   

SA = drinkers[(drinkers['S7Q31A']==1)]
print(SA['S2AQ8B'].mean())

NO_SA = drinkers[(drinkers['S7Q31A']==0)]
print(NO_SA['S2AQ8B'].mean())

#CONSUMER - (1 current drinker, 2 ex drinker, 3 lifetime abstainer)
#S7Q31A - EVER DRANK ALCOHOL TO AVOID SOCIAL PHOBIA (1:yes,2:no,9:Unknown, BL: N/A)
#S2AQ10 - HOW OFTEN DRANK ENOUGH TO FEEL INTOXICATED IN LAST 12 MONTHS'
#S2AQ8B NUMBER OF DRINKS OF ANY ALCOHOL USUALLY CONSUMED ON DAYS WHEN DRANK ALCOHOL IN LAST 12 MONTHS
#S2AQ8C LARGEST NUMBER OF DRINKS OF ANY ALCOHOL CONSUMED ON DAYS WHEN DRANK ALCOHOL IN LAST 12 MONTHS

#S2BQ1A2 - EVER HAD TO DRINK MORE TO GET THE EFFECT WANTED')
#S2BQ1A4 - EVER INCREASE DRINKING BECAUSE AMOUNT FORMERLY CONSUMED NO LONGER GAVE DESIRED EFFECT')
#S2BQ1A7 -  EVER HAVE PERIOD WHEN ENDED UP DRINKING MORE THAN INTENDED')
#S2BQ1A8 -  EVER HAVE PERIOD WHEN KEPT DRINKING LONGER THAN INTENDED') 
#S2BQ3B - NUMBER OF EPISODES OF ALCOHOL ABUSE
#ABUSECNT_GRP #S2BQ3B BINNED INTO INTERVALS OF 10

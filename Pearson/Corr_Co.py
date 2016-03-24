import pandas as pd 
import numpy 
import statsmodels.formula.api as smf
import scipy.stats as stats 
import seaborn as sns
import matplotlib.pyplot as plot

print ('reading data file...')
data = pd.read_csv('nesarc_pds.csv', low_memory=False)
data.columns = map(str.upper, data.columns)

# bug fix for display formats to avoid run time errors - put after code for loading data above
pd.set_option('display.float_format', lambda x:'%f'%x)
pd.set_option('display.max_rows', None)
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

drinkers['S7Q31A'] = drinkers['S7Q31A'].map({1:'SA',2:'NO_SA'}) # Give S7Q31A more intuitive names

#PEARSON

drinkers_clean = drinkers[['S2BQ3B','S2AQ8B']].dropna()

plt = sns.regplot(drinkers_clean['S2BQ3B'],drinkers_clean['S2AQ8B'])
plt.set(xlabel='Number of episodes of alcohol abuse', ylabel='Number of drinks usually consumed')

stats.pearsonr(drinkers_clean['S2BQ3B'],drinkers_clean['S2AQ8B'])

#S2AQ10 - HOW OFTEN DRANK ENOUGH TO FEEL INTOXICATED IN LAST 12 MONTHS')
#S2AQ8B NUMBER OF DRINKS OF ANY ALCOHOL USUALLY CONSUMED ON DAYS WHEN DRANK ALCOHOL IN LAST 12 MONTHS
#S2AQ8C LARGEST NUMBER OF DRINKS OF ANY ALCOHOL CONSUMED ON DAYS WHEN DRANK ALCOHOL IN LAST 12 MONTHS

#S2BQ1A2 - EVER HAD TO DRINK MORE TO GET THE EFFECT WANTED')
#S2BQ1A4 - EVER INCREASE DRINKING BECAUSE AMOUNT FORMERLY CONSUMED NO LONGER GAVE DESIRED EFFECT')
#S2BQ1A7 -  EVER HAVE PERIOD WHEN ENDED UP DRINKING MORE THAN INTENDED')
#S2BQ1A8 -  EVER HAVE PERIOD WHEN KEPT DRINKING LONGER THAN INTENDED') 
#S2BQ3B - NUMBER OF EPISODES OF ALCOHOL ABUSE)

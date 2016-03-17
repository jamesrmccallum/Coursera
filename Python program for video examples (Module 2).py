import pandas as pd 
import numpy 
import statsmodels.formula.api as smf
import scipy.stats as stats 

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

anova_set = drinkers[['S2BQ3B','S7Q31A']].dropna()

#ANOVA
model1= smf.ols(formula='S2BQ3B ~ C(S7Q31A)',data=anova_set).fit()
print (model1.summary())

print ('Means for S2BQ3B by social anxiety status')
print (anova_set.groupby('S7Q31A').mean())
print ('stdev for S2BQ3B by social anxiety status')
print (anova_set.groupby('S7Q31A').std())

#CHI SQUARE 
drinkers['ABUSECNT_GRP'] = pd.cut(drinkers['S2BQ3B'],bins=[0,10,20,30,40,50,60,70,80,90,100], labels=[0,10,20,30,40,50,60,70,80,90])

print(drinkers['ABUSECNT_GRP'].value_counts())

remap = {0:0,10:10,20:20,30:30,40:40,90:90}
drinkers['ABUSECNT_GRP2'] = drinkers['ABUSECNT_GRP'].map(remap)

print(drinkers['ABUSECNT_GRP2'].value_counts())

ct= pd.crosstab(drinkers['S7Q31A'],drinkers['ABUSECNT_GRP2'])
print(ct)

cs = stats.chi2_contingency(ct)
print(cs)
#print('#S2AQ10 - HOW OFTEN DRANK ENOUGH TO FEEL INTOXICATED IN LAST 12 MONTHS')
#['S2AQ8B'] NUMBER OF DRINKS OF ANY ALCOHOL USUALLY CONSUMED ON DAYS WHEN DRANK ALCOHOL IN LAST 12 MONTHS
#S2AQ8C LARGEST NUMBER OF DRINKS OF ANY ALCOHOL CONSUMED ON DAYS WHEN DRANK ALCOHOL IN LAST 12 MONTHS

#print('#S2BQ1A2 - EVER HAD TO DRINK MORE TO GET THE EFFECT WANTED')
#print('#S2BQ1A4 - EVER INCREASE DRINKING BECAUSE AMOUNT FORMERLY CONSUMED NO LONGER GAVE DESIRED EFFECT')
#print('#S2BQ1A7 -  EVER HAVE PERIOD WHEN ENDED UP DRINKING MORE THAN INTENDED')
#print('#S2BQ1A8 -  EVER HAVE PERIOD WHEN KEPT DRINKING LONGER THAN INTENDED') 
#   #S2BQ3B - NUMBER OF EPISODES OF ALCOHOL ABUSE)
#print('ABUSECNT_GRP #S2BQ3B BINNED INTO INTERVALS OF 10 ') 

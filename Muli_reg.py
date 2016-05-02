import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
import statsmodels.api as sm
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

#Binary variables are all of the form (1 -yes,2 - no,9 -unkown,NA) - this fixes them
mapper = ({1:1,2:0})

#Get rid of everything unneeded 
drinkers = drinkerstemp[['AGE','SEX','S7Q31A','S2AQ8B','S2AQ8C',
'S2AQ10','S2BQ1A2','S2BQ1A4','S2BQ1A7', 'S2BQ1A8','S2BQ3B', 
'S7Q1','S7Q2','S7Q3','S7Q4A1','S7Q4A2','S7Q4A3','S7Q4A4','S7Q4A5','S7Q4A6','S7Q4A7','S7Q4A8','S7Q4A9',
'S7Q4A10','S7Q4A11','S7Q4A12','S7Q4A13','S7Q4B','S7Q5','S7Q6']].copy()

del drinkerstemp 
del data

for col in drinkers: # Convert columns  to numeric and replace 99's and nulls
    drinkers[col] = pd.to_numeric(col)
    drinkers[col]=drinkers[col].replace(99 ,np.nan).fillna(np.nan)

for col in ['S2BQ1A2','S2BQ1A4','S2BQ1A7']: # Set missing values to Nan
    drinkers[col]=drinkers[col].replace(9 ,np.nan).fillna(np.nan)

#Fix yes/no to binary 
for col in ['SEX','S7Q31A','S7Q1','S7Q2','S7Q3','S7Q4A1','S7Q4A2','S7Q4A3','S7Q4A4','S7Q4A5','S7Q4A6','S7Q4A7','S7Q4A8','S7Q4A9',
'S7Q4A10','S7Q4A11','S7Q4A12','S7Q4A13','S7Q4B','S7Q5','S7Q6']:
    drinkers[col] = drinkers[col].map(mapper)

del col
del mapper
 
#Add up all the drinkers Social anxiety symptoms
drinkers['sa_symptoms'] = drinkers[['S7Q1','S7Q2','S7Q3','S7Q4A1','S7Q4A2','S7Q4A3','S7Q4A4','S7Q4A5','S7Q4A6','S7Q4A7','S7Q4A8','S7Q4A9',
'S7Q4A10','S7Q4A11','S7Q4A12','S7Q4A13','S7Q4B','S7Q5','S7Q6']].sum(axis=1)

#Center quantitatives
drinkers['maxdrinks_c'] = (drinkers['S2AQ8C'] - drinkers['S2AQ8C'].mean())
drinkers['usualdrinks_c'] = (drinkers['S2AQ8B'] - drinkers['S2AQ8B'].mean())
drinkers['age_c'] = (drinkers['AGE'] - drinkers['AGE'].mean())
drinkers['sa_symptoms_c'] = (drinkers['sa_symptoms'] - drinkers['sa_symptoms'].mean())

#DATA MANAGEMENT END----------------------------------------------------------------------------------------------------


#Testing a linear regression model

#Linear regression - social anxiety and incidents of abuse    
model1 = smf.ols('S2BQ3B ~ S7Q31A',data=drinkers).fit()
print(model1.summary())   

#Testing some multiple regression models 

#multiple regression - added centered number of anxiety symptoms
model2 = smf.ols('S2BQ3B ~ S7Q31A + I(sa_symptoms_c**2)',data=drinkers).fit()
print(model2.summary())

#multiple regression - added max drinks consumed
model3 = smf.ols('S2BQ3B ~ S7Q31A + SEX + maxdrinks_c',data=drinkers).fit()
print(model3.summary())

#Multi reg - usual drinks normally consumed
model4 = smf.ols('S2BQ3B ~ S7Q31A + usualdrinks_c',data=drinkers).fit()
print(model4.summary())

#Model 5 - with age 
model5 = smf.ols('S2BQ3B ~ S7Q31A  + age_c',data=drinkers).fit()
print(model5.summary())

#Model 5 - adjusted
model5b = smf.ols('np.log(S2BQ3B) ~ S7Q31A  + age_c +usualdrinks_c',data=drinkers).fit()
print(model5b.summary()) 

#Model 6 - with age + sex 
model6 = smf.ols('S2BQ3B ~ S7Q31A + SEX',data=drinkers).fit()
print(model6.summary())

#multiple regression - added max drinks consumed
model7 = smf.ols('S2BQ3B ~ S7Q31A + SEX + usualdrinks_c',data=drinkers).fit()
print(model7.summary())

#Delving into Model 5 

#Age vs abuse incidents for SA and NON SA
g = sns.regplot(x='AGE',y='S2BQ3B',scatter=True,data=drinkers[(drinkers['S7Q31A']==1)])
g.set(ylim=(0,40))

g= sns.regplot(x='AGE',y='S2BQ3B',scatter=True,data=drinkers[(drinkers['S7Q31A']==0)])
g.set(ylim=(0,40))

#Regression diagnostic plots

#qq
fig1=sm.qqplot(model5.resid,line='r')

fig1=sm.qqplot(model5b.resid,line='r')

#Pearson residuals plot
stdres=pd.DataFrame(model5.resid_pearson)
fig2 = plt.plot(stdres,'o',ls='None')
l = plt.axhline(y=0,color='r')
plt.ylabel('Standardised Residuals')
plt.xlabel('Obeservation Number')

# diagnostic plot for exogenous 
fig3 = plt.figure(figsize = (12,8))
fig3 = sm.graphics.plot_regress_exog(model5,'age_c',fig=fig3)

# leverage plot
sm.graphics.influence_plot(model5,size=8,fontsize='small')

#CONSUMER - (1 current drinker, 2 ex drinker, 3 lifetime abstainer)
#S7Q31A - EVER DRANK ALCOHOL TO AVOID SOCIAL PHOBIA (1:yes,2:no,9:Unknown, BL: N/A)

#S2AQ10 - HOW OFTEN DRANK ENOUGH TO FEEL INTOXICATED IN LAST 12 MONTHS' (1-11.99, N/A)
#S2AQ8B NUMBER OF DRINKS OF ANY ALCOHOL USUALLY CONSUMED ON DAYS WHEN DRANK ALCOHOL IN LAST 12 MONTHS (1-98,99,BL)
#S2AQ8C LARGEST NUMBER OF DRINKS OF ANY ALCOHOL CONSUMED ON DAYS WHEN DRANK ALCOHOL IN LAST 12 MONTHS (1-98,99,BL)

#S2BQ1A2 - EVER HAD TO DRINK MORE TO GET THE EFFECT WANTED'
#S2BQ1A4 - EVER INCREASE DRINKING BECAUSE AMOUNT FORMERLY CONSUMED NO LONGER GAVE DESIRED EFFECT'

#S2BQ1A7 -  EVER HAVE PERIOD WHEN ENDED UP DRINKING MORE THAN INTENDED'
#S2BQ1A8 -  EVER HAVE PERIOD WHEN KEPT DRINKING LONGER THAN INTENDED'

#S2BQ3B - NUMBER OF EPISODES OF ALCOHOL ABUSE


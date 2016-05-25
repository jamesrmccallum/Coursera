from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

os.chdir("/Users/jmccallum/GIT/Coursera/Coursera")

###### Data Shaping ############
#Load the dataset

print ('reading data file...')
data = pd.read_csv('nesarc_pds.csv', low_memory=False)
data.columns = map(str.upper, data.columns)

# bug fix for display formats to avoid run time errors - put after code for loading data above
pd.set_option('display.float_format', lambda x:'%f'%x)
pd.set_option('display.max_rows', None)
# Current drinkers(CONSUMER  -  DRINKING STATUS ) Either 1 (yes) or 2(no) to (S7Q31A -  EVER DRANK ALCOHOL TO AVOID SOCIAL PHOBIA)
drinkerstemp=data[(data['CONSUMER'] ==1) & ((data['S7Q31A']=='1') | (data['S7Q31A']=='2'))].dropna()

#Binary variables are all of the form (1 -yes,2 - no,9 -unkown,NA) - this fixes them
mapper = ({1:1,2:0})

#Get rid of everything unneeded 
drinkers = drinkerstemp[['AGE','S1Q1D5','SEX','S7Q31A','S2AQ8B','S2AQ8C',
'S2AQ10','S2BQ1A2','S2BQ1A4','S2BQ1A7', 'S2BQ1A8','S2BQ3B', 
'S7Q1','S7Q2','S7Q3','S7Q4A1','S7Q4A2','S7Q4A3','S7Q4A4','S7Q4A5','S7Q4A6','S7Q4A7','S7Q4A8','S7Q4A9',
'S7Q4A10','S7Q4A11','S7Q4A12','S7Q4A13','S7Q4B','S7Q5','S7Q6']]


del drinkerstemp 
del data

# Convert columns  to numeric and replace 99's and nulls
for col in drinkers: 
    drinkers[col] = pd.to_numeric(drinkers[col],errors='coerce')
    drinkers[col]=drinkers[col].replace(99 ,np.nan).fillna(np.nan)

# Set missing values to Nan
for col in ['S2BQ1A2','S2BQ1A4','S2BQ1A7']: 
    drinkers[col]=drinkers[col].replace(9 ,np.nan).fillna(np.nan)

#Fix yes/no to binary 
for col in ['SEX','S1Q1D5','S7Q31A','S7Q1','S7Q2','S7Q3',
            'S7Q4A1','S7Q4A2','S7Q4A3','S7Q4A4','S2BQ1A2','S7Q4A5',
            'S7Q4A6','S7Q4A7','S7Q4A8','S7Q4A9','S7Q4A10',
            'S7Q4A11','S7Q4A12','S7Q4A13','S7Q4B','S7Q5','S7Q6']:
    drinkers[col] = drinkers[col].map(mapper)

del col
del mapper

drinkers = drinkers.dropna()

drinkers['UNDER30'] = drinkers['AGE'].apply(lambda x: 1 if x<30 else 0)

drinkers['S7Q31A'].value_counts()
##### Modeling and Prediction ########################
#Split into training and testing sets

predictors=drinkers[['SEX','S1Q1D5','S7Q31A']]

#Ever have a period where drank more than intended
targets = drinkers['S2BQ1A7']

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
classifier=DecisionTreeClassifier()


classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out,proportion=True)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())


#CONSUMER - (1 current drinker, 2 ex drinker, 3 lifetime abstainer)
#S7Q31A - EVER DRANK ALCOHOL TO AVOID SOCIAL PHOBIA (1:yes,2:no,9:Unknown, BL: N/A)

#S1Q1D5 - WHITE?

#S2AQ10 - HOW OFTEN DRANK ENOUGH TO FEEL INTOXICATED IN LAST 12 MONTHS' (1-11.99, N/A)
#S2AQ8B NUMBER OF DRINKS OF ANY ALCOHOL USUALLY CONSUMED ON DAYS WHEN DRANK ALCOHOL IN LAST 12 MONTHS (1-98,99,BL)
#S2AQ8C LARGEST NUMBER OF DRINKS OF ANY ALCOHOL CONSUMED ON DAYS WHEN DRANK ALCOHOL IN LAST 12 MONTHS (1-98,99,BL)

#S2BQ1A2 - EVER HAD TO DRINK MORE TO GET THE EFFECT WANTED' - BINARY
#S2BQ1A4 - EVER INCREASE DRINKING BECAUSE AMOUNT FORMERLY CONSUMED NO LONGER GAVE DESIRED EFFECT' - BINARY

#S2BQ1A7 -  EVER HAVE PERIOD WHEN ENDED UP DRINKING MORE THAN INTENDED' - BINARY
#S2BQ1A8 -  EVER HAVE PERIOD WHEN KEPT DRINKING LONGER THAN INTENDED' - BINARY

#S2BQ3B - NUMBER OF EPISODES OF ALCOHOL ABUSE

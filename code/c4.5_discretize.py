import warnings
warnings.filterwarnings('ignore')
from scipy import optimize
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import tree
import pydotplus
import graphviz


filename = 'dataset/diabetes_mean.csv'
data = pd.read_csv(filename)

for index,item in enumerate(data[data.columns[0]]):
	if(item==0 or item==1):
		data[data.columns[0]][index]="low"
	elif(item<=5):
		data[data.columns[0]][index]="medium"
	else:
		data[data.columns[0]][index]="high"

for index,item in enumerate(data[data.columns[1]]):
	if(item<95):
		data[data.columns[1]][index]="low"
	elif(item<=140):
		data[data.columns[1]][index]="medium"
	else:
		data[data.columns[1]][index]="high"

for index,item in enumerate(data[data.columns[2]]):
	if(item<80):
		data[data.columns[2]][index]="normal"
	elif(item<=90):
		data[data.columns[2]][index]="normal-high"
	else:
		data[data.columns[2]][index]="high"

data[data.columns[3]]=pd.qcut(data[data.columns[3]],3, labels=["low", "normal","high"])
data[data.columns[4]]=pd.qcut(data[data.columns[4]],2, labels=["low", "high"])

for index,item in enumerate(data[data.columns[5]]):
	if(item<24.9):
		data[data.columns[5]][index]="low"
	elif(item<=29.9):
		data[data.columns[5]][index]="normal"
	elif(item<=34.9):
		data[data.columns[5]][index]="obese"
	else:
		data[data.columns[5]][index]="severely-obese"

for index,item in enumerate(data[data.columns[6]]):
	if(item<0.5275):
		data[data.columns[6]][index]="low"
	else:
		data[data.columns[6]][index]="high"

for index,item in enumerate(data[data.columns[7]]):
	if(item<28.5):
		data[data.columns[7]][index]="range-1"
	else:
		data[data.columns[7]][index]="range-2"



#data.to_csv('C:/Users/Dell/Desktop/dataset/diabetes_paper_mean.csv',index=False)



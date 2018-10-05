import subprocess
from scipy import optimize
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.metrics import confusion_matrix, precision_score


#read the file in csv format
filename = 'dataset/diabetes.csv'
data = pd.read_csv(filename)


#preprocessing with median
m=data.median()
attributes_to_replace_zero =list(data.columns[1:6])
data[attributes_to_replace_zero] = data[attributes_to_replace_zero].replace(0, np.NaN)
data.fillna(m, inplace=True)

#Splitting
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y = data[['Outcome']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#DecisionTreeClassifier - CART
decision_tree_classifier = DecisionTreeClassifier(criterion="gini",random_state = 0)

# Train the classifier on the training set

Y_pred=decision_tree_classifier.fit(X_train, Y_train).predict(X_test)

#Evaluation Parameters
mat=confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix:')
print(mat)
print('Accuracy:')
print(decision_tree_classifier.score(X_test, Y_test))
print('Sensitivity:')
print(mat[1,1]/(mat[1,0]+mat[1,1]))
print('Specificity:')
print(mat[0,0]/(mat[0,0]+mat[0,1]))
print('Precision:')
print(precision_score(Y_test, Y_pred))


"""
dot_file = tree.export_graphviz(decision_tree_classifier, out_file='cart_median.dot', feature_names = list(data)[0:-1],class_names = ['healthy', 'ill'])
dot_file_pruned = tree.export_graphviz(decision_tree_classifier, out_file='cart_median_pruned.dot', feature_names = list(data)[0:-1],class_names = ['healthy', 'ill'],max_depth=3)

subprocess.call(['dot', '-Tpng', 'cart_median.dot', '-o' 'cart_median.png'])
subprocess.call(['dot', '-Tpng', 'cart_median_pruned.dot', '-o' 'cart_median_pruned.png'])
"""


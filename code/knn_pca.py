import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def knn(X, column, k=10):
	means=np.nanmean(X,0)
	clf=None
	clf = neighbors.KNeighborsRegressor(n_neighbors=k)
	missing_idxes = np.where(np.isnan(X[:, column]))[0]
	X_copy = np.delete(X, missing_idxes, 0)
	X_train = np.delete(X_copy, column, 1)
	y_train = X_copy[:,column]
	X_test=X[missing_idxes,:]
	X_test=np.delete(X_test,column,1)
	np.delete(means,column)
	col_mean = None
	for col_id in range(0,7):
		col_missing_idxes = np.where(np.isnan(X_train[:, col_id]))[0]
		if len(col_missing_idxes) == 0:
			continue
		else:
			col_mean = np.nanmean(X_train[:,col_id])
			X_train[col_missing_idxes, col_id] = col_mean

	for col_id in range(0,7):
		col_missing_idxes = np.where(np.isnan(X_test[:, col_id]))[0]
		if len(col_missing_idxes) == 0:
			continue
		else:
			temp=np.delete(X_test,col_missing_idxes,0)
			if np.any(temp) == False:
				X_test[col_missing_idxes,col_id]=means[col_id]
			else:
				temp1=temp[:,col_id]
				col_mean=np.mean(temp1)
				X_test[col_missing_idxes,col_id]=col_mean
	clf.fit(X_train,y_train)
	y_test=clf.predict(X_test)
	X[missing_idxes,column]=y_test
	return X


filename = 'dataset/diabetes.csv'
data = pd.read_csv(filename)
attributes_to_replace_zero =list(data.columns[1:6])
data[attributes_to_replace_zero] = data[attributes_to_replace_zero].replace(0, np.NaN)
D=data.as_matrix()

for col in range(1,6):
	D[:,:-1]=knn(D[:,:-1],col)


pca = PCA(n_components=8)
pca.fit(D[:,:-1])
X=pca.transform(D[:,:-1])
Y = np.reshape(D[:,-1], (-1, 1))
Z = np.append(X,Y,axis=1)


#cols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
cols=['A','B','C','D','E','F','G','H','Outcome']
data=pd.DataFrame(Z,columns=cols)
#data.to_csv('C:/Users/Dell/Desktop/dataset/diabetes_knn_pca.csv',index=False)


import time
start_time = time.time()
import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
np.random.seed(1337)
import pandas as pd 
import tensorflow as tf
import subprocess
from scipy import optimize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Reading the data
data = pd.read_csv('dataset/diabetes.csv')


#Pre-processing by removing rows having 0
attributes_to_replace_zero =list(data.columns[1:6])
data[attributes_to_replace_zero] = data[attributes_to_replace_zero].replace(0, np.NaN)
data.dropna(inplace=True)


#Splitting the data
features = list(data.columns.values)
features.remove('Outcome')
X = data[features]
y = data['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



#Building the ANN

model = Sequential()

# 1st layer: input_dim=8, 12 nodes, Activation: RELU
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))

# 2nd layer: 8 nodes, RELU, Activation: RELU
model.add(Dense(8, init='uniform', activation='relu'))

# output layer: dim=1, Activation: sigmoid
model.add(Dense(1, init='uniform', activation='sigmoid' ))

model.compile(loss='binary_crossentropy',   
             optimizer='adam',
             metrics=['accuracy'])
ckpt_model = 'pima-weights.best.hdf5'
checkpoint = ModelCheckpoint(ckpt_model, 
                            monitor='val_acc',
                            verbose=0,
                            save_best_only=True,
                            mode='max')
callbacks_list = [checkpoint]


print('Starting training...')

#Training the model
model.fit(X_train, y_train,nb_epoch=1000,validation_data=(X_test, y_test),batch_size=16,callbacks=callbacks_list,verbose=0)
Y_pred=model.predict_classes(X_test,verbose=0,batch_size=16)

#Validation
mat=confusion_matrix(y_test, Y_pred)
print(mat)
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print('Sensitivity:')
print((mat[1,1]/(mat[1,0]+mat[1,1]))*100)
print('Specificity:')
print((mat[0,0]/(mat[0,0]+mat[0,1]))*100)
print('Precision:')
print((mat[1,1]/(mat[1,1]+mat[0,1]))*100)
print("\n %s seconds" % (time.time() - start_time))



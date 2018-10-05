COMPARISON OF CLASSIFICATION RESULTS OBTAINED FROM DIABETES DATA

The CD contains 4 folders namely dataset, code, final report and references and below is a detailed 
explanation of each folder. 

-------------------------------------------------------------------------------------------------------------------------------

Dataset:

In this project we built 4 classifier models namely ANN, ID3, CART and C4.5 using the Pima Indian 
dataset downloaded from the UCI Machine Learning Repository. The following are the datasets to be 
used according to model and pre-processing technique.

Model		Pre-processing		Dataset
ID3		Mean			diabetes.csv
		Median			diabetes.csv
		Drop			diabetes.csv
		KNN			diabetes_knn.csv
		KNN+PCA		diabetes_knn_pca.csv
CART		Mean			diabetes.csv
		Median			diabetes.csv
		Drop			diabetes.csv
		KNN			diabetes_knn.csv
		KNN+PCA		diabetes_knn_pca.csv
C4.5		Mean			diabetes_paper_mean.csv
		Median			diabetes_paper_median.csv
		Drop			diabetes_paper_drop.csv
		KNN			diabetes_paper_knn.csv
ANN		Mean			diabetes.csv
		Median			diabetes.csv
		Drop			diabetes.csv
		KNN			diabetes_knn.csv
		KNN+PCA		diabetes_knn_pca.csv
		
-------------------------------------------------------------------------------------------------------------------------------

Code:

In this project python was the primary language used. However, for C4.5 R programming was used.
Therefore the pre-requisits required to run this project are as follows
1. R programming language 
2. Python 3 
3. Python libraries - NumPy, Pandas, sklearn
4. Keras - backend TensorFlow or Theano

We have used anaconda distribution (python), RStudio IDE (R).

The following are the codes to be run according to the model and pre-processing technique.

Model		Pre-processing		File name
ID3		Mean			id3_mean.py
		Median			id3_median.py
		Drop			id3_drop.py
		KNN			id3_knn.py
		KNN+PCA		id3_knn_pca.py
CART		Mean			cart_mean.py
		Median			cart_median.py
		Drop			cart_drop.py
		KNN			cart_knn.py
		KNN+PCA		cart_knn_pca.py
C4.5		Mean			c4.5_mean.r
		Median			c4.5_median.r
		Drop			c4.5_drop.r
		KNN			c4.5_knn.r
ANN		Mean			ann_mean.py
		Median			ann_median.py
		Drop			ann_drop.py
		KNN			ann_knn.py
		KNN+PCA		ann_knn_pca.py

knn.py and knn_pca.py was used to generate the corresponding datasets after performing
KNN pre-processing and KNN followed by PCApre-processing respectively.

-------------------------------------------------------------------------------------------------------------------------------

Final report

This folder contains a subfolder called documentation which contains the necessary source file to run the report
 in ShareLaTex. The final soft copy of the project report is saved as Final Report.pdf. 

-------------------------------------------------------------------------------------------------------------------------------

References

This contains 11 literature papers used as a reference to complete the project.

-------------------------------------------------------------------------------------------------------------------------------


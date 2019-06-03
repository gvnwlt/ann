# Artificial Neural Network 

# Installing Theano

# Installing Tensorflow

# Installing Keras


# import the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset 
dataset = pd.read_csv('Churn_Modelling.csv')

# adding test observation
#dataset_test = pd.read_csv('test_observation.csv') 

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# drop dummy variable
X = X[:, 1:]

# splitting the dataset into the training set and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# storing test row before scaling 
#X_select_test = X_test[1325, 0:]

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# initialize the ANN
classifier = Sequential()

# adding the input layer and the first hidden layer with dropout
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
classifier.add(Dropout(p = 0.1))

# adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
classifier.add(Dropout(p = 0.1))

# adding the output layer 
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fitting the ann to the training set 
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=25)

# predicting the test set results 
y_pred = classifier.predict(X_test)

# convert probabilities into the predicted results 
y_pred = (y_pred > 0.5)

"""
Prediction of Customer Leaving the Bank
Geography: France
Credit Score: 600
Gender: Male
Age: 40 
Tenure: 3 
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""

# use the matrix of features X and the dataset to see what the encoded values are
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# making the confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))  
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

# improving the ANN
# dropout regularization to reduce overfitting if needed

# tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))  
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'nb_epoch':[100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_











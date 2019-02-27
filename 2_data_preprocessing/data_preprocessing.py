# Importing the libraries.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset.
data = pd.read_csv('client_data.csv')
X = data.iloc[:,:3].values
y = data.iloc[:,-1].values

# Handle Missing data.
from sklearn.preprocessing import Imputer
im = Imputer()
X[:, 1:3] = im.fit_transform(X[:,1:3])
# Encode the data.
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label_x = LabelEncoder()
y = label_x.fit_transform(y)

label_countries = LabelEncoder()
X[:,0] = label_countries.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Train/Test Splitting.
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X , y,test_size = 0.3 , random_state = 2424)

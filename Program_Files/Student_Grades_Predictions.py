"""
this programme has for purpose to use the linear regression
to predict the futur grades of the students
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#read data from the file "student-mat.csv"
data = pd.read_csv("data/student-mat.csv", sep=";")

#just for testing printing our data head from the file
print(data.head())

#data structure am going to use
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

"""
for this program i ll be only using data with integer type 
you can see the difference after i ve set the new data structure
"""
print(data.head())

#G3 is the final grade which the program will be predicting based on all the different attributs
predict = "G3"

"""
In this part am going to set up 2 arrays
one for the attributs and the other for the labels  
"""

arr_attributs = np.array(data.drop([predict], 1))
arr_labels = np.array(data[predict])

"""
so in this part, i split all of my values into test and train arrays
in order to test 10% of our data so that the computer doesn't memorise the patterns 
and we can get better accurate results
"""
arr_attributs_train, arr_attributs_test, arr_labels_train, arr_labels_test = sklearn.model_selection.train_test_split(arr_attributs, arr_labels, test_size=0.1)

#saving models and plotting data
best = 0
for _ in range(30):
    arr_attributs_train, arr_attributs_test, arr_labels_train, arr_labels_test = sklearn.model_selection.train_test_split(arr_attributs, arr_labels, test_size=0.1)

    #create training model
    linear = linear_model.LinearRegression()

    linear.fit(arr_attributs_train, arr_labels_train)
    acc = linear.score(arr_attributs_test, arr_labels_test)
    #this is our accuracy
    print(acc)
    #it's only goning to save a model if it current accuracy is better than the ones before
    if acc > best:
        best = acc
        with open("Model_Save/studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("Model_Save/studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


print ('Coefficients : \n', linear.coef_)
print ('Intercept : \n', linear.intercept_)

#this is how this works on students (as a test)

predictions = linear.predict(arr_attributs_test)

for x in range(len(predictions)):
    print (predictions[x], arr_attributs_test[x], arr_labels_test[x])

#drawing the graph
p = "G1"

style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")

pyplot.show()


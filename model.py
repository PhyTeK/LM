#!/Users/philippe.martinet/anaconda3/bin/python3.6
# Text Mining Project
# Philippe Martinet. May 2018.

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

def main():
    print("# Import Data for adapted alogrithm")
    from sklearn.feature_extraction.text import CountVectorizer

    X_train = pd.read_csv('TrainingData.txt',sep='~',header=None)
    X_test = pd.read_csv('TestData.txt',sep='~',header=None)
    y_train_c = pd.read_csv("TrainCategoryMatrix.csv",sep=',',header=None)
    y_test_c = pd.read_csv("TestTruth.csv",sep=',',header=None)
    
    # Replace -1 with 0
    y_train = y_train_c.replace(-1,0)
    y_test = y_test_c.replace(-1,0)

    print("X and y trainning data")
    print(X_train.shape)
    print(y_train.shape)
    
    print("X and y Testing data")
    print(X_test.shape)
    print(y_test.shape)

    # Tranform/Vectorize Data
    print("Vectorize and normalize")

    count_vect = CountVectorizer()

    X_train = count_vect.fit_transform(X_train[1]) # Vectorize and normalize text data
    X_test = count_vect.transform(X_test[1]) # Vectorize (only) text data

    y_train = y_train.to_sparse().to_coo()
    y_test = y_test.to_sparse().to_coo()


    from skmultilearn.adapt import MLkNN
    classifier = MLkNN(k=20)

    print("Train Adapted algorithm")
        
    classifier.fit(X_train, y_train)

    print("Predict")
    predictions = classifier.predict(X_test)
    
    from sklearn.metrics import accuracy_score

    print("Accuracy")
    print(accuracy_score(y_test.toarray(),predictions))
    
    print("End")
    

if __name__ == '__main__':
    print("Started ...")
    main()


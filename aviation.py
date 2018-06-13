#!/Users/philippe.martinet/anaconda3/bin/python3.6
# Text Mining Project
# Philippe Martinet. May 2018.

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

def main():
    print("# Import Data")
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
    X_test = count_vect.transform(X_test[1]) # Vectorize Test data using same vocabular index
    print('X_train:', X_train.shape)
    print('X_test', X_test.shape)
    
    def adapt(X_train,y_train,X_test,y_test):

        y_train = y_train.to_sparse().to_coo()
        y_test = y_test.to_sparse().to_coo()


        from skmultilearn.adapt import MLkNN
        classifier = MLkNN(k=4)

        print("Train Adapted algorithm")

        classifier.fit(X_train, y_train)

        print("Predict")
        predictions = classifier.predict(X_test)

        from sklearn.metrics import accuracy_score

        print("Accuracy")
        print(y_test.shape,predictions.shape)
        print(accuracy_score(y_test.toarray(),predictions))

    adapt(X_train,y_train,X_test,y_test)

    def naive(X_train,y_train,X_test,y_test):

        y_train = y_train.values   # DataFrame to numpy array
        #y_test = y_test.values
        #y_test = y_test.to_coo()
        
        #from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer

        print("MultiLabelBinarized")
        #clf = GaussianNB()
        classif = OneVsRestClassifier(estimator=SVC(random_state=0))
        print('y_train', y_train.shape)
        #y_train = MultiLabelBinarizer().fit_transform(y_train)
        print('y_train', y_train.shape)
        print(2)
        clf= classif.fit(X_train, y_train)
        print('X_test', X_test.shape)


        print("Predict")
        
        predictions = clf.predict(X_test)
        print(3)
        
        from sklearn.metrics import accuracy_score


        # for i in range(0,21):
        #     print("Train category {}".format(i))
        #     clf.fit(X_train.toarray(), y_train[i])

        #     print("Predict category {}".format(i))
        #     predictions = clf.predict(X_test.toarray())


        #     print("Accuracy of category {}".format(i))
        #     print(accuracy_score(y_test[i],predictions))


        def test():
            X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
            y = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4]]
            print(y.shape)
            y = MultiLabelBinarizer().fit_transform(y)
            print(y.shape)
            print(classif.fit(X, y).predict(X))
        
        #test()
        
        print("y and pred shapse {} {}".format(y_test.shape,predictions.shape))
        #print('y', y_test)
        #print('pred',dir( predictions))
        #print(predictions.view())
        print(y_test.shape,predictions.shape)
        print(accuracy_score(y_test,predictions))



    naive(X_train,y_train,X_test,y_test)
    
   
    

if __name__ == '__main__':
    print("Started ...")
    main()
    print("End")

'''
Results: 180611

Started ...
# Import Data for adapted alogrithm
X and y trainning data
(21519, 2)
(21519, 22)
X and y Testing data
(7077, 2)
(7077, 22)
Vectorize and normalize
X_train: (21519, 25539)
X_test (7077, 25539)
Train Adapted algorithm
Predict
Accuracy
(7077, 22) (7077, 22)
0.1520418256323301
MultiLabelBinarized
y_train (21519, 22)
y_train (21519, 22)
2
X_test (7077, 25539)
Predict
3
y and pred shape (7077, 22) (7077, 22)
(7077, 22) (7077, 22)
0.13183552352691819
End

real	44m6.005s
user	43m21.202s
sys	0m32.208s


'''

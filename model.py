#!/Users/philippe.martinet/anaconda3/bin/python3.6
# Text Mining Project
# Philippe Martinet. May 2018.
# Tryone() without Pandas
# Trytwo() Using Pandas
# Trythird() Using naive Bayne
# TryFourth() Using adapted alogrithm

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

def tryone():
    df = pd.read_csv("TrainingData_small.txt",sep='~',header=None)
    dc = pd.read_csv("TrainCategoryMatrix.csv",sep=',',header=None)

    print(df.head(10))
    #print(df.describe())
    #print(dc.head(10))
    #print(dc.describe())

    #dc[12].hist(bins=50)
    #plt.plot(df[0],'o-')
    #plt.show()

    vocab = {}
    docs = {}
    text = df.values
    n = 0
    for line in text:
        #print(line[1])
        for word in line[1].split():
            wd = word.lower()

            #print(wd)
            # if wd in vocab:
            #     vocab[wd] += 1
            # else:
            #     vocab[wd] = 1
            if wd in docs:
                docs[wd].append(n)
            else:
                docs[wd] = [n]

        n += 1
        #if n == 2:break

    Ndocs = n-1
    for item in docs:
         vocab[item] = len(docs[item])

    print('vocab docs',len(docs))
    print('vocab length',len(vocab))
    # vocab['word'] = [Id,[doc1,doc2,...],[1,23,...]]
    #print(docs)
    #print(vocab)
    #sys.exit()
    # wdId = 0
    # for line in text:
    #     for word in line[1].split():
    #         wd = word.lower()
    #         wordindoc = 1
    #         #print(wd)
    #         if wd in vocab:
    #             try:
    #                 idxdoc = vocab[wd][1].index(n)
    #                 vocab[wd][2][idxdoc] += 1
    #             except:
    #                 vocab[wd][1].append(n)
    #                 vocab[wd][2].append(1)

    #         else:

    #             vocab[wd] = [wdId,[n],[1]]
    #             wdId += 1

    #     n += 1
    #     if n==10000:break

    # print(vocab[1][100])

    # sys.exit()

    #voc ={}
    #for n in vocab:
    #    voc[vocab[1][0]] = sum(vocab[1][2])

    vocab_sort = sorted(vocab.items(), key=lambda x: x[1])
    #for i in range(12000,12010):
    #        print(vocab_sort[i])
    nmax = len(vocab_sort)
    print(nmax)
    n=0
    vocab_idx = {}

    for idx in vocab_sort:
        vocab_idx[n] = idx[1]
        n += 1

    #vocab.clear()        
    #vocab_sort.clear()


    #a = len(vocab_idx) - 200
    #b = len(vocab_idx)
    #x = range(a,b)
    #y = np.array([vocab_idx[i] for i in range(a,b)])

    #plt.plot(x,y)

    #plt.show()


    # Plot total count of a word versus documents
    x = np.array([])
    y = np.array([])
    c = np.array([])
    #d = 0
    #print(docs['locate'])

    # while( d < 500):
    #     for word in docs:
    #         m = docs[word].count(d)
    #         if(m > 10):
    #             #print(d,m,word)
    #             x.append(d)
    #             y.append(m)
    #             #print(word,docs[word])

    #     d += 1

    # x=np.array([0,1,2,3,4])
    # y=np.array([3,3,2,6,4])
    # N=5
    # area = (20 * np.random.rand(N))**2 
    # c = np.sqrt(area)
    # s = 10*x**2
    # print(c)
    # plt.scatter(x,y,marker='>',s=s,c=c)

    # plt.show()

    # sys.exit()
    w = 0

    for word in docs:
        ar = docs[word]
        #print(ar)
        for d in ar:
            m = ar.count(d)

            if(m >= 5):
                x = np.concatenate([x,[d]])
                y = np.concatenate([y,[m]])
                c = np.concatenate([c,[w]])
        w += 1

    print(len(x))
    print(len(y))
    Nwords = len(docs)
    maxx = np.amax(x)
    maxy = np.amax(y)
    N = Nwords
    area = (40 * y/maxy)**2
    print(Nwords,maxx,maxy)
    #c = x/maxx
    # Create new color map
    cmap = plt.cm.jet
    cmaplist= [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist,cmap.N)
    # define the bins and normalize
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    color= ['red' if l == 6 else 'blue' for l in c]

    s = y
    plt.scatter(x,y,marker='.',s=s, c=color, cmap=cmap, norm=norm)
    plt.show()

    #df.boxplot(column='ApplicantIncome')
    #plt.show()
    #df.boxplot(column='ApplicantIncome', by = 'Education')
    #plt.show()
    #df['LoanAmount'].hist(bins=50)
    #plt.show()
    #df.boxplot(column='LoanAmount')
    #plt.show()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

def trytwo():

    # Import Data
    from sklearn.feature_extraction.text import CountVectorizer

    #filenames = ['TrainingData_small.txt']
    X_train = pd.read_csv('TrainingData.txt',sep='~',header=None)
    X_test = pd.read_csv('TestData.txt',sep='~',header=None)
    y_train_c = pd.read_csv("TrainCategoryMatrix.csv",sep=',',header=None)
    y_test_c = pd.read_csv("TestTruth.csv",sep=',',header=None)
    
    #categ_train = np.ravel(categ_train_c.replace(-1,0))
    y_train = y_train_c.replace(-1,0)
    y_test = y_test_c.replace(-1,0)

#    categ_train_join = categ_train.map(str)
#    print(categ_train.head(10))
#    print(categ_train_join.shape)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    

    # Tranform/Vectorize Data

    #count_vect = CountVectorizer()
    #X = count_vect.fit_transform(aviat_train[1]) # Vectorize text data


    # using binary relevance
    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.naive_bayes import GaussianNB

    # initialize binary relevance multi-label classifier
    # with a gaussian naive bayes base classifier
    classifier = BinaryRelevance(GaussianNB())

    # train
    classifier.fit(X_train, y_train)

    # predict
    predictions = classifier.predict(X_test)

    from sklearn.metrics import accuracy_score
    accuracy_score(y_test,predictions)


    #y = count_vect.fit_transform(categ_train) # Vectorize the categories


    
    
    # How many?
    #vocab = count_vect.get_feature_names()
    #str=u'locate'
    #print(str,end=' ')
    #print(count_vect.vocabulary_.get(str))


    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.preprocessing import LabelBinarizer
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X)
    #y_train_tfidf = LabelBinarized().fit_transform(y)

    print(X_train_tfidf.shape)
    #print(y_train_tfidf.shape)
    #print(categ_train.shape)
    #print(categ_train)
    
    # Trainning
    from sklearn.naive_bayes import MultinomialNB

    # Define a pipeline combining a text feature extractor with multi lable classifier
    NB_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(MultinomialNB(
            fit_prior=True, class_prior=None))),
        ])

    
    aviat_test = pd.read_csv('TestData.txt',sep='~',header=None)
    categ_test_c = pd.read_csv("TestTruth.csv",sep=',',header=None)
    categ_test = np.ravel(categ_test_c.replace(-1,0))
    X_test_counts = count_vect.transform(aviat_test[1]) # Transform test data

    categs = [range(0,22)]

    for cat in categs:
        print('Processing category {}'.format(cat))
        clf = MultinomialNB().fit(X_train_tfidf, y[cat]) # Train with the 22 labels

        # train the model using X_dtm & y
        NB_pipeline.fit(X_train, train[category])
        # compute the testing accuracy
        prediction = NB_pipeline.predict(X_test)
        print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))

        X_test_tfidf = tfidf_transformer.transform(X_test_counts)
        y_test_counts = count_vect.transform(categ_test) # Transform categ data
        y_test_tfidf = tfidf_transformer.transform(y_test_counts)

        predicted = clf.predict(X_test_tfidf)
    
        acc = np.mean(predicted == y_test_tfidf)
        
        print('Accuracy {}'.format(acc))
    

def trythree():

    # Import Data
    from sklearn.feature_extraction.text import CountVectorizer

    #filenames = ['TrainingData_small.txt']
    X_train = pd.read_csv('TrainingData.txt',sep='~',header=None)
    X_test = pd.read_csv('TestData.txt',sep='~',header=None)
    y_train_c = pd.read_csv("TrainCategoryMatrix.csv",sep=',',header=None)
    y_test_c = pd.read_csv("TestTruth.csv",sep=',',header=None)
    
    #categ_train = np.ravel(categ_train_c.replace(-1,0))
    y_train = y_train_c.replace(-1,0)
    y_test = y_test_c.replace(-1,0)

#    categ_train_join = categ_train.map(str)
#    print(categ_train.head(10))
#    print(categ_train_join.shape)
    print("X and y trainning data")
    print(X_train.shape)
    print(y_train.shape)
    print("X and y Testing data")
    print(X_test.shape)
    print(y_test.shape)
    

    # Tranform/Vectorize Data
    print("Vector")
    count_vect = CountVectorizer()
    X_train = count_vect.fit_transform(X_train[1]) # Vectorize text data
    X_test = count_vect.transform(X_test[1]) # Vectorize text data
    print(X_train.shape)

    # using binary relevance
    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.naive_bayes import GaussianNB

    # initialize binary relevance multi-label classifier
    # with a gaussian naive bayes base classifier
    classifier = BinaryRelevance(GaussianNB())

    # train
    #print(X_train)
    print("Train")
    classifier.fit(X_train, y_train)

    # predict
    print("Predict")
    print(X_test.shape)
    predictions = classifier.predict(X_test)

    from sklearn.metrics import accuracy_score
    print("Accuracy")
    print(accuracy_score(y_test,predictions))

def tryfourth():
    print("# Import Data for adapted alogrithm")
    from sklearn.feature_extraction.text import CountVectorizer

    #filenames = ['TrainingData_small.txt']
    X_train = pd.read_csv('TrainingData_small.txt',sep='~',header=None)
    X_test = pd.read_csv('TestData.txt',sep='~',header=None)
    y_train_c = pd.read_csv("TrainCategoryMatrix_small.csv",sep=',',header=None)
    y_test_c = pd.read_csv("TestTruth.csv",sep=',',header=None)
    
    #categ_train = np.ravel(categ_train_c.replace(-1,0))
    y_train = y_train_c.replace(-1,0)
    y_test = y_test_c.replace(-1,0)

#    categ_train_join = categ_train.map(str)
#    print(categ_train.head(10))
#    print(categ_train_join.shape)
    print("X and y trainning data")
    print(X_train.shape)
    print(y_train.shape)
    print("X and y Testing data")

    print(X_test.shape)
    print(y_test.shape)
    

    # Tranform/Vectorize Data
    print("Vector")

    count_vect = CountVectorizer()

    X_train = count_vect.fit_transform(X_train[1]) # Vectorize and normalize text data
    X_test = count_vect.transform(X_test[1]) # Vectorize (only) text data

    print(type(X_train))
    print("dir(X_train)",dir(X_train))
    print()
    print(type(y_train))
    print("dir(y_train)",dir(y_train))
    #y_train = count_vect.fit_transform(y_train) # Vectorize and normalize text data
    #y_test = count_vect.transform(y_test) # Vectorize (only) text data
    #y_train = y_train.to_sparse().to_coo()
    #y_test = y_test.to_sparse().to_coo()

    print(dir(y_train))
    y_train = y_train.to_sparse().to_coo()
    y_test = y_test.to_sparse().to_coo()

    print(X_train.shape)


    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.naive_bayes import GaussianNB

    # initialize binary relevance multi-label classifier
    # with a gaussian naive bayes base classifier
    #classifier = BinaryRelevance(GaussianNB())

    from skmultilearn.adapt import MLkNN
    classifier = MLkNN(k=5)

    print("Train Adapted algorithm")
    # train

    print(type(X_train))
    print(type(y_train))
    print(type(X_test))
    print(type(y_test))

    print(X_train.dtype)
    
    classifier.fit(X_train, y_train)

    print("Predict")
    # predict
    print(type(X_test))
    print(type(y_test))

    print(X_test)
    
    predictions = classifier.predict(X_test)
    
    print(type(predictions))

    #print("Accuracy")
    from sklearn.metrics import accuracy_score

    print(dir(predictions))
    print('y_test',y_test)
    print()
    print('predictions')
    print(predictions)
    print(accuracy_score(y_test,predictions))
    
    print("End")
    



if __name__ == '__main__':
    print("Started ...")
    tryfourth()


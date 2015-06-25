from get_params import get_params
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import pickle
import random
import sys
import time
import copy
params = get_params()

def fit_classifier(X_train,X_test,y_train,y_test):

    '''
    Function that fits the training data to the classifier.
    This could be used to train our own classifier from layers 6 or 7 instead of taking the softmax layer.

    :param feature_matrix: features stored in each row of the matrix
    :param labels: class indices of each row of the matrix
    :return: clf: the trained classifier
    '''
    '''

    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    '''


    param_grid = [
        {'C': [0.1, 1, 10, 100], 'kernel': ['linear'], 'class_weight': [{1: 0.1}, {1: 0.5}, {1: 1}]},
    ]



    clf = tune_parameters(X_train,X_test,y_train,y_test,param_grid)

    clf.fit(X_train,y_train)

    return clf

def tune_parameters(X_train,X_test, y_train,y_test,param_grid):

    '''
    Function to tune an SVM classifier and choose its parameters

    :param feature_matrix: training data
    :param labels: labels for training data
    :param param_grid: grid of parameters to try
    :return: clf.best_estimator_ the best classifier

    '''
    #X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.2, random_state=0)

    #X_train,X_test,y_train,y_test = split_data(feature_matrix,labels,params['split_percentage'])

    score = 'f1_weighted'

    clf = GridSearchCV(SVC(C=1), param_grid, cv=5, scoring=score, n_jobs=10)

    clf.fit(X_train, y_train)

    print "Best score during training: ", clf.best_score_
    print "Best estimator", clf.best_estimator_

    print "Classification report for validation set:"
    print classification_report(y_test,clf.predict(X_test))

    return clf.best_estimator_

def update_negatives(clf,feature_matrix,labels):

    # score all samples, choose false positives as negatives and split data again

    pred = clf.predict(feature_matrix)
    labels = np.squeeze(labels)
    new_data = np.vstack((feature_matrix[np.where(labels == 1)[0],:],feature_matrix[np.intersect1d( np.where(labels == 0)[0] , np.where(pred == 1)[0]),:]))

    new_labels = np.ones(len(new_data))
    new_labels[sum(labels):len(new_labels)] = 0

    print "Selected", np.shape(feature_matrix[np.intersect1d( np.where(labels == 0)[0] , np.where(pred == 1)[0]),:]), "negative samples."

    return new_data,new_labels

def split_data(feature_matrix,labels,percentage):

    '''
    X_train = np.vstack(( feature_matrix[np.where(labels==1)[0],:][0:int(percentage * sum(labels) )], feature_matrix[np.where(labels==0)[0],:][0:int(percentage * (len(labels) - sum(labels)) )]))
    X_test = np.vstack(( feature_matrix[np.where(labels==1)[0],:][int(percentage * sum(labels)):len(labels[labels==1]) ], feature_matrix[np.where(labels==0)[0],:][int(percentage * (len(labels) - sum(labels))):len(labels[labels==0])] ))

    y_train = np.hstack(( labels[labels==1][0:int(percentage*sum(labels))], labels[labels==0][0:int(percentage*(len(labels) - sum(labels)))] ))
    y_test = np.hstack(( labels[labels==1][int(percentage*sum(labels)):len(labels[labels==1])] , labels[labels==0][int(percentage*(len(labels) - sum(labels))):len(labels[labels==0])]))
    '''

    X_train,X_test,y_train,y_test = train_test_split(feature_matrix,labels,train_size=percentage)
    return X_train,X_test,y_train,y_test

def select_data(feature_matrix,labels,negatives_multi = 10):

    pos = feature_matrix[np.where(labels==1)[0],:]

    neg = random.sample(feature_matrix[np.where(labels==0)[0],:], min(len(feature_matrix[np.where(labels==0)[0],:]),len(pos)*negatives_multi))

    feature_matrix = np.vstack((pos,neg))

    labels = np.ones(len(feature_matrix))

    labels[len(pos):len(labels)] = 0

    return feature_matrix, labels

def expand_negatives(clf,feats):

    pred = clf.predict(feats)

    return feats[np.where(pred == 0)[0],:]

def load(params):


    feats_train = []
    labels_train = []
    feats_test = []
    labels_test = []

    queries = range(9099,9129)

    i = 1
    # For all queries
    for query in queries:
        # but the unused ones
        if query not in (9100,9113,9117):

            DESCRIPTORS_PATH = os.path.join(params['root'],'5_descriptors',params['net'],'query' + params['year'] + '_selective_search',str(query))

            # Load the descriptors of all 4 instances and stack them
            for f in os.listdir(os.path.join(DESCRIPTORS_PATH,'train')):

                file_to_load = open(os.path.join(DESCRIPTORS_PATH,'train',f),'rb')

                if len(feats_train) == 0:

                    feats_train = pickle.load(file_to_load)
                    _ = pickle.load(file_to_load)
                    labels_train = pickle.load(file_to_load)

                    labels_train = np.reshape(labels_train,((np.shape(labels_train)[0],1)))
                else:

                    feats_train = np.vstack((feats_train,pickle.load(file_to_load)))
                    _ = pickle.load(file_to_load)

                    labels_aux = pickle.load(file_to_load)

                    # Give a different number for the positives of each query
                    labels_aux = np.reshape(labels_aux,((np.shape(labels_aux)[0],1))) * i

                    labels_train = np.vstack((labels_train,labels_aux))

                file_to_load.close()

            for f in os.listdir(os.path.join(DESCRIPTORS_PATH,'test')):

                file_to_load = open(os.path.join(DESCRIPTORS_PATH,'test',f),'rb')

                if len(feats_test) == 0:

                    feats_test = pickle.load(file_to_load)
                    _ = pickle.load(file_to_load)
                    labels_test = pickle.load(file_to_load)

                    labels_test = np.reshape(labels_test,((np.shape(labels_test)[0],1)))
                else:

                    feats_test = np.vstack((feats_test,pickle.load(file_to_load)))
                    _ = pickle.load(file_to_load)

                    labels_aux = pickle.load(file_to_load)

                    # Give a different number for the positives of each query
                    labels_aux = np.reshape(labels_aux,((np.shape(labels_aux)[0],1))) * i

                    labels_test = np.vstack((labels_test,labels_aux))

                file_to_load.close()


            i = i + 1

    return feats_train, feats_test, labels_train, labels_test

def run(params,X_train,X_test,y_train,y_test):


    model_file = os.path.join(params['root'], '9_other','svm_data', 'models',params['distance_type'],'multi_svm' + '.model')

    '''
    feats_ = np.vstack( (feature_matrix[np.where(labels == i)[0],: ], feature_matrix[np.where(labels == 0)[0],:] ))

    labels_query = np.ones(len(feats_))
    labels_query[len( np.where(labels == i)[0] ) : len(labels_query)] = 0
    '''


    t5 = time.time()
    clf = fit_classifier(X_train,X_test,np.squeeze(y_train),np.squeeze(y_test))
    t6 = time.time()

    print "==============="
    print "Saving"
    pickle.dump(clf,open(model_file,'wb'))
    print "Saved"
    print "==============="
    print "Trained and saved in", t6 - t5, "seconds."
    return clf

if __name__ == '__main__':

    params = get_params()

    print "Loading all descriptors..."

    t = time.time()
    X_train,X_test,y_train,y_test = load(params)
    print "Loaded. "
    print np.shape(X_train), np.shape(X_test),np.unique(y_train)
    print "Done. Elapsed time:", time.time() - t


    run(params,X_train,X_test,y_train,y_test)







    








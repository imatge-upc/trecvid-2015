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

def fit_classifier(feature_matrix,labels):

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



    clf = tune_parameters(feature_matrix,labels,param_grid)

    clf.fit(feature_matrix,labels)

    return clf

def tune_parameters(feature_matrix,labels,param_grid):

    '''
    Function to tune an SVM classifier and choose its parameters

    :param feature_matrix: training data
    :param labels: labels for training data
    :param param_grid: grid of parameters to try
    :return: clf.best_estimator_ the best classifier

    '''
    #X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.2, random_state=0)

    X_train,X_test,y_train,y_test = split_data(feature_matrix,labels,params['split_percentage'])

    score = 'f1'

    clf = GridSearchCV(SVC(C=1), param_grid, cv=5, scoring=score)

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


    X_train = np.vstack(( feature_matrix[np.where(labels==1)[0],:][0:int(percentage * sum(labels) )], feature_matrix[np.where(labels==0)[0],:][0:int(percentage * (len(labels) - sum(labels)) )]))
    X_test = np.vstack(( feature_matrix[np.where(labels==1)[0],:][int(percentage * sum(labels)):len(labels[labels==1]) ], feature_matrix[np.where(labels==0)[0],:][int(percentage * (len(labels) - sum(labels))):len(labels[labels==0])] ))

    y_train = np.hstack(( labels[labels==1][0:int(percentage*sum(labels))], labels[labels==0][0:int(percentage*(len(labels) - sum(labels)))] ))
    y_test = np.hstack(( labels[labels==1][int(percentage*sum(labels)):len(labels[labels==1])] , labels[labels==0][int(percentage*(len(labels) - sum(labels))):len(labels[labels==0])]))


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


    feats = []
    labels = []

    queries = range(9099,9129)

    i = 1
    # For all queries
    for query in queries:
        # but the unused ones
        if query not in (9100,9113,9117):

            DESCRIPTORS_PATH = os.path.join(params['root'],'5_descriptors',params['net'],'query' + params['year'] + '_selective_search',str(query))

            # Load the descriptors of all 4 instances and stack them
            for f in os.listdir(DESCRIPTORS_PATH):

                file_to_load = open(os.path.join(DESCRIPTORS_PATH,f),'rb')

                if len(feats) == 0:

                    feats = pickle.load(file_to_load)
                    _ = pickle.load(file_to_load)
                    labels = pickle.load(file_to_load)

                    labels = np.reshape(labels,((np.shape(labels)[0],1)))
                else:

                    feats = np.vstack((feats,pickle.load(file_to_load)))
                    _ = pickle.load(file_to_load)

                    labels_aux = pickle.load(file_to_load)

                    # Give a different number for the positives of each query
                    labels_aux = np.reshape(labels_aux,((np.shape(labels_aux)[0],1))) * i

                    labels = np.vstack((labels,labels_aux))

                file_to_load.close()

            i = i + 1

    #feats = random.sample(feats, min(len(feats),params['num_additional']))


    return feats, labels

def run(i,params,feature_matrix,labels):


    model_file = os.path.join(params['root'], '9_other','svm_data', 'models',params['distance_type'],params['query_name'] + '.model')

    feats_ = np.vstack( (feature_matrix[np.where(labels == i)[0],: ], feature_matrix[np.where(labels == 0)[0],:] ))

    labels_query = np.ones(len(feats_))
    labels_query[len( np.where(labels == i)[0] ) : len(labels_query)] = 0

    print np.shape(labels_query),  np.shape(feats)
    print "Number of positives", np.shape(feats_[np.where(labels_query==1)[0],:])
    print "Number of negatives",np.shape(feats_[np.where(labels_query==0)[0],:])

    print "==============="

    #feats_, labels_ = select_data(feature_matrix,labels_query)

    for idx in range(num_iterations):

        print "Iteration", idx, "for query", params['query_name']

        '''
        if idx > 0:

            t8 = time.time()
            print "==============="
            print "Updating samples..."
            feature_matrix,labels_query = update_negatives(clf,feature_matrix,labels_query)

            print "Done in", time.time() - t8


        if np.shape(feats_)[0] > sum(labels_query)*params['min_negatives']:

            t5 = time.time()
            clf = fit_classifier(feats_,np.squeeze(labels_query))
            t6 = time.time()

            print "Done in", t6 - t5
        else:

            print "I did not find enough false positives. Saving original model..."
        '''

        t5 = time.time()
        clf = fit_classifier(feats_,np.squeeze(labels_query))
        t6 = time.time()

        print "Done in", t6 - t5

    print "==============="
    print "Saving for query", params['query_name']
    pickle.dump(clf,open(model_file,'wb'))
    print "Saved model for query", params['query_name']
    print "==============="

if __name__ == '__main__':

    params = get_params()
    num_iterations = 2
    queries = range(9099,9129)

    print "Loading all descriptors..."

    t = time.time()
    feats, labels = load(params)
    print "Loaded. "
    print np.shape(feats)
    print "Done. Elapsed time:", time.time() - t

    i = 1
    
    for query in queries:

        if query not in (9100,9113,9117):

            t = time.time()
            print query
            params['query_name'] = str(query)
            run(i,params,feats,labels)

            i = i + 1

            print "Done for query", query, '. Elapsed time:', time.time() - t





    








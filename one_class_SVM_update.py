
# coding: utf-8

# In[99]:

import argparse
import sys
import os
import pickle 
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

def parse_args():

    # Parse command line options/arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train_data_path', required=True)
    parser.add_argument('-test', '--test_data_path', required=True)
    parser.add_argument('-out', '--output_directory', required=True)
    args = parser.parse_args()

    return args
    
def load_data(train_file, test_file):
    train = pd.read_csv(train_file, index_col=None, header=None, sep='\t')
    test = pd.read_csv(test_file, index_col=None, header=None, sep='\t')
    X_train = train.values[:, 1:]
    X_test = test.values[:, 1:]
    ID_train = train.values[:, 0]
    ID_test = test.values[:, 0]
    
    return X_train, X_test, ID_train, ID_test

def COSVM(training_data, testing_data, nu_list, kernel_list):
    # Build SVM model
    clf = OneClassSVM(nu=nu_list, kernel=kernel_list, gamma=0.1)
    clf.fit(training_data)
    y_pred_test = clf.predict(testing_data)
    n_error_test = y_pred_test[y_pred_test == -1].size
    testing_accuracy = 1 - 1.0 * n_error_test / testing_data.shape[0]
#    
#    print(n_error_test)
#    print('final accuracy on testing data: ', testing_accuracy, '\n')

    test_score = clf.decision_function(testing_data)
    test_score = test_score.reshape(-1)

    return test_score

def generate_output(ID_train, ID_test, test_score, result_directory):
    score_result = pd.DataFrame(columns=['id','ranking_score','rating'])
    score_result['id'] = ID_test
    score_result['ranking_score'] = test_score
    score_result.loc[score_result.loc[:, 'ranking_score']>0.05, 'rating']='highly trustable'
    score_result.loc[score_result.loc[:, 'ranking_score']<=0.05, 'rating']='trustable'
    score_result.loc[score_result.loc[:, 'ranking_score']<0, 'rating']='unreliable'
    score_result.loc[score_result.loc[:, 'ranking_score']<=-0.05, 'rating']='highly unreliable'
    fake_df = score_result[score_result.loc[:, 'ranking_score']<0]
    true_df = score_result[score_result.loc[:, 'ranking_score']>=0]
    fake_df = fake_df.sort_values('ranking_score', ascending=True)
    true_df = true_df.sort_values('ranking_score', ascending=False)
    fake_df.index = range(fake_df.shape[0])
    true_df.index = range(true_df.shape[0])

    fake_df = fake_df.transpose()
    true_df = true_df.transpose()
    if fake_df.shape[1]>10:
        fake_df = fake_df.iloc[:, :10]
    if true_df.shape[1]>10:
        true_df = true_df.iloc[:, :10]    
    fake_df.to_json(os.path.join(result_directory, "top_10_fake_reviews.json"))
    true_df.to_json(os.path.join(result_directory, "top_10_genuine_reviews.json"))
    score_result = score_result.sort_values('ranking_score', ascending=False)
    score_result.index = range(score_result.shape[0])
    score_result = score_result.transpose()
    score_result.to_json(os.path.join(result_directory, "total_testing_reviews.json"))

def main(argv=None):
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    args = parse_args()
    
    X_train, X_test, ID_train, ID_test = load_data(args.train_data_path, args.test_data_path)
    test_score = COSVM(X_train, X_test, 0.1, 'sigmoid')
    generate_output(ID_train, ID_test, test_score, args.output_directory)


if __name__ == "__main__":
    sys.exit(main())






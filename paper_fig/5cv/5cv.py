import networkx as nx
import numpy as np
import math
import random
import os
import matplotlib.pyplot as plt
from datetime import datetime


dataPath = "m_d.txt"
resultPath = "out/"



def cross_validation_experiment(miRNA_dis_matrix, seed):
    none_zero_position = np.where(np.triu(miRNA_dis_matrix, 1) != 0)
    none_zero_row_index = none_zero_position[0]
    none_zero_col_index = none_zero_position[1]

    zero_position = np.where(np.triu(miRNA_dis_matrix, 1) == 0)
    zero_row_index = zero_position[0]
    zero_col_index = zero_position[1]

    np.random.seed(seed)
    zero_random_index = random.sample(range(len(zero_row_index)), len(none_zero_row_index))
    zero_row_index = zero_row_index[zero_random_index]
    zero_col_index = zero_col_index[zero_random_index]

    positive_randomlist = [i for i in range(len(none_zero_row_index))]
    negative_randomlist = [i for i in range(len(zero_row_index))]
    random.shuffle(positive_randomlist)
    random.shuffle(negative_randomlist)

    metric = np.zeros((1, 7))
    k_folds = 5
    print("seed=%d, evaluating miRNA-disease...." % (seed))

    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k + 1))
        if k != k_folds - 1:
            positive_test = positive_randomlist[k * int(len(none_zero_row_index) / k_folds):(k + 1) * int(
                len(none_zero_row_index) / k_folds)]
            negative_test = negative_randomlist[
                            k * int(len(zero_row_index) / k_folds):(k + 1) * int(len(zero_row_index) / k_folds)]
        else:
            positive_test = positive_randomlist[k * int(len(none_zero_row_index) / k_folds)::]
            negative_test = negative_randomlist[k * int(len(zero_row_index) / k_folds)::]

        positive_test_row = none_zero_row_index[positive_test]
        positive_test_col = none_zero_col_index[positive_test]
        negative_test_row = zero_row_index[negative_test]
        negative_test_col = zero_col_index[negative_test]
        test_row = np.append(positive_test_row, negative_test_row)
        test_col = np.append(positive_test_col, negative_test_col)

        train_miRNA_dis_matrix = np.copy(miRNA_dis_matrix)
        train_miRNA_dis_matrix[positive_test_row, positive_test_col] = 0
        train_miRNA_dis_matrix[positive_test_col, positive_test_row] = 0

        # name = 'miRNA_disease.csv'
        # np.savetxt(name, train_miRNA_dis_matrix, delimiter=',')
        #miRNA_disease_score = du.get_new_scoring_matrices(train_miRNA_dis_matrix)
        name = resultPath+'result_cv=' + str(k) + '.csv'
        np.savetxt(name, train_miRNA_dis_matrix, delimiter=',')

    return metric





if __name__ == "__main__":


    result = np.loadtxt(dataPath,delimiter=',');
    val = cross_validation_experiment(result,10);
    print(val)
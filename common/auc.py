import networkx as nx
import numpy as np
import math
import random
from datetime import datetime


# yz = 0.5

def model_evaluate(real_score, predict_score):
    # predict_score = np.where(predict_score>yz,1,0)
    AUPR = get_AUPR(real_score, predict_score)
    AUC = get_AUC(real_score, predict_score)
    [f1, accuracy, recall, spec, precision] = get_Metrics(real_score, predict_score)
    return np.array([AUPR, AUC, f1, accuracy, recall, spec, precision])


def get_AUPR(real_score, predict_score):
    sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholdlist = []
    for i in range(999):
        threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
        thresholdlist.append(threshold)
    thresholds = np.matrix(thresholdlist)
    TN = np.zeros((1, len(thresholdlist)))
    TP = np.zeros((1, len(thresholdlist)))
    FN = np.zeros((1, len(thresholdlist)))
    FP = np.zeros((1, len(thresholdlist)))
    for i in range(thresholds.shape[1]):
        p_index = np.where(predict_score >= thresholds[0, i])
        TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
        FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
        # print(TP[0, i], FP[0, i])
        n_index = np.where(predict_score < thresholds[0, i])
        FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
        TN[0, i] = len(np.where(real_score[n_index] == 0)[0])
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    x = list(np.array(recall).flatten())
    y = list(np.array(precision).flatten())
    xy = [(x, y) for x, y in zip(x, y)]
    xy.sort()
    x = [x for x, y in xy]
    y = [y for x, y in xy]
    new_x = [x for x, y in xy]
    new_y = [y for x, y in xy]
    new_x[0] = 0
    new_y[0] = 1
    new_x.append(1)
    new_y.append(0)
    area = 0
    for i in range(thresholds.shape[1]):
        area = area + (new_y[i] + new_y[i + 1]) * (new_x[i + 1] - new_x[i]) / 2
    return area


def get_AUC(real_score, predict_score):
    sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholdlist = []
    for i in range(999):
        threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
        thresholdlist.append(threshold)
    thresholds = np.matrix(thresholdlist)
    TN = np.zeros((1, len(thresholdlist)))
    TP = np.zeros((1, len(thresholdlist)))
    FN = np.zeros((1, len(thresholdlist)))
    FP = np.zeros((1, len(thresholdlist)))
    for i in range(thresholds.shape[1]):
        p_index = np.where(predict_score >= thresholds[0, i])
        TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
        FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
        n_index = np.where(predict_score < thresholds[0, i])
        FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
        TN[0, i] = len(np.where(real_score[n_index] == 0)[0])
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)
    x = list(np.array(1 - spe).flatten())
    y = list(np.array(sen).flatten())
    xy = [(x, y) for x, y in zip(x, y)]
    xy.sort()
    new_x = [x for x, y in xy]
    new_y = [y for x, y in xy]
    new_x[0] = 0
    new_y[0] = 0
    new_x.append(1)
    new_y.append(1)
    # print(list(np.array(new_x).flatten()))
    # print(list(np.array(new_y).flatten()))
    area = 0
    for i in range(thresholds.shape[1]):
        area = area + (new_y[i] + new_y[i + 1]) * (new_x[i + 1] - new_x[i]) / 2
    return area


def get_Metrics(real_score, predict_score):
    sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholdlist = []
    for i in range(999):
        threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
        thresholdlist.append(threshold)
    thresholds = np.matrix(thresholdlist)
    TN = np.zeros((1, len(thresholdlist)))
    TP = np.zeros((1, len(thresholdlist)))
    FN = np.zeros((1, len(thresholdlist)))
    FP = np.zeros((1, len(thresholdlist)))
    for i in range(thresholds.shape[1]):
        p_index = np.where(predict_score >= thresholds[0, i])
        TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
        FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
        n_index = np.where(predict_score < thresholds[0, i])
        FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
        TN[0, i] = len(np.where(real_score[n_index] == 0)[0])
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sen = TP / (TP + FN)
    recall = sen
    spec = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1 = 2 * recall * precision / (recall + precision)
    max_index = np.argmax(f1)
    max_f1 = f1[0, max_index]
    max_accuracy = accuracy[0, max_index]
    max_recall = recall[0, max_index]
    max_spec = spec[0, max_index]
    max_precision = precision[0, max_index]
    return [max_f1, max_accuracy, max_recall, max_spec, max_precision]


def constructNet(miRNA_dis_matrix):
    miRNA_matrix = np.matrix(np.zeros((miRNA_dis_matrix.shape[0], miRNA_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(np.zeros((miRNA_dis_matrix.shape[1], miRNA_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((miRNA_matrix, miRNA_dis_matrix))
    mat2 = np.hstack((miRNA_dis_matrix.T, dis_matrix))

    return np.vstack((mat1, mat2))


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
        miRNA_disease_score = du.get_new_scoring_matrices(train_miRNA_dis_matrix)

        test_label_vector = []
        predict_y_proba = []
        for num in range(len(test_row)):
            test_label_vector.append(miRNA_dis_matrix[test_row[num], test_col[num]])
            predict_y_proba.append(miRNA_disease_score[test_row[num], test_col[num]])

        test_label_vector = np.array(test_label_vector)
        predict_y_proba = np.array(predict_y_proba)
        metric += model_evaluate(test_label_vector, predict_y_proba)

    print(metric / k_folds)

    metric = np.array(metric / k_folds)

    name = 'result_seed=' + str(seed) + '.csv'
    np.savetxt(name, metric, delimiter=',')

    return metric


def load_data(file):
    g = nx.read_edgelist(file, delimiter='\t')  # 将对应关系读进邻接矩阵
    adj = nx.adjacency_matrix(g)
    return adj


class MF():

    def __init__(self, X, k, alpha, beta, iterations):
        """
        Perform matrix factorization to predict np.nan entries in a matrix.
        Arguments
        - X (ndarray)   : sample-feature matrix
        - k (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.X = X
        self.num_samples, self.num_features = X.shape
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        # True if not nan
        self.not_nan_index = (np.isnan(self.X) == False)

    def train(self):
        # Initialize factorization matrix U and V
        self.U = np.random.normal(scale=1. / self.k, size=(self.num_samples, self.k))
        self.V = np.random.normal(scale=1. / self.k, size=(self.num_features, self.k))

        # Initialize the biases
        self.b_u = np.zeros(self.num_samples)
        self.b_v = np.zeros(self.num_features)
        self.b = np.mean(self.X[np.where(self.not_nan_index)])
        # Create a list of training samples
        self.samples = [
            (i, j, self.X[i, j])
            for i in range(self.num_samples)
            for j in range(self.num_features)
            if not np.isnan(self.X[i, j])
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            # total square error
            se = self.square_error()
            training_process.append((i, se))
            if (i + 1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i + 1, se))

        return training_process

    def square_error(self):
        """
        A function to compute the total square error
        """
        predicted = self.full_matrix()
        error = 0
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if self.not_nan_index[i, j]:
                    error += pow(self.X[i, j] - predicted[i, j], 2)
        return error

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, x in self.samples:
            # Computer prediction and error
            prediction = self.get_x(i, j)
            e = (x - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (2 * e - self.beta * self.b_u[i])
            self.b_v[j] += self.alpha * (2 * e - self.beta * self.b_v[j])

            # Update factorization matrix U and V
            """
            If RuntimeWarning: overflow encountered in multiply,
            then turn down the learning rate alpha.
            """
            self.U[i, :] += self.alpha * (2 * e * self.V[j, :] - self.beta * self.U[i, :])
            self.V[j, :] += self.alpha * (2 * e * self.U[i, :] - self.beta * self.V[j, :])

    def get_x(self, i, j):
        """
        Get the predicted x of sample i and feature j
        """
        prediction = self.b + self.b_u[i] + self.b_v[j] + self.U[i, :].dot(self.V[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, U and V
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_v[np.newaxis, :] + self.U.dot(self.V.T)

    def replace_nan(self, X_hat):
        """
        Replace np.nan of X with the corresponding value of X_hat
        """
        X = np.copy(self.X)
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if np.isnan(X[i, j]):
                    X[i, j] = X_hat[i, j]
        return X


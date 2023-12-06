import numpy as np
import sys
import temp
import math


def read_matrix(matrix_raw):
    elements = list(map(float, matrix_raw.split()))
    rows, cols = int(elements[0]), int(elements[1])
    matrix = np.array(elements[2:]).reshape(rows, cols)

    return matrix


def answer(matrix):
    print(str(len(matrix)) + ' ' + str(len(matrix[0])), end=' ')
    for row in matrix:
        for element in row:
            print(round(element, 6), end=" ")


class EstimateModel():
    def __init__(self, A, B, pi, emissions):
        self.A = A                      # transition matrix
        self.B = B                      # emission matrix
        self.pi = pi                    # initial state matrix
        self.emissions = emissions      # emission sequence

        self.N = self.A.shape[0]        # number of states
        self.M = self.B.shape[1]        # number of emissions
        self.T = len(self.emissions)    # number of emissions in sequence

    def forward(self):
        T = len(self.emissions)
        N = len(self.A)

        # Initialize alpha matrix and scaling factors
        alpha = np.zeros((T, N))
        scale = np.zeros(T)

        # Initialize first row of alpha matrix with the first observation
        alpha[0, :] = self.pi * self.B[:, self.emissions[0]]
        scale[0] = 1.0 / np.sum(alpha[0, :])
        alpha[0, :] *= scale[0]  # Scale the first row

        # Compute alpha values for the rest of the observations
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t - 1, :] * self.A[:, j]) * self.B[j, self.emissions[t]]
            scale[t] = 1.0 / np.sum(alpha[t, :])
            alpha[t, :] *= scale[t]  # Scale each row

        return alpha, scale

    def backward(self, c):
        beta = np.zeros((self.T, self.N))
        # Initialize the last time step with scaling
        beta[self.T - 1, :] = c[self.T - 1]

        for t in range(self.T - 2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.dot(self.A[i, :], self.B[:, self.emissions[t + 1]] * beta[t + 1, :])
            beta[t, :] *= c[t]  # Scale beta[t]

        return beta

    def compute_gamma_digamma(self, alpha, beta):
        gamma = np.zeros((self.T, self.N))
        digamma = np.zeros((self.T - 1, self.N, self.N))

        for t in range(self.T - 1):
            denom = np.sum([alpha[t, i] * beta[t, i] for i in range(self.N)])
            for i in range(self.N):
                gamma[t, i] = alpha[t, i] * beta[t, i] / denom
                for j in range(self.N):
                    digamma[t, i, j] = (alpha[t, i] * self.A[i, j] * self.B[j, self.emissions[t + 1]] * beta[t + 1, j]) / denom

        denom = np.sum([alpha[self.T - 1, i] * beta[self.T - 1, i] for i in range(self.N)])
        gamma[self.T - 1] = [alpha[self.T - 1, i] * beta[self.T - 1, i] / denom for i in range(self.N)]

        return gamma, digamma

    def re_estimate(self, gamma, di_gamma):
        self.pi = gamma[0, :]

        for i in range(self.N):
            denominator = np.sum(gamma[:-1, i])
            if denominator == 0:
                raise Exception("Converge")
            for j in range(self.N):
                numerator = np.sum([di_gamma[t, i, j] for t in range(self.T - 1)])
                self.A[i, j] = numerator / denominator

        for i in range(self.N):
            denominator = np.sum(gamma[:, i])
            for k in range(self.M):
                numerator = np.sum([gamma[t, i] for t in range(self.T) if self.emissions[t] == k])
                self.B[i, k] = numerator / denominator if denominator != 0 else 0

    def fit(self):

        transition_new, emission_new = temp.baum_welch_algorithm_v2(self.A.copy(), self.B.copy(), self.pi.copy(), self.emissions.copy(), 100, 0.00001)
        answer(transition_new)
        print()
        print(70*"-")
        # print()
        # answer(emission_new)

        for iteration in range(100):
            alpha, c = self.forward()
            beta = self.backward(c)

            gamma, di_gamma = self.compute_gamma_digamma(alpha, beta)

            try:
                self.re_estimate(gamma, di_gamma)
            except Exception:
                print("Converged after {} iterations.".format(iteration + 1))
                break


def main(input_data):
    input_data = input_data.split("\n")

    A = read_matrix(input_data[0])
    B = read_matrix(input_data[1])
    pi = read_matrix(input_data[2])
    emission_sequence = np.array(list(map(int, input_data[3].split()))[1:])

    hmm = EstimateModel(A, B, pi, emission_sequence)
    hmm.fit()

    print(" ".join(map(str, hmm.A.flatten())))


if __name__ == "__main__":
    if True:
        input_data = """          4 4 0.4 0.2 0.2 0.2 0.2 0.4 0.2 0.2 0.2 0.2 0.4 0.2 0.2 0.2 0.2 0.4 
4 4 0.4 0.2 0.2 0.2 0.2 0.4 0.2 0.2 0.2 0.2 0.4 0.2 0.2 0.2 0.2 0.4 
1 4 0.241896 0.266086 0.249153 0.242864 
1000 0 1 2 3 3 0 0 1 1 1 2 2 2 3 0 0 0 1 1 1 2 3 3 0 0 0 1 1 1 2 3 3 0 1 2 3 0 1 1 1 2 3 3 0 1 2 2 3 0 0 0 1 1 2 2 3 0 1 1 2 3 0 1 2 2 2 2 3 0 0 1 2 3 0 1 1 2 3 3 3 0 0 1 1 1 1 2 2 3 3 3 0 1 2 3 3 3 3 0 1 1 2 2 3 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 3 3 3 0 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3 3 0 1 2 3 0 1 1 1 2 3 0 1 1 2 2 2 2 2 3 0 1 1 1 2 2 2 2 3 0 0 0 0 0 1 1 1 1 2 2 3 3 0 1 2 3 3 0 0 0 0 0 0 1 1 2 2 3 0 0 1 1 1 1 1 1 2 3 3 0 0 1 1 1 2 3 0 0 1 2 3 0 1 1 2 3 3 0 0 0 1 2 3 3 3 0 1 1 1 1 2 3 3 3 3 3 3 0 1 2 2 2 2 2 2 3 0 1 1 1 2 2 3 3 3 3 0 1 2 3 0 0 0 1 1 2 2 3 0 0 0 0 0 0 0 1 2 2 2 3 3 3 3 0 0 1 2 2 2 3 3 3 0 0 1 2 2 3 0 0 0 0 1 1 1 2 3 3 3 3 3 3 3 3 0 1 2 3 0 0 1 2 3 3 3 0 0 0 0 0 1 1 1 1 2 3 0 0 0 1 2 2 3 3 0 0 0 1 1 1 1 1 2 3 3 3 3 0 1 1 1 2 2 3 0 1 2 3 3 3 3 0 0 0 0 1 2 3 3 0 1 2 2 3 3 0 0 1 1 2 3 3 0 1 2 2 3 3 3 0 0 1 1 2 3 3 3 3 0 0 1 1 2 3 3 0 1 2 3 0 1 1 2 2 3 0 1 2 3 3 0 1 1 1 2 2 2 3 3 0 0 1 1 1 1 1 2 3 3 3 0 1 1 2 2 2 2 3 3 0 0 1 2 3 0 1 1 2 2 2 2 3 0 0 1 2 2 3 0 0 0 0 0 1 1 1 2 3 0 0 1 2 3 3 0 0 0 1 2 2 2 3 3 0 0 0 1 2 2 2 2 2 3 0 1 1 2 3 0 0 1 1 1 2 2 3 0 0 0 0 1 1 1 2 2 3 0 1 1 1 2 2 2 3 3 0 0 1 2 2 3 3 3 0 1 1 2 3 0 0 0 0 0 1 2 2 2 3 3 3 0 0 0 1 2 3 0 1 1 2 3 3 3 0 1 2 2 2 3 0 0 1 1 1 1 2 3 3 0 0 0 0 1 2 3 3 3 0 0 0 1 1 2 3 0 1 1 1 1 2 2 2 2 2 2 3 0 0 0 0 1 2 2 2 2 3 0 1 2 2 3 0 1 2 3 0 1 2 3 0 0 0 1 1 2 2 3 3 0 1 1 1 1 2 2 3 3 0 1 1 1 2 2 2 3 3 3 0 1 1 2 3 3 0 1 2 3 0 0 0 0 1 2 3 0 0 0 0 0 0 1 2 2 3 3 0 0 1 2 3 0 1 2 2 3 0 0 0 1 1 2 2 2 2 2 3 3 3 3 3 0 1 2 2 3 3 3 3 3 0 0 1 1 2 2 3 0 0 1 2 2 3 3 3 0 0 0 1 2 2 2 2 3 3 0 1 2 3 0 0 1 1 1 2 2 3 0 0 1 1 2 2 2 3 3 0 0 1 1 1 1 1 2 3 3 3 0 1 2 2 2 2 3 3 3 3 3 3 0 0 0 0 0 0 1 2 3 0 0 1 1 1 2 3 0 0 1 1 2 2 2 2 3 3 3 0 1 1 2 2 2 3 3 0 0 0 0 0 0 1 2 2 3 3 0 0 0 0 0 0 1 2 3 3 3 0 1 1 1 2 2 2 2 2 3 3 3 0 1 2 2 2 3 3 3 3 0 0 0 0 1 2 3 3 3 3 3 3 0 0 1 1 1 1 2 3 0 1 2 3 0 1 1 2 3 3 3 0 0 0 0 1 1 2 3 3 3 3 0 0 1 1 1 2 2 2 2 2 2 3 3 0 0 0 1 2 3 0 0 1 1 2 2 3 3 3 3 3 0 0 1 2 2 2 2 3 0 0 1 1 1 1 1 2 3 3 0 0 1 1 1 2 3 3 3 0 0 
"""
        input_data_2 = """          4 4 0.2 0.4 0.2 0.2 0.4 0.2 0.2 0.2 0.2 0.2 0.2 0.4 0.2 0.2 0.4 0.2 
4 4 0.7 0.1 0.1 0.1 0.1 0.7 0.1 0.1 0.1 0.1 0.7 0.1 0.1 0.1 0.1 0.7 
1 4 1.0 0.0 0.0 0.0 
1000 1 0 1 0 2 3 0 1 0 1 2 3 2 0 1 0 3 2 3 2 3 2 1 0 1 0 1 2 3 2 0 2 1 0 1 3 2 3 2 3 2 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 2 3 2 0 1 0 1 2 3 2 3 0 3 2 1 0 3 2 1 2 3 0 1 0 1 2 0 3 0 1 0 3 0 1 2 3 1 2 3 2 1 0 1 0 3 2 3 2 1 2 1 0 2 1 3 2 3 2 3 1 0 1 0 3 2 3 2 0 1 0 2 1 0 1 0 3 1 0 1 0 2 1 0 3 2 3 2 3 2 3 0 1 2 1 0 1 0 1 0 3 2 3 2 0 1 3 2 3 2 3 0 1 0 3 2 3 2 3 0 1 3 2 3 1 0 1 0 3 1 0 1 2 3 0 1 0 3 2 3 0 1 0 1 2 1 0 3 0 1 0 1 3 2 3 1 0 1 2 1 0 3 2 1 0 2 3 1 2 3 2 3 2 3 2 3 0 3 0 1 2 3 2 1 3 2 0 3 2 3 2 3 2 1 0 1 3 0 1 3 2 3 2 0 3 2 3 2 3 2 3 0 1 0 1 2 3 2 3 2 1 0 1 0 3 0 1 0 2 3 0 1 0 1 0 1 2 1 0 1 0 1 3 2 3 2 3 2 3 0 2 1 2 3 2 3 2 0 1 0 1 0 1 0 3 2 3 2 3 0 1 0 1 0 2 3 2 3 2 3 2 3 2 1 0 2 3 2 3 0 1 2 3 2 1 0 1 2 0 3 1 0 1 0 3 0 2 3 1 0 1 2 1 2 3 0 3 0 1 0 3 2 1 0 1 0 3 2 0 1 0 1 2 3 0 3 2 1 0 3 2 0 1 0 1 0 1 0 1 0 1 2 0 1 3 2 3 2 3 0 2 1 0 3 2 1 0 1 0 1 0 2 3 2 1 0 1 0 1 0 1 2 3 0 1 2 0 1 0 1 0 1 0 1 0 3 2 1 0 1 0 1 0 1 2 3 2 3 0 1 0 1 0 3 2 3 2 1 2 1 0 1 0 1 2 3 2 3 2 3 2 3 0 3 0 1 0 3 2 3 2 3 2 1 2 3 2 3 2 3 0 1 3 0 2 1 0 1 0 1 3 0 1 0 1 0 1 0 3 2 3 2 1 0 1 3 1 0 1 0 3 0 1 0 2 3 2 3 2 3 2 3 2 1 2 3 2 3 2 1 0 1 0 1 2 3 2 3 2 1 0 1 2 1 0 1 3 0 3 2 3 2 3 0 1 0 1 2 3 2 1 0 1 0 1 3 2 1 0 3 2 1 0 3 2 3 2 0 1 0 1 2 3 0 2 3 2 3 2 1 0 1 2 3 2 1 0 1 3 2 0 1 2 3 0 2 3 2 3 0 1 0 1 2 3 2 0 1 0 3 2 0 1 0 1 0 1 0 1 0 1 0 1 3 2 3 2 1 2 3 2 1 0 1 0 1 0 1 0 3 0 1 0 1 0 3 0 1 0 1 2 3 0 1 0 3 0 1 0 1 0 1 2 3 0 1 0 3 0 1 0 1 2 3 0 1 0 1 2 1 3 0 1 0 1 0 1 0 3 2 3 2 3 2 1 0 1 0 1 3 0 3 0 1 2 3 1 3 2 3 2 1 3 1 2 1 0 1 0 1 0 2 1 0 1 0 1 2 3 2 1 0 2 3 2 3 2 3 2 3 0 3 2 3 0 1 2 3 2 3 2 3 2 3 1 0 1 0 1 0 1 2 0 3 0 3 2 0 3 2 1 0 1 0 1 0 1 2 3 2 3 2 1 2 3 0 1 0 3 2 0 3 2 0 1 0 1 0 3 0 1 0 1 2 3 2 3 2 1 0 1 0 1 2 3 2 3 2 3 2 1 0 1 0 1 0 3 0 1 2 1 0 3 2 3 2 3 2 0 3 2 3 2 3 0 1 2 3 1 0 1 0 3 2 3 2 1 2 3 0 3 2 3 2 3 2 1 2 3 2 3 2 1 2 3 2 3 2 3 2 1 0 1 0 1 0 3 2 3 0 3 2 3 2 3 0 1 2 1 0 3 0 3 2 3 2 3 2 0 1 2 3 0 1 0 1 2 0 1 0 1 0 1 0 1 3 2 1 0 1 0 1 0 1 0 3 2 3 2 1 0 1 0 1 0 3 2 1 0 1 0 1 0 1 3 2 3 0 3 0 1 
"""
        input_data_3 = """          3 3 0.800000 0.100000 0.100000 0.100000 0.800000 0.100000 0.100000 0.100000 0.800000 
3 2 0.600000 0.400000 0.400000 0.600000 0.400000 0.600000 
1 3 1.000000 0.000000 0.000000 
1300 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 0 0 1 1 1 1 1 0 1 1 1 0 1 0 0 0 1 0 0 0 0 0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 1 0 1 1 1 0 1 1 1 0 1 0 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 1 0 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 0 1 0 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 0 1 1 1 0 0 1 1 0 1 1 1 1 1 0 0 0 0 0 1 1 0 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1 1 1 0 1 0 0 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 0 1 0 0 1 1 0 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 0 0 0 1 0 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 1 0 0 1 1 0 1 1 0 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 0 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 0 1 1 0 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 0 1 0 1 0 1 1 1 1 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 0 0 1 1 1 1 1 0 0 1 0 0 0 0 1 0 1 0 1 0 1 1 0 0 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 0 0 1 1 1 1 1 1 0 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 0 0 0 1 0 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 0 0 1 0 1 1 0 1 1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 0 0 0 0 0 0 0 1 1 1 0 1 1 1 1 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 1 1 1 1 1 0 0 1 0 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 1 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 1 0 0 0 1 1 0 1 1 0 1 1 1 1 1 0 1 0 1 0 0 0 1 1 1 0 0 1 0 0 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 0 0 0 1 0 0 1 1 0 0 0 1 0 1 1 0 0 0 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 0 1 1 1 1 1 0 1 0 0 0 0 0 0 1 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 0 1 0 1 0 0 0 1 0 1 0 1 0 1 0 1 1 0 1 0 0 0 1 1 0 1 1 1 1 0 1 0 1 

"""
    else:
        input_data = sys.stdin.read()

    output_result = main(input_data)

    print(output_result)

    output_data_1 = """          4 4 0.545455 0.454545 0.0 0.0 0.0 0.506173 0.493827 0.0 0.0 0.0 0.504132 0.495868 0.478088 0.0 0.0 0.521912 
4 4 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 
"""
    output_data_2 = """          4 4 0.0 0.694045 0.070896 0.235060 0.684412 0.0 0.228137 0.087451 0.105932 0.271192 0.0 0.622876 0.266083 0.060087 0.673830 0.0 
4 4 1.0 0.0 0.0 0.0 0.0 0.999980 0.0 0.000020 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 
"""
    output_data_3 = """          3 3 0.845052 0.0774739 0.0774739 0.0841673 0.814073 0.101759 0.0841673 0.101759 0.814073 
3 2 0.684137 0.315863 0.126646 0.873354 0.126646 0.873354 
"""

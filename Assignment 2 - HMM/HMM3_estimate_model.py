"""
Problem D
HMM3 Estimate Model
In this task you should show that you know how to estimate the model parameters for an HMM. You will be given a starting guess of a HMM (transition matrix, emission matrix and initial state probability distribution) and a sequence of emissions and you should train the HMM to maximize the probability of observing the given sequence of emissions.

Input
You will be given a starting guess of the three matrices; transition matrix, emission matrix, and initial state probability distribution followed by the number of emissions and the sequence of emissions itself. The initial state probability distribution is a row vector encoded as a matrix with only one row. Each matrix is given on a separate line with the number of rows and columns followed by the matrix elements (ordered row by row). Note that the rows and column size can be different from the sample input. It is assumed that there are M different discrete emission types and these are indexed 0 through M-1 in the emission sequence. For example, if there were M=3 possible different emissions (could be the three colours red, green and blue for example), they would be identified by 0, 1 and 2 in the emission sequence.

Output
You should output the estimated transition matrix and emission matrix on one line each in the same matrix format as they were given, including the dimensions. Do not output the estimated initial state probability distribution.
"""
import sys
import tools
import copy


class EstimateModel():
    def __init__(self, A, B, pi, emissions):
        self.A = A                          # transition matrix
        self.B = B                          # emission matrix
        self.pi = pi                        # initial state matrix
        self.emissions = emissions[1:]      # emission sequence

        self.N = len(A[0])                  # number of states
        self.M = len(B[0])                  # number of possible emissions
        self.T = len(self.emissions)        # number of emissions in sequence

    def forward(self):
        """
        The probability of ending up in a particular state after seeing the first t observations, given the current model parameters.
        """
        alpha = tools.matrix_initialization(self.T, self.N)
        scale = tools.matrix_initialization(self.T, 1)

        for i in range(self.N):
            alpha[0][i] = self.pi[0][i] * self.B[i][self.emissions[0]]
        scale[0] = 1 / sum(alpha[0]) if sum(alpha[0]) != 0 else 0
        for i in range(self.N):
            alpha[0][i] *= scale[0]

        for t in range(1, self.T):
            for j in range(self.N):
                sum_alpha_A = sum(alpha[t - 1][i] * self.A[i][j] for i in range(self.N))
                alpha[t][j] = sum_alpha_A * self.B[j][self.emissions[t]]

            scale[t] = 1 / sum(alpha[t]) if sum(alpha[t]) != 0 else 0

            for i in range(self.N):
                alpha[t][i] *= scale[t]

        return alpha, scale

    def backward(self, c):
        """
        he probability of seeing the remaining observations after time point t, starting from a particular state.
        """
        beta = tools.matrix_initialization(self.T, self.N)

        for i in range(self.N):
            beta[self.T - 1][i] = c[self.T - 1]

        for t in range(self.T - 2, -1, -1):
            for i in range(self.N):
                for j in range(self.N):
                    beta[t][i] += self.A[i][j] * self.B[j][self.emissions[t + 1]] * beta[t + 1][j]
                beta[t][i] *= c[t]

        return beta

    def compute_gamma_digamma(self, alpha, beta):
        gamma = tools.matrix_initialization(self.T, self.N)
        digamma = tools.matrix_initialization_3d(self.T - 1, self.N, self.N)

        for t in range(self.T - 1):
            denom = 0
            for i in range(self.N):
                denom += alpha[t][i] * beta[t][i]

            for i in range(self.N):
                gamma[t][i] = alpha[t][i] * (beta[t][i] / denom if denom != 0 else 0)
                for j in range(self.N):
                    digamma[t][i][j] = alpha[t][i] * self.A[i][j] * self.B[j][self.emissions[t + 1]] * beta[t + 1][j]

        return gamma, digamma

    def re_estimate(self, gamma, di_gamma):
        for i in range(self.N):
            self.pi[0][i] = gamma[0][i]

        for i in range(self.N):
            denominator = 0
            for t in range(self.T - 1):
                denominator += gamma[t][i]

            for j in range(self.N):
                numerator = 0
                for t in range(self.T - 1):
                    numerator += di_gamma[t][i][j]

                self.A[i][j] = numerator / denominator if denominator != 0 else 0

        for i in range(self.N):
            denominator = 0
            for t in range(self.T):
                denominator += gamma[t][i]
            for k in range(self.M):
                numerator = 0
                for t in range(self.T):
                    if self.emissions[t] == k:
                        numerator += gamma[t][i]
                self.B[i][k] = numerator / denominator if denominator != 0 else 0

    def calculate_norm(self, matrix, prev_matrix):
        total = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                total += (matrix[i][j] - prev_matrix[i][j]) ** 2
        return total ** 0.5

    def fit(self):
        # Expectation-Maximization (EM)
        prev_A = copy.deepcopy(self.A)
        prev_B = copy.deepcopy(self.B)
        convergence_threshold = 0.001

        for _ in range(30):
            # Expectation Step
            alpha, c = self.forward()
            beta = self.backward(c)

            # Maximization
            gamma, di_gamma = self.compute_gamma_digamma(alpha, beta)

            self.re_estimate(gamma, di_gamma)

            delta_A = self.calculate_norm(self.A, prev_A)
            delta_B = self.calculate_norm(self.B, prev_B)

            if delta_A < convergence_threshold and delta_B < convergence_threshold:
                break

            prev_A = copy.deepcopy(self.A)
            prev_B = copy.deepcopy(self.B)


def main(input_data):
    input_data = input_data.split("\n")

    A = tools.read_matrix(input_data[0])
    B = tools.read_matrix(input_data[1])
    pi = tools.read_matrix(input_data[2])
    emission_sequence = list(map(int, input_data[3].split()))

    hmm = EstimateModel(A, B, pi, emission_sequence)

    hmm.fit()

    tools.print_matrix(hmm.A)
    tools.print_matrix(hmm.B)


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
        input_data_4 = """4 4 0.7 0.1 0.1 0.1 0.1 0.7 0.1 0.1 0.1 0.1 0.7 0.1 0.1 0.1 0.1 0.7
4 3 0.5 0.3 0.2 0.4 0.3 0.3 0.3 0.4 0.3 0.2 0.4 0.4
1 4 1.0 0.0 0.0 0.0
15 0 1 2 1 0 2 1 0 1 2 0 1 2 1 0

"""
        input_data_5 = """3 3 0.6 0.2 0.2 0.3 0.6 0.1 0.1 0.2 0.7
3 4 0.5 0.2 0.2 0.1 0.1 0.3 0.3 0.3 0.4 0.2 0.1 0.3
1 3 1.0 0.0 0.0
12 0 1 2 3 0 2 3 1 0 3 1 2 0
"""
    else:
        input_data = sys.stdin.read()

    output_result = main(input_data)

    output_data_1 = """          4 4 0.545455 0.454545 0.0 0.0 0.0 0.506173 0.493827 0.0 0.0 0.0 0.504132 0.495868 0.478088 0.0 0.0 0.521912 
4 4 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 
"""
    output_data_2 = """          4 4 0.0 0.694045 0.070896 0.235060 0.684412 0.0 0.228137 0.087451 0.105932 0.271192 0.0 0.622876 0.266083 0.060087 0.673830 0.0 
4 4 1.0 0.0 0.0 0.0 0.0 0.999980 0.0 0.000020 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 
"""
    output_data_3 = """          3 3 0.845052 0.0774739 0.0774739 0.0841673 0.814073 0.101759 0.0841673 0.101759 0.814073 
3 2 0.684137 0.315863 0.126646 0.873354 0.126646 0.873354 
"""

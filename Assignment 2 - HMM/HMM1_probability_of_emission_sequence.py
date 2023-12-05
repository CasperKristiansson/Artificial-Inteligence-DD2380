"""
Problem B
HMM1 Probability of Emission Sequence
In this task you should show that you know how to calculate the probability to observe a certain emission sequence given a HMM model. You will be given the HMM model and a sequence of observations (aka emissions, events, etc) and your task is to calculate the probability for this sequence.

Input
You will be given three matrices; transition matrix, emission matrix, and initial state probability distribution followed by the number of emissions and the sequence of emissions itself. The initial state probability distribution is a row vector encoded as a matrix with only one row. Each matrix is given on a separate line with the number of rows and columns followed by the matrix elements (ordered row by row). Note that the rows and column size can be different from the sample input. It is assumed that there are M different discrete emission types and these are indexed 0 through M-1 in the emission sequence. For example, if there were M=3 possible different emissions (could be the three colours red, green and blue for example), they would be identified by 0, 1 and 2 in the emission sequence.

Output
You should output the probability of the given sequence as a single scalar.

Sample Input 1:
4 4 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0 
4 4 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9 
1 4 1.0 0.0 0.0 0.0 
8 0 1 2 3 0 1 2 3 

Sample Output 1:
0.090276 
"""
import numpy as np
import sys


def read_matrix(matrix_raw):
    elements = list(map(float, matrix_raw.split()))
    rows, cols = int(elements[0]), int(elements[1])
    matrix = np.array(elements[2:]).reshape(rows, cols)

    return matrix


def main(input_data):
    input_data = input_data.split("\n")

    transition_matrix = read_matrix(input_data[0])
    number_of_states = transition_matrix.shape[0]

    emission_matrix = read_matrix(input_data[1])
    initial_state_matrix = read_matrix(input_data[2])

    emission_sequence = list(map(int, input_data[3].split()))
    number_of_emissions = emission_sequence[0]
    emission_sequence = np.array(emission_sequence[1:])

    # Initialization
    alpha = np.zeros((number_of_emissions, number_of_states))
    alpha[0, :] = initial_state_matrix * emission_matrix[:, emission_sequence[0]]

    # Induction
    for t in range(1, number_of_emissions):
        for j in range(number_of_states):
            alpha[t, j] = np.sum(alpha[t - 1, :] * transition_matrix[:, j]) * emission_matrix[j, emission_sequence[t]]

    # Termination
    probability_sequence = np.sum(alpha[number_of_emissions - 1, :])

    return f"{probability_sequence:.6f}"


if __name__ == "__main__":
    if True:
        input_data = """4 4 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0 
4 4 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9 
1 4 1.0 0.0 0.0 0.0 
8 0 1 2 3 0 1 2 3 
    """
    else:
        input_data = sys.stdin.read()

    output_result = main(input_data)

    print(output_result)

    output_data = """0.090276 """

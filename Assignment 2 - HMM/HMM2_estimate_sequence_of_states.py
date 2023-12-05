"""
Problem C
HMM2 Estimate Sequence of States
In this task you should show that you know how to calculate the most likely sequence of (hidden) states that the system moves through given an emission sequence and an HMM model.

Input
You will be given three matrices; transition matrix, emission matrix, and initial state probability distribution followed by the number of emissions and the sequence of emissions itself. The initial state probability distribution is a row vector encoded as a matrix with only one row. Each matrix is given on a separate line with the number of rows and columns followed by the matrix elements (ordered row by row). Note that the rows and column size can be different from the sample input. It is assumed that there are M different discrete emission types and these are indexed 0 through M-1 in the emission sequence. For example, if there were M=3 possible different emissions (could be the three colours red, green and blue for example), they would be identified by 0, 1 and 2 in the emission sequence.

Output
You should output the most probable sequence of states as zero-based indices separated by spaces. Do not output the length of the sequence.

Sample Input 1:
4 4 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0 
4 4 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9 
1 4 1.0 0.0 0.0 0.0 
4 1 1 2 2 

Sample Output 1:
0 1 2 1
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

    delta = np.zeros((number_of_emissions, number_of_states))
    path = np.zeros((number_of_emissions, number_of_states), dtype=int)

    # Initialization step
    delta[0, :] = initial_state_matrix * emission_matrix[:, emission_sequence[0]]

    # Recursion step
    for t in range(1, number_of_emissions):
        for j in range(number_of_states):
            prob = delta[t - 1] * transition_matrix[:, j] * emission_matrix[j, emission_sequence[t]]
            delta[t, j] = np.max(prob)
            path[t, j] = np.argmax(prob)

    # Termination and path-backtracking
    last_state = np.argmax(delta[-1, :])
    number_of_steps = path.shape[0]

    best_path = [last_state]
    for t in range(number_of_steps - 1, 0, -1):
        last_state = path[t, last_state]
        best_path.insert(0, last_state)

    return " ".join(map(str, best_path))


if __name__ == "__main__":
    if True:
        input_data = """4 4 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0 
4 4 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9 
1 4 1.0 0.0 0.0 0.0 
4 1 1 2 2 
    """
    else:
        input_data = sys.stdin.read()

    output_result = main(input_data)

    print(output_result)

    output_data = """0 1 2 1 """

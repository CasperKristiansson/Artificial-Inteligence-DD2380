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
import sys
import tools


def main(input_data):
    input_data = input_data.split("\n")

    transition_matrix = tools.read_matrix(input_data[0])
    emission_matrix = tools.read_matrix(input_data[1])
    initial_state_matrix = tools.read_matrix(input_data[2])
    emission_sequence = list(map(int, input_data[3].split()))

    number_of_states = len(initial_state_matrix[0])
    number_of_emissions = emission_sequence[0]

    emission_sequence = emission_sequence[1:]

    delta = tools.matrix_initialization(number_of_emissions, number_of_states)
    path = tools.matrix_initialization(number_of_emissions, number_of_states)

    # Viterbi - finding the most likely sequence of hidden states

    # Initialization step
    for j in range(number_of_states):
        delta[0][j] = initial_state_matrix[0][j] * emission_matrix[j][emission_sequence[0]]

    # Recursion step
    for t in range(1, number_of_emissions):
        for j in range(number_of_states):
            prob = [delta[t - 1][i] * transition_matrix[i][j] * emission_matrix[j][emission_sequence[t]] for i in range(number_of_states)]
            delta[t][j] = max(prob)
            path[t][j] = prob.index(max(prob))

    # Termination and path-backtracking
    last_state = delta[number_of_emissions - 1].index(max(delta[number_of_emissions - 1]))

    best_path = [last_state]
    for t in range(number_of_emissions - 1, 0, -1):
        last_state = path[t][last_state]
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

"""
Problem A
HMM0 Next Emission Distribution
In this task you should show that you know how to predict how the system will evolve over time and estimate the probability for different emissions / events in the system i.e. what can be observed from the HMM. You will be given the state probability distribution (i.e. the probability that the system is in each of the N states), the transition matrix (i.e. the matrix that gives the probability to transition from one state to another) and the emission matrix (i.e. the matrix that gives the probability for the different emissions / events / observations given a certain state).

More specifically, given the current state probability distribution what is the probability for the different emissions after the next transition (i.e. after the system has made a single transition)?

Input
You will be given three matrices (in this order); transition matrix, emission matrix, and initial state probability distribution. The initial state probability distribution is a row vector encoded as a matrix with only one row. Each matrix is given on a separate line with the number of rows and columns followed by the matrix elements (ordered row by row). Note that the rows and column size can be different from the sample input.

Output
You should output the emission probability distribution on a single line in the same matrix format, including the dimensions.

Sample Input 1:
4 4 0.2 0.5 0.3 0.0 0.1 0.4 0.4 0.1 0.2 0.0 0.4 0.4 0.2 0.3 0.0 0.5
4 3 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.2 0.6 0.2
1 4 0.0 0.0 0.0 1.0

Sample Output 1:
1 3 0.3 0.6 0.1
"""
import sys
import tools


def main(input_data):
    input_data = input_data.split("\n")

    transition_matrix = tools.read_matrix(input_data[0])
    emission_matrix = tools.read_matrix(input_data[1])
    initial_state_matrix = tools.read_matrix(input_data[2])

    emission_distribution = tools.matrix_multiply(initial_state_matrix, tools.matrix_multiply(transition_matrix, emission_matrix))
    rows, cols = tools.matrix_dimensions(emission_distribution)

    emission_distribution = tools.matrix_round(emission_distribution)

    flattened_list = [item for sublist in emission_distribution for item in sublist]

    return f"{rows} {cols} " + " ".join(map(str, flattened_list))


if __name__ == "__main__":
    if True:
        input_data = """4 4 0.2 0.5 0.3 0.0 0.1 0.4 0.4 0.1 0.2 0.0 0.4 0.4 0.2 0.3 0.0 0.5
4 3 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.2 0.6 0.2
1 4 0.0 0.0 0.0 1.0
    """
    else:
        input_data = sys.stdin.read()

    output_result = main(input_data)

    print(output_result)

    output_data = """1 3 0.3 0.6 0.1"""

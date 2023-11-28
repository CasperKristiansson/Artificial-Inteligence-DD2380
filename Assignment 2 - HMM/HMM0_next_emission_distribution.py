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
    emission_matrix = read_matrix(input_data[1])
    initial_state_matrix = read_matrix(input_data[2])

    emission_distribution = np.dot(initial_state_matrix, np.dot(transition_matrix, emission_matrix))
    rows, cols = emission_distribution.shape
    emission_distribution = np.round(emission_distribution, decimals=2)

    return f"{rows} {cols} " + " ".join(map(str, emission_distribution.flatten()))


if __name__ == "__main__":
    if False:
        input_data = """4 4 0.2 0.5 0.3 0.0 0.1 0.4 0.4 0.1 0.2 0.0 0.4 0.4 0.2 0.3 0.0 0.5
4 3 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.2 0.6 0.2
1 4 0.0 0.0 0.0 1.0
    """
    # 0.2 0.5 0.3 0.0
    # 0.1 0.4 0.4 0.1
    # 0.2 0.0 0.4 0.4
    # 0.2 0.3 0.0 0.5

    # 1.0 0.0 0.0
    # 0.0 1.0 0.0
    # 0.0 0.0 1.0
    # 0.2 0.6 0.2

    # 0.0 0.0 0.0 1.0
    else:
        input_data = sys.stdin.read()

    output_result = main(input_data)

    print(output_result)

    output_data = """1 3 0.3 0.6 0.1"""

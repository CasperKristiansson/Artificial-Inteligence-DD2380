def read_matrix(matrix_raw):
    elements = list(map(float, matrix_raw.split()))
    rows, cols = int(elements[0]), int(elements[1])
    elements = elements[2:]
    matrix = [elements[cols * i: cols * (i + 1)] for i in range(rows)]

    if rows == 1:
        matrix = matrix[0]

    return matrix


def matrix_multiply(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


def matrix_dimensions(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    return rows, cols


def matrix_round(matrix, digits=2):
    return [[round(value, digits) for value in row] for row in matrix]


def matrix_initialization(rows, cols, value=0):
    return [[value for _ in range(cols)] for _ in range(rows)]


def matrix_initialization_3d(rows, cols, depth, value=0):
    return [[[value for _ in range(depth)] for _ in range(cols)] for _ in range(rows)]


def print_array(arr):
    flattened = [item for sublist in arr for item in sublist]
    string_elements = [f"{round(elem, 6)}" for elem in flattened]

    print(" ".join(string_elements))

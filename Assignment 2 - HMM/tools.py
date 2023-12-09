def read_matrix(matrix_raw):
    elements = list(map(float, matrix_raw.split()))
    rows, cols = int(elements[0]), int(elements[1])
    elements = elements[2:]
    matrix = [elements[cols * i: cols * (i + 1)] for i in range(rows)]

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
from decimal import Decimal, getcontext
import copy
import numpy as np

getcontext().prec = 50


def convert_matrix_to_decimals(matrix):
    """Конвертирует матрицу в Decimal"""
    if not matrix or len(matrix) == 0:
        return []

    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    decimal_matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(Decimal(str(float(matrix[i][j]))))
        decimal_matrix.append(row)
    return decimal_matrix


def generate_matrix(rows, cols, value=Decimal(0)):
    """Создает матрицу заданного размера"""
    return [[value for _ in range(cols)] for _ in range(rows)]


def matrix_multiply(A, B):
    """Умножение матриц любых совместимых размеров"""
    if not A or not B:
        return []

    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError(f"Несовместимые размеры матриц для умножения: {cols_A} != {rows_B}")

    result = generate_matrix(rows_A, cols_B)
    for i in range(rows_A):
        for j in range(cols_B):
            total = Decimal(0)
            for k in range(cols_A):
                total += A[i][k] * B[k][j]
            result[i][j] = total
    return result


def transpose(matrix):
    """Транспонирование для матриц любых размеров"""
    if not matrix:
        return []

    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    result = generate_matrix(cols, rows)
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    return result


def column(matrix, j):
    """Извлекает столбец j из матрицы"""
    if not matrix:
        return []
    return [matrix[i][j] for i in range(len(matrix))]


def row(matrix, i):
    """Извлекает строку i из матрицы"""
    if not matrix or i >= len(matrix):
        return []
    return matrix[i].copy()


def set_column(matrix, j, vector):
    """Устанавливает столбец j матрицы в vector"""
    if not matrix or len(vector) != len(matrix):
        return matrix

    for i in range(len(matrix)):
        matrix[i][j] = Decimal(vector[i])
    return matrix


def dot_product(vector1, vector2):
    """Скалярное произведение векторов"""
    if len(vector1) != len(vector2):
        raise ValueError("Векторы должны иметь одинаковую длину")

    result = Decimal(0)
    for i in range(len(vector1)):
        result += Decimal(vector1[i]) * Decimal(vector2[i])
    return result


def vector_norm(vector):
    """Норма вектора"""
    return dot_product(vector, vector).sqrt()


def vector_add(vec1, vec2):
    """Сложение векторов"""
    if len(vec1) != len(vec2):
        raise ValueError("Векторы должны иметь одинаковую длину")
    return [Decimal(vec1[i]) + Decimal(vec2[i]) for i in range(len(vec1))]


def vector_subtract(vec1, vec2):
    """Вычитание векторов"""
    if len(vec1) != len(vec2):
        raise ValueError("Векторы должны иметь одинаковую длину")
    return [Decimal(vec1[i]) - Decimal(vec2[i]) for i in range(len(vec1))]


def vector_scale(vector, scalar):
    """Умножение вектора на скаляр"""
    return [Decimal(scalar) * Decimal(x) for x in vector]


def project_1_on_2(vector1, vector2):
    """Проекция vector1 на vector2"""
    dot12 = dot_product(vector1, vector2)
    dot22 = dot_product(vector2, vector2)

    if abs(dot22) < Decimal('1e-30'):
        return [Decimal(0) for _ in vector2]

    scale = dot12 / dot22
    return vector_scale(vector2, scale)


def gram_schmidt(A):
    """алгоритм Грама-Шмидта"""
    if not A:
        return [], []

    m, n = len(A), len(A[0])
    Q = generate_matrix(m, n)
    R = generate_matrix(n, n)

    V = [column(A, j) for j in range(n)]

    for j in range(n):
        # Ортогонализация относительно предыдущих столбцов
        for k in range(j):
            R[k][j] = dot_product(column(Q, k), V[j])
            proj = project_1_on_2(V[j], column(Q, k))
            V[j] = vector_subtract(V[j], proj)

        # Норма текущего столбца
        norm = vector_norm(V[j])

        if norm < Decimal('1e-15'):  # Линейно зависимый столбец
            # Используем базисный вектор
            for i in range(m):
                Q[i][j] = Decimal(1.0 if i == j else 0.0)
            R[j][j] = Decimal(0)
        else:
            # Нормализуем
            for i in range(m):
                Q[i][j] = V[j][i] / norm
            R[j][j] = norm

    return Q, R


def gram_schmidt_orthonormalization(matrix):
    """Ортонормализация Грама-Шмидта"""
    Q, _ = gram_schmidt(matrix)
    return Q


def verify_QR_decomposition(A, Q, R):
    """Проверяет корректность QR-разложения"""
    if not A or not Q or not R:
        return False

    n = len(A)

    # 1. Проверка: Q * R ≈ A
    QR = matrix_multiply(Q, R)
    error_reconstruction = Decimal(0)
    for i in range(n):
        for j in range(len(A[0])):
            error_reconstruction += (QR[i][j] - A[i][j]) ** 2
    error_reconstruction = error_reconstruction.sqrt()

    # 2. Проверка ортогональности Q: Q^T * Q ≈ I
    QT = transpose(Q)
    QTQ = matrix_multiply(QT, Q)

    error_orthogonality = Decimal(0)
    for i in range(len(QTQ)):
        for j in range(len(QTQ[0])):
            expected = Decimal(1.0) if i == j else Decimal(0.0)
            error_orthogonality += (QTQ[i][j] - expected) ** 2
    error_orthogonality = error_orthogonality.sqrt()

    print(f"Ошибка восстановления (Q*R - A): {float(error_reconstruction):.2e}")
    print(f"Ошибка ортогональности (Q^T*Q - I): {float(error_orthogonality):.2e}")

    return error_reconstruction < Decimal('1e-10') and error_orthogonality < Decimal('1e-10')


def decimal_to_numpy(decimal_list):
    """Конвертирует список списков Decimal в numpy array"""
    if not decimal_list:
        return np.array([])
    return np.array([[float(x) for x in row] for row in decimal_list])


def numpy_to_decimal(numpy_array):
    """Конвертирует numpy array в матрицу Decimal"""
    if numpy_array.size == 0:
        return []
    return convert_matrix_to_decimals(numpy_array.tolist())


def robust_svd(A, max_iterations=1000, tolerance=Decimal('1e-20')):
    """SVD через QR-алгоритм"""
    if isinstance(A, np.ndarray):
        A = A.tolist()
    if not A or len(A) == 0 or len(A[0]) == 0:
        raise ValueError("Пустая матрица")

    A_dec = convert_matrix_to_decimals(A)
    m, n = len(A_dec), len(A_dec[0])

    print(f"Вычисление SVD для матрицы {m}x{n}")

    # Определяем, с какой матрицей работать (A^T A или A A^T)
    if m >= n:
        # Случай "высокой" матрицы: работаем с A^T A (n x n)
        print("Используем A^T A (высокая матрица)")
        return _svd_tall_matrix(A_dec, max_iterations, tolerance)
    else:
        # Случай "широкой" матрицы: работаем с A A^T (m x m)
        print("Используем A A^T (широкая матрица)")
        return _svd_wide_matrix(A_dec, max_iterations, tolerance)


def _svd_tall_matrix(A, max_iterations, tolerance):
    """SVD для высокой матрицы (m >= n)"""
    m, n = len(A), len(A[0])

    # Вычисляем A^T A
    AT = transpose(A)
    ATA = matrix_multiply(AT, A)

    # QR-алгоритм для нахождения собственных векторов A^T A
    V = generate_matrix(n, n)
    for i in range(n):
        V[i][i] = Decimal(1.0)

    temp_ATA = copy.deepcopy(ATA)

    print("Запуск QR-алгоритма...")
    for iteration in range(max_iterations):
        Q, R = gram_schmidt(temp_ATA)
        temp_ATA = matrix_multiply(R, Q)
        V = matrix_multiply(V, Q)

        # Проверка сходимости
        off_diag_norm = Decimal(0)
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diag_norm += temp_ATA[i][j] ** 2
        off_diag_norm = off_diag_norm.sqrt()

        if iteration % 100 == 0:
            print(f"Итерация {iteration}, норма недиагональных: {float(off_diag_norm):.2e}")

        if off_diag_norm < tolerance:
            print(f"Сходимость достигнута на итерации {iteration}")
            break
    else:
        print(f"Достигнуто максимальное число итераций {max_iterations}")

    # Сингулярные значения (корни из собственных значений A^T A)
    Sigma = generate_matrix(m, n)
    singular_values = []
    for i in range(n):
        eigenval = max(Decimal(0), temp_ATA[i][i])  # Собственные значения неотрицательны
        sigma_val = eigenval.sqrt()
        Sigma[i][i] = sigma_val
        singular_values.append(sigma_val)

    # Сортируем сингулярные значения по убыванию
    sorted_indices = sorted(range(n), key=lambda i: singular_values[i], reverse=True)

    # Переупорядочиваем V и Sigma согласно сортировке
    V_sorted = generate_matrix(n, n)
    Sigma_sorted = generate_matrix(m, n)

    for new_idx, old_idx in enumerate(sorted_indices):
        # Переупорядочиваем столбцы V
        for i in range(n):
            V_sorted[i][new_idx] = V[i][old_idx]
        # Переупорядочиваем сингулярные значения
        Sigma_sorted[new_idx][new_idx] = Sigma[old_idx][old_idx]

    # Вычисляем U
    U = generate_matrix(m, m)
    AV_sorted = matrix_multiply(A, V_sorted)

    for i in range(n):
        if Sigma_sorted[i][i] > tolerance:
            factor = Decimal(1) / Sigma_sorted[i][i]
            for j in range(m):
                U[j][i] = AV_sorted[j][i] * factor

    # Ортогонализуем оставшиеся столбцы U (если m > n)
    if m > n:
        for i in range(n, m):
            # Начинаем с базисного вектора
            for j in range(m):
                U[j][i] = Decimal(1.0) if j == i else Decimal(0.0)

            # Ортогонализуем относительно предыдущих столбцов
            for k in range(i):
                proj = project_1_on_2(column(U, i), column(U, k))
                for j in range(m):
                    U[j][i] -= proj[j]

            # Нормализуем
            norm = vector_norm(column(U, i))
            if norm > tolerance:
                for j in range(m):
                    U[j][i] /= norm

    # Финальная ортогонализация U
    U, _ = gram_schmidt(U)

    return decimal_to_numpy(U), decimal_to_numpy(Sigma_sorted), decimal_to_numpy(transpose(V_sorted))


def _svd_wide_matrix(A, max_iterations, tolerance):
    """SVD для широкой матрицы (m < n) через A A^T"""
    m, n = len(A), len(A[0])

    # Вычисляем A A^T
    AT = transpose(A)
    AAT = matrix_multiply(A, AT)

    # QR-алгоритм для A A^T
    U = generate_matrix(m, m)
    for i in range(m):
        U[i][i] = Decimal(1.0)

    temp_AAT = copy.deepcopy(AAT)

    print("Запуск QR-алгоритма для A A^T...")
    for iteration in range(max_iterations):
        Q, R = gram_schmidt(temp_AAT)
        temp_AAT = matrix_multiply(R, Q)
        U = matrix_multiply(U, Q)

        # Проверка сходимости
        off_diag_norm = Decimal(0)
        for i in range(m):
            for j in range(m):
                if i != j:
                    off_diag_norm += temp_AAT[i][j] ** 2
        off_diag_norm = off_diag_norm.sqrt()

        if iteration % 100 == 0:
            print(f"Итерация {iteration}, норма недиагональных: {float(off_diag_norm):.2e}")

        if off_diag_norm < tolerance:
            print(f"Сходимость достигнута на итерации {iteration}")
            break
    else:
        print(f"Достигнуто максимальное число итераций {max_iterations}")

    # Сингулярные значения
    Sigma = generate_matrix(m, n)
    singular_values = []
    for i in range(m):
        eigenval = max(Decimal(0), temp_AAT[i][i])
        sigma_val = eigenval.sqrt()
        Sigma[i][i] = sigma_val
        singular_values.append(sigma_val)

    # Сортируем
    sorted_indices = sorted(range(m), key=lambda i: singular_values[i], reverse=True)

    U_sorted = generate_matrix(m, m)
    Sigma_sorted = generate_matrix(m, n)

    for new_idx, old_idx in enumerate(sorted_indices):
        for i in range(m):
            U_sorted[i][new_idx] = U[i][old_idx]
        Sigma_sorted[new_idx][new_idx] = Sigma[old_idx][old_idx]

    # Вычисляем V
    V = generate_matrix(n, n)
    UT_A = matrix_multiply(transpose(U_sorted), A)

    for i in range(m):
        if Sigma_sorted[i][i] > tolerance:
            factor = Decimal(1) / Sigma_sorted[i][i]
            for j in range(n):
                V[j][i] = UT_A[i][j] * factor

    # Ортогонализуем оставшиеся столбцы V (если n > m)
    if n > m:
        for i in range(m, n):
            for j in range(n):
                V[j][i] = Decimal(1.0) if j == i else Decimal(0.0)

            for k in range(i):
                proj = project_1_on_2(column(V, i), column(V, k))
                for j in range(n):
                    V[j][i] -= proj[j]

            norm = vector_norm(column(V, i))
            if norm > tolerance:
                for j in range(n):
                    V[j][i] /= norm

    V, _ = gram_schmidt(V)

    return decimal_to_numpy(U_sorted), decimal_to_numpy(Sigma_sorted), decimal_to_numpy(transpose(V))
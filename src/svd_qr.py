import numpy as np
import copy


def generate_matrix(rows, cols, value=0.0):
    """Создает матрицу заданного размера"""
    return np.full((rows, cols), value, dtype=np.float64)


def column(matrix, j):
    """Извлекает столбец j из матрицы"""
    return matrix[:, j].copy()


def row(matrix, i):
    """Извлекает строку i из матрицы"""
    return matrix[i, :].copy()


def set_column(matrix, j, vector):
    """Устанавливает столбец j матрицы в vector"""
    matrix[:, j] = vector
    return matrix


def dot_product(vector1, vector2):
    """Скалярное произведение векторов"""
    return np.dot(vector1, vector2)


def vector_norm(vector):
    """Норма вектора"""
    return np.linalg.norm(vector)


def vector_add(vec1, vec2):
    """Сложение векторов"""
    return vec1 + vec2


def vector_subtract(vec1, vec2):
    """Вычитание векторов"""
    return vec1 - vec2


def vector_scale(vector, scalar):
    """Умножение вектора на скаляр"""
    return vector * scalar


def project_1_on_2(vector1, vector2):
    """Проекция vector1 на vector2"""
    dot12 = dot_product(vector1, vector2)
    dot22 = dot_product(vector2, vector2)

    if abs(dot22) < 1e-30:
        return np.zeros_like(vector2)

    scale = dot12 / dot22
    return vector_scale(vector2, scale)


def gram_schmidt(A):
    """Алгоритм Грама-Шмидта"""
    if A.size == 0:
        return np.array([]), np.array([])

    m, n = A.shape
    Q = generate_matrix(m, n)
    R = generate_matrix(n, n)

    V = [column(A, j) for j in range(n)]

    for j in range(n):
        # Ортогонализация относительно предыдущих столбцов
        for k in range(j):
            R[k, j] = dot_product(column(Q, k), V[j])
            proj = project_1_on_2(V[j], column(Q, k))
            V[j] = vector_subtract(V[j], proj)

        # Норма текущего столбца
        norm = vector_norm(V[j])

        if norm < 1e-15:  # Линейно зависимый столбец
            # Используем базисный вектор
            Q[:, j] = np.eye(m, n)[:, j] if j < m else np.zeros(m)
            R[j, j] = 0.0
        else:
            # Нормализуем
            Q[:, j] = V[j] / norm
            R[j, j] = norm

    return Q, R


def gram_schmidt_orthonormalization(matrix):
    """Ортонормализация Грама-Шмидта"""
    Q, _ = gram_schmidt(matrix)
    return Q


def verify_QR_decomposition(A, Q, R):
    """Проверяет корректность QR-разложения"""
    if A.size == 0 or Q.size == 0 or R.size == 0:
        return False

    # 1. Проверка: Q * R ≈ A
    QR = Q @ R
    error_reconstruction = np.linalg.norm(QR - A)

    # 2. Проверка ортогональности Q: Q^T * Q ≈ I
    QTQ = Q.T @ Q
    identity = np.eye(Q.shape[1])
    error_orthogonality = np.linalg.norm(QTQ - identity)

    print(f"Ошибка восстановления (Q*R - A): {error_reconstruction:.2e}")
    print(f"Ошибка ортогональности (Q^T*Q - I): {error_orthogonality:.2e}")

    return error_reconstruction < 1e-10 and error_orthogonality < 1e-10


def robust_svd(A, max_iterations=1000, tolerance=1e-15):
    """SVD через QR-алгоритм"""
    if A.size == 0:
        raise ValueError("Пустая матрица")

    A = np.array(A, dtype=np.float64)
    m, n = A.shape

    print(f"Вычисление SVD для матрицы {m}x{n}")

    # Определяем, с какой матрицей работать (A^T A или A A^T)
    if m >= n:
        # Случай "высокой" матрицы: работаем с A^T A (n x n)
        print("Используем A^T A (высокая матрица)")
        return _svd_tall_matrix(A, max_iterations, tolerance)
    else:
        # Случай "широкой" матрицы: работаем с A A^T (m x m)
        print("Используем A A^T (широкая матрица)")
        return _svd_wide_matrix(A, max_iterations, tolerance)


def _svd_tall_matrix(A, max_iterations, tolerance):
    """SVD для высокой матрицы (m >= n)"""
    m, n = A.shape

    # Вычисляем A^T A
    ATA = A.T @ A

    # QR-алгоритм для нахождения собственных векторов A^T A
    V = np.eye(n, dtype=np.float64)
    temp_ATA = ATA.copy()

    print("Запуск QR-алгоритма...")
    for iteration in range(max_iterations):
        Q, R = gram_schmidt(temp_ATA)
        temp_ATA = R @ Q
        V = V @ Q

        # Проверка сходимости
        off_diag = temp_ATA - np.diag(np.diag(temp_ATA))
        off_diag_norm = np.linalg.norm(off_diag)

        if iteration % 100 == 0:
            print(f"Итерация {iteration}, норма недиагональных: {off_diag_norm:.2e}")

        if off_diag_norm < tolerance:
            print(f"Сходимость достигнута на итерации {iteration}")
            break
    else:
        print(f"Достигнуто максимальное число итераций {max_iterations}")

    # Сингулярные значения (корни из собственных значений A^T A)
    Sigma = np.zeros((m, n), dtype=np.float64)
    singular_values = []
    for i in range(n):
        eigenval = max(0.0, temp_ATA[i, i])  # Собственные значения неотрицательны
        sigma_val = np.sqrt(eigenval)
        Sigma[i, i] = sigma_val
        singular_values.append(sigma_val)

    # Сортируем сингулярные значения по убыванию
    sorted_indices = np.argsort(singular_values)[::-1]

    # Переупорядочиваем V и Sigma согласно сортировке
    V_sorted = V[:, sorted_indices]
    Sigma_sorted = np.zeros_like(Sigma)
    for new_idx, old_idx in enumerate(sorted_indices):
        Sigma_sorted[new_idx, new_idx] = Sigma[old_idx, old_idx]

    # Вычисляем U
    U = np.zeros((m, m), dtype=np.float64)
    AV_sorted = A @ V_sorted

    for i in range(n):
        if Sigma_sorted[i, i] > tolerance:
            factor = 1.0 / Sigma_sorted[i, i]
            U[:, i] = AV_sorted[:, i] * factor

    # Ортогонализуем оставшиеся столбцы U (если m > n)
    if m > n:
        for i in range(n, m):
            # Начинаем с базисного вектора
            U[:, i] = np.eye(m, m)[:, i]

            # Ортогонализуем относительно предыдущих столбцов
            for k in range(i):
                proj = project_1_on_2(U[:, i], U[:, k])
                U[:, i] -= proj

            # Нормализуем
            norm = vector_norm(U[:, i])
            if norm > tolerance:
                U[:, i] /= norm

    # Финальная ортогонализация U
    U, _ = gram_schmidt(U)

    return U, Sigma_sorted, V_sorted.T


def _svd_wide_matrix(A, max_iterations, tolerance):
    """SVD для широкой матрицы (m < n) через A A^T"""
    m, n = A.shape

    # Вычисляем A A^T
    AAT = A @ A.T

    # QR-алгоритм для A A^T
    U = np.eye(m, dtype=np.float64)
    temp_AAT = AAT.copy()

    print("Запуск QR-алгоритма для A A^T...")
    for iteration in range(max_iterations):
        Q, R = gram_schmidt(temp_AAT)
        temp_AAT = R @ Q
        U = U @ Q

        # Проверка сходимости
        off_diag = temp_AAT - np.diag(np.diag(temp_AAT))
        off_diag_norm = np.linalg.norm(off_diag)

        if iteration % 100 == 0:
            print(f"Итерация {iteration}, норма недиагональных: {off_diag_norm:.2e}")

        if off_diag_norm < tolerance:
            print(f"Сходимость достигнута на итерации {iteration}")
            break
    else:
        print(f"Достигнуто максимальное число итераций {max_iterations}")

    # Сингулярные значения
    Sigma = np.zeros((m, n), dtype=np.float64)
    singular_values = []
    for i in range(m):
        eigenval = max(0.0, temp_AAT[i, i])
        sigma_val = np.sqrt(eigenval)
        Sigma[i, i] = sigma_val
        singular_values.append(sigma_val)

    # Сортируем
    sorted_indices = np.argsort(singular_values)[::-1]

    U_sorted = U[:, sorted_indices]
    Sigma_sorted = np.zeros_like(Sigma)
    for new_idx, old_idx in enumerate(sorted_indices):
        Sigma_sorted[new_idx, new_idx] = Sigma[old_idx, old_idx]

    # Вычисляем V
    V = np.zeros((n, n), dtype=np.float64)
    UT_A = U_sorted.T @ A

    for i in range(m):
        if Sigma_sorted[i, i] > tolerance:
            factor = 1.0 / Sigma_sorted[i, i]
            V[:, i] = UT_A[i, :] * factor

    # Ортогонализуем оставшиеся столбцы V (если n > m)
    if n > m:
        for i in range(m, n):
            V[:, i] = np.eye(n, n)[:, i]

            for k in range(i):
                proj = project_1_on_2(V[:, i], V[:, k])
                V[:, i] -= proj

            norm = vector_norm(V[:, i])
            if norm > tolerance:
                V[:, i] /= norm

    V, _ = gram_schmidt(V)

    return U_sorted, Sigma_sorted, V.T

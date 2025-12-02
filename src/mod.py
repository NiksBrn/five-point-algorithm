import numpy as np
import estimation
import svd_qr

class EssentialMatrix(object):
    
    def __init__(self, points1, points2, K, method='Five'):
        """
        points1: 2D точки на первом изображении (N, 2)
        points2: 2D точки на втором изображении (N, 2)
        K: матрица внутренних параметров 1 камеры
        K: матрица внутренних параметров 2 камеры
        """
        if points1.shape[0] != points2.shape[0]:
            raise ValueError("Different point arrays size!")
        # Нормирование координат при помощи матриц хар-ик камер
        ones = np.ones((points1.shape[0], 1))
        points1 = np.hstack((points1, ones)) @ np.linalg.inv(K).T 
        points2 = np.hstack((points2, ones)) @ np.linalg.inv(K).T

        self.points1 = points1
        self.points2 = points2
        # Нормализация
        # self.points1, self.T1 = normalized(points1)
        # self.points2, self.T2 = normalized(points2)
        # self.T2T_INV = np.linalg.inv(self.T2.T)
        # self.T1T_INV= np.linalg.inv(self.T1.T)
        # self.unNormPoints1 = points1
        # self.unNormPoints2 = points2
        self.numberOfPoints = len(points1)
        self.K = K
        if method == 'Five':
            self.NUMBER_OF_POINTS = 5
            self.estimation = estimation.fivePointsAlg
        elif method == 'Eight':
            self.NUMBER_OF_POINTS = 8
            self.estimation = estimation.eightPointsAlg
    
    
    def sampsonDistance(self, E):
        distances = [None for _ in range(self.numberOfPoints)]
        distances = np.zeros(self.numberOfPoints)

        for i in range(self.numberOfPoints):
            dot1 = self.points1[i]
            dot2 = self.points2[i]

            numerator = (dot2.T @ E @ dot1) ** 2
            # ||T_2^{-T}Ex_1[:2]||^2 + ||T_1^{-T}E^Tx_2[:2]||^2 - так как координаты смещены
            # |T_2^{-T}Ex_1||^2 = a;   ||T_1^{-T}E^Tx_2||^2 = b
            a = self.T2T_INV @ E @ dot1
            b = self.T1T_INV @ E.T @ dot2
            denominator = a[0] ** 2 + a[1] ** 2 + b[0] ** 2 + b[1] ** 2
            
            dist = numerator / denominator if denominator >= 1e-15 else np.inf

            distances[i] = dist   
        
        return distances 
    
    def sampsonDistance_numpy_vectorized(self, E):
        # Векторизованная версия с NumPy
        points1 = self.points1  # [n, 3]
        points2 = self.points2  # [n, 3]
        
        # numerator = (x2^T * E * x1)^2 для всех точек
        Ex1 = points1 @ E.T  # [n, 3]
        x2_Ex1 = np.sum(points2 * Ex1, axis=1)  # [n]
        numerator = x2_Ex1 ** 2  # [n]
        
        # denominator = a[0]^2 + a[1]^2 + b[0]^2 + b[1]^2
        # a = points1 @ (E.T @ self.T2T_INV.T)  # [n, 3]
        # b = points2 @ (E @ self.T1T_INV.T)  # [n, 3]
        a = points1 @ E.T  # [n, 3]
        b = points2 @ E  
        
        
        a_sq = np.sum(a[:, :2] ** 2, axis=1)  # [n]
        b_sq = np.sum(b[:, :2] ** 2, axis=1)  # [n]
        denominator = a_sq + b_sq  # [n]
        
        # Вычисляем расстояния
        distances = np.zeros(self.numberOfPoints)
        mask = denominator >= 1e-15
        distances[mask] = numerator[mask] / denominator[mask]
        distances[~mask] = np.inf
        
        return distances
    
    def ransac(self, maxIterations=1000, threshold=1.0, confidence=0.99):
        """
        maxIterations: максимальное количество итераций алгоритма
        threshold: допустимая дистанция от идеального значения
        minInliers: коэффициент соостветствий для минимального числа инлаеров относительно общего числа точек 
        """
        # threshold = threshold / (fx+fy)/2
        threshold = threshold / ((self.K[0][0] + self.K[1][1]) / 2)
        EBest = None
        inliersBest = []
        errorBest = np.inf
        iteration = 0
        niters = max(maxIterations, 1)
        flag = False
        while iteration < niters:
            if self.numberOfPoints < self.NUMBER_OF_POINTS:
                raise ValueError("Not enough points to estimate Essential matrix")

            # выборка уникальных точек
            indices = np.random.choice(self.numberOfPoints, self.NUMBER_OF_POINTS, replace=False)
            samplePoints1 = self.points1[indices]
            samplePoints2 = self.points2[indices]

            E_array = self.estimation(samplePoints1, samplePoints2)
            for E in E_array:
                if E is None:
                    counter += 1
                    continue

                # dist = self.sampsonDistance(E)
                dist = self.sampsonDistance_numpy_vectorized(E)
                # print(f'Diff opt:\n{np.linalg.norm(dist - distf)}\n')

                inlierMask = dist < (threshold ** 2)
                if not np.any(inlierMask):
                    continue  # инлаеров нет, переход к следующей итерации

                inliersNum = np.sum(inlierMask)
                mean_error = np.mean(dist[inlierMask])

                if inliersNum > max(len(inliersBest), self.NUMBER_OF_POINTS - 1):
                    EBest = E
                    inliersBest = np.where(inlierMask)[0]
                    errorBest = mean_error
                    niters = self.updateNumIters(confidence, ((self.numberOfPoints - inliersNum) / self.numberOfPoints), niters)
            iteration += 1

        self.E = EBest
        if E is None:
            raise ValueError("Couldn't estimate essential matrix")
        
        self.inliers = inliersBest
        self.error = errorBest

    def updateNumIters(self, p, ep, maxIter):
        p = np.clip(p, 0, 1)
        ep = np.clip(ep, 0, 1)
        
        # Вычисляем числитель и знаменатель с double precision
        num = max(1.0 - p, np.finfo(np.float64).min)
        denom = 1.0 - np.power(1.0 - ep, self.NUMBER_OF_POINTS)
        
        if denom < np.finfo(np.float64).min:
            return 0
        
        num_log = np.log(num)
        denom_log = np.log(denom)
        
        if denom_log >= 0 or -num_log >= maxIter * (-denom_log):
            return maxIter
        else:
            d = num_log / denom_log
            t = int(np.round(d))
            return t
        
    
    def decomposeEssentialMatrix(self):
        # E = self.T2.T @ self.E @ self.T1              # Денормализация
        E = self.E
        # Приведение матрицы к рангу 2
        # Алгоритм 5 точек выдает корректные матрицы, дополнительная коррекция сингулярных чисел
        # требуется алгоритму 7 или 8 точек из-за учитывания меньших условий
        if self.NUMBER_OF_POINTS != 5:
            U, S, Vt = svd_qr.robust_svd(E)                   # Декомпозиция
            S = [(S[0] + S[1]) / 2, (S[0] + S[1]) / 2, 0] # Первые два сингулярных числа равны, 3 - равно 0
            E = U @ np.diag(S) @ Vt                       # Обратная композиция
        
        U, S, Vt = svd_qr.robust_svd(E)
        print('E dec\n', S, E)
        W = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
        
        # Декомпозируем в разные случае R, t
        # tx = U @ W @ np.diag(S) @ U.T
        R1 = U @ W.T @ Vt
        R2 = U @ W @ Vt
        
        t = U[:, 2].reshape(3, 1)

        # det(R) = 1 - условие матрицы вращения
        if np.linalg.det(R1) < 0:
            R1 = -R1
        if np.linalg.det(R2) < 0:
            R2 = -R2
        
        # tx - matrix 3*3, t - vector 3d
        # t = np.array([tx[2][1], tx[0][2], tx[1][0]])
        
        self.R, self.t = self.qualifier([(R1, t), (R1, -t),(R2, t), (R2, -t)])
        
    def qualifier(self, variants):
        """
        variants: Массив кортежей вида (R,t) - все возможные кандидаты матрицы вращения и вектора смещения
        """
        distanceThreshold = 50
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        best = None
        bestCount = -1
        # points1 = self.unNormPoints1[self.inliers]
        # points2 = self.unNormPoints2[self.inliers]
        points1 = self.points1[self.inliers]
        points2 = self.points2[self.inliers]
        
        for (R, t) in variants:
            P2 = np.hstack([R, t])
            count = 0
            
            for i in range(len(points1)):
                X = self.triangulatePoints(P1, P2, points1[i], points2[i])
                z1 = X[2]
                z2 = (R @ X + t.reshape(3, ))[2]
                if z1 > 1e-10 and z2 > 1e-10 and z1 < distanceThreshold and z2 < distanceThreshold:
                    count += 1
            
            if count > bestCount:
                bestCount = count
                best = (R, t)
        
        return best

    def triangulatePoints(self, P1, P2, x1, x2):
        """
        DLT triangulation
        """
        A = np.zeros((4, 4))
        
        A[0] = x1[0] * P1[2] - P1[0]
        A[1] = x1[1] * P1[2] - P1[1]
        A[2] = x2[0] * P2[2] - P2[0]
        A[3] = x2[1] * P2[2] - P2[1]
	    
        _, _, Vt = svd_qr.robust_svd(A, 500)
        X = Vt[-1]
        return X[:3] / X[3]


    def evulate(self):
        self.ransac()
        self.decomposeEssentialMatrix()
        return self.R, self.t, (self.error, self.inliers)

def normalized(points):
    mean = np.mean(points, axis=0)
    centered_point = points - mean
    sigma = np.mean(np.sqrt(np.sum(centered_point ** 2, axis=1)))
    
    scale = np.sqrt(2) / sigma if sigma > 1e-10 else 1.0
    
    T = np.array([[scale, 0, -scale * mean[0]],
                [0, scale, -scale * mean[1]],
                [0, 0, 1]])
    
    normalized_points = points @ T.T
    return normalized_points, T

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import numpy as np
from mod import *


def extract_correspondences(img_path1, img_path2, K, fast_threshold=25):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Не удалось загрузить одно из изображений")
    # Детектирование ключевых точек
    fast = cv2.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=True)
    keypoints1 = fast.detect(img1, None)
    pts1 = np.array([kp.pt for kp in keypoints1], dtype=np.float32)

    if len(pts1) == 0:
        raise RuntimeError("Не удалось найти ключевые точки на первом изображении")

    pts2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, pts1, None)
    st = st.reshape(-1)
    pts1_good = pts1[st == 1]
    pts2_good = pts2[st == 1]

    return pts1_good, pts2_good


class MonoVisualOdometry:
    def __init__(self, folder_path, K, param):
        # фиксим возможные проблемы с Qt/Wayland
        os.environ["QT_QPA_PLATFORM"] = os.environ.get("QT_QPA_PLATFORM", "xcb")
        self.param = param
        self.folder_path = folder_path
        self.K = K
        self.photos = self._load_photos()
        self.traj = np.zeros((600, 600, 3), dtype=np.uint8)
        self.pos = np.zeros((3, 1))
        self.R_total = np.eye(3)

    def _load_photos(self):
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        photos = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path)
                  if f.lower().endswith(extensions)]
        photos.sort()
        if len(photos) < 2:
            raise ValueError("Недостаточно изображений для визуальной одометрии")
        return photos

    def process(self):
        print("=== Нажми 'n' чтобы перейти к следующему кадру, 'q' — выход ===")
        prev_pos = self.pos.copy()
        for i in range(len(self.photos) - 1):
            img1_path = self.photos[i]
            img2_path = self.photos[i + 1]
            img2_color = cv2.imread(img2_path)
            if img2_color is None:
                continue

            pts1, pts2 = extract_correspondences(img1_path, img2_path, self.K)

            # --- Матрица эссенции ---
            if self.param == 1:
                E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)
            else:
                R, t, _ = EssentialMatrix(pts1, pts2, K, 'Five').evulate()

            # --- Масштаб (пока фиктивный) ---
            scale = 1.0

            # --- Обновляем позицию ---
            self.pos += scale * self.R_total @ t
            self.R_total = R @ self.R_total

            # --- Обновляем траекторию ---
            scale_factor = 30
            offset = np.array([300, 300])

            draw_x = int(self.pos[0, 0] * scale_factor + offset[0])
            draw_z = int(self.pos[2, 0] * scale_factor + offset[1])

            if i > 0:
                prev_x = int(prev_pos[0, 0] * scale_factor + offset[0])
                prev_z = int(prev_pos[2, 0] * scale_factor + offset[1])
                cv2.line(self.traj, (prev_x, prev_z), (draw_x, draw_z), (0, 255, 0), 1)

            cv2.circle(self.traj, (draw_x, draw_z), 2, (0, 255, 0), -1)
            cv2.rectangle(self.traj, (10, 5), (200, 25), (0, 0, 0), -1)
            cv2.putText(self.traj, f"Frame: {i+1}/{len(self.photos)-1}", (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            prev_pos = self.pos.copy()

            # --- Показываем два окна ---
            cv2.imshow("Frame", img2_color)
            cv2.imshow("Trajectory", self.traj)

            auto_key = cv2.waitKey(50) & 0xFF
            if auto_key == ord('q'):
                cv2.destroyAllWindows()
                return
            
            # --- Управление ---
            # while True:
            #     key = cv2.waitKey(0) & 0xFF
            #     if key == ord('n'):  # след. кадр
            #         break
            #     elif key == ord('q'):  # выход
            #         cv2.destroyAllWindows()
            #         return

        print("=== Обработка завершена ===")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# === Пример использования ===
if __name__ == "__main__":
    focal = 718.8560
    pp = (607.1928, 185.2157)
    K = np.array([[focal, 0, pp[0]],
				[0, focal, pp[1]],
				[0, 0, 1]])

    folder = "./data/image_0/"
    folder2 = './data/104-106'
    vo = MonoVisualOdometry(folder2, K, 2)
    vo.process()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
import pickle
import os
import mediapipe as mp

# 全局配置
MODEL_PATH = knn_model.pkl
LABEL_ENCODER_PATH = label_encoder.pkl
FEATURE_COLS = [
    dist_shoulder_wrist_left,
    dist_shoulder_wrist_right,
    dist_hip_ankle_left,
    dist_hip_ankle_right,
    dist_wrist_wrist,
    dist_ankle_ankle,
    shoulder_hip_ratio,
    hip_knee_angle_left,
    hip_knee_angle_right
]

class SportsAnalysisApp
    def __init__(self, root)
        self.root = root
        self.root.title(KNN姿态识别系统 v1.0)

        # 尝试加载模型
        self.model, self.label_encoder = self.load_model()
        if self.model is None
            messagebox.showerror(错误, 无法加载KNN模型，请确保 knn_model.pkl 和 label_encoder.pkl 存在！)
            self.root.destroy()
            return

        # 初始化 MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

        # 视频流组件
        self.video_frame = tk.LabelFrame(root, text=运动检测)
        self.video_label = tk.Label(self.video_frame)
        self.video_frame.pack(pady=10, padx=10, fill=tk.BOTH)
        self.video_label.pack()

        # 控制面板
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X, padx=10)

        # 动作选择（仅用于提示，实际由KNN输出）
        self.action_var = tk.StringVar()
        self.action_selector = ttk.Combobox(
            control_frame,
            textvariable=self.action_var,
            values=[深蹲, 引体向上],
            state=readonly
        )
        self.action_selector.pack(side=tk.LEFT, padx=5)
        self.action_selector.set(请选择动作（仅参考）)

        # 控制按钮
        self.btn_start = ttk.Button(control_frame, text=开始检测, command=self.start_detection)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_upload = ttk.Button(control_frame, text=上传视频, command=self.upload_video)
        self.btn_upload.pack(side=tk.LEFT, padx=5)

        self.btn_exit = ttk.Button(control_frame, text=退出, command=self.exit_app)
        self.btn_exit.pack(side=tk.LEFT, padx=5)

        # 数据展示面板
        self.data_panel = tk.LabelFrame(root, text=KNN识别结果)
        self.data_panel.pack(fill=tk.BOTH, padx=10, pady=5)

        self.result_label = tk.Label(self.data_panel, text=当前姿态 --)
        self.result_label.pack(side=tk.LEFT, padx=20)

        self.count_label = tk.Label(self.data_panel, text=完成次数 0)
        self.count_label.pack(side=tk.LEFT, padx=20)

        # 状态变量
        self.camera_active = False
        self.video_active = False
        self.cap = None
        self.video_path = None
        self.selected_action = None

        # EMA & 计数逻辑
        self.ema_alpha = 0.3  # EMA 平滑系数
        self.current_smooth_prob = {}
        self.is_in_action = False
        self.action_count = 0
        self.enter_threshold = 0.7
        self.exit_threshold = 0.3

        # 支持的动作标签（需与训练时一致）
        self.supported_actions = {squat_down, pullup_up}

    def load_model(self)
        if not (os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH))
            return None, None
        with open(MODEL_PATH, 'rb') as f
            model = pickle.load(f)
        with open(LABEL_ENCODER_PATH, 'rb') as f
            le = pickle.load(f)
        return model, le

    def start_detection(self)
        if not self.camera_active
            self.initialize_camera()
            self.camera_active = True
            self.btn_start.config(text=停止检测)
            threading.Thread(target=self.update_frame, daemon=True).start()
        else
            self.release_camera()

    def initialize_camera(self)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def update_frame(self)
        while self.camera_active
            ret, frame = self.cap.read()
            if ret
                processed_frame = self.process_frame(frame)
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

    def process_frame(self, frame)
        # 使用 MediaPipe 提取关键点
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks
            self.result_label.config(text=当前姿态 未检测到人体)
            return frame

        # 提取关键点坐标（归一化到 [0,1]）
        landmarks = []
        h, w, _ = frame.shape
        for lm in results.pose_landmarks.landmark
            landmarks.append([lm.x, lm.y])

        # 构造特征向量
        try
            features = self.extract_features(landmarks)
        except Exception as e
            print(f特征提取失败 {e})
            return frame

        # 预测
        proba = self.model.predict_proba([features])[0]
        labels = self.label_encoder.classes_
        pred_dict = dict(zip(labels, proba))

        # EMA 平滑
        for label in pred_dict
            if label not in self.current_smooth_prob
                self.current_smooth_prob[label] = pred_dict[label]
            else
                self.current_smooth_prob[label] = (
                    self.ema_alpha  pred_dict[label] +
                    (1 - self.ema_alpha)  self.current_smooth_prob[label]
                )

        # 找出最高概率的姿态
        best_label = max(self.current_smooth_prob, key=self.current_smooth_prob.get)
        best_prob = self.current_smooth_prob[best_label]

        # 更新 GUI 显示
        display_name = {
            squat_down 深蹲（下）,
            squat_up 深蹲（上）,
            pullup_up 引体向上（上）,
            pullup_down 引体向上（下）
        }.get(best_label, best_label)

        self.result_label.config(text=f当前姿态 {display_name} ({best_prob.2f}))

        # 状态机计数（仅对完成态触发）
        if best_label in [squat_down, pullup_up]
            if not self.is_in_action and best_prob = self.enter_threshold
                self.is_in_action = True
            elif self.is_in_action and best_prob = self.exit_threshold
                self.action_count += 1
                self.is_in_action = False
                self.count_label.config(text=f完成次数 {self.action_count})
        else
            self.is_in_action = False

        # 绘制结果
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
        )
        cv2.putText(frame, fKNN {display_name}, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return frame

    def extract_features(self, landmarks)
        从归一化关键点构造特征向量
        lm = np.array(landmarks)

        def dist(i, j)
            return np.linalg.norm(lm[i] - lm[j])

        # 关键点索引（MediaPipe Pose）
        # 0 nose, 11 left shoulder, 12 right shoulder
        # 13 left elbow, 14 right elbow
        # 15 left wrist, 16 right wrist
        # 23 left hip, 24 right hip
        # 25 left knee, 26 right knee
        # 27 left ankle, 28 right ankle

        features = []
        # 肩-腕距离
        features.append(dist(11, 15))
        features.append(dist(12, 16))
        # 髋-踝距离
        features.append(dist(23, 27))
        features.append(dist(24, 28))
        # 腕-腕、踝-踝
        features.append(dist(15, 16))
        features.append(dist(27, 28))
        # 躯干比例（肩宽  髋宽）
        shoulder_width = dist(11, 12)
        hip_width = dist(23, 24)
        features.append(shoulder_width  (hip_width + 1e-6))
        # 膝关节角度（简化版）
        features.append(self.calculate_angle(lm[23], lm[25], lm[27]))  # 左腿
        features.append(self.calculate_angle(lm[24], lm[26], lm[28]))  # 右腿

        return features

    def calculate_angle(self, a, b, c)
        计算三点夹角（度）
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine_angle = np.dot(ba, bc)  (np.linalg.norm(ba)  np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def release_camera(self)
        self.camera_active = False
        if self.cap
            self.cap.release()
        self.btn_start.config(text=开始检测)

    def upload_video(self)
        self.video_path = filedialog.askopenfilename(filetypes=[(Video files, .mp4 .avi)])
        if self.video_path
            self.process_video_file()

    def process_video_file(self)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened()
            messagebox.showerror(错误, 无法打开视频文件)
            return

        output_path = self.video_path.replace(.mp4, _knn_output.mp4).replace(.avi, _knn_output.avi)
        fourcc = cv2.VideoWriter_fourcc('mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))

        while cap.isOpened()
            ret, frame = cap.read()
            if not ret
                break
            processed = self.process_frame_for_video(frame)
            out.write(processed)

        cap.release()
        out.release()
        messagebox.showinfo(完成, f处理完成！n输出文件 {output_path})

    def process_frame_for_video(self, frame)
        # 同 process_frame，但不更新 GUI（避免跨线程）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        if not results.pose_landmarks
            return frame
        landmarks = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]
        try
            features = self.extract_features(landmarks)
            proba = self.model.predict_proba([features])[0]
            labels = self.label_encoder.classes_
            pred_dict = dict(zip(labels, proba))
            best_label = max(pred_dict, key=pred_dict.get)
            display_name = {squat_down 深蹲下, pullup_up 引体向上上}.get(best_label, best_label)
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, fKNN {display_name}, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        except
            pass
        return frame

    def exit_app(self)
        self.release_camera()
        self.root.quit()

    def reset_state(self)
        self.current_smooth_prob = {}
        self.is_in_action = False
        self.action_count = 0
        self.count_label.config(text=完成次数 0)


if __name__ == __main__
    root = tk.Tk()
    app = SportsAnalysisApp(root)
    root.protocol(WM_DELETE_WINDOW, app.release_camera)
    root.mainloop()
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
from ultralytics import YOLO


class SportsAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("单人运动姿态评测系统")

        # 加载 YOLOv8 模型
        self.model = YOLO('yolov8n-pose.pt')  # 加载官方预训练模型

        # 视频流组件
        self.video_frame = tk.LabelFrame(root, text="运动检测")
        self.video_label = tk.Label(self.video_frame)
        self.video_frame.pack(pady=10, padx=10, fill=tk.BOTH)
        self.video_label.pack()

        # 控制面板
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X, padx=10)

        # 动作选择
        self.action_var = tk.StringVar()
        self.action_selector = ttk.Combobox(control_frame,
                                            textvariable=self.action_var,
                                            values=["跳绳", "引体向上", "深蹲"])
        self.action_selector.pack(side=tk.LEFT, padx=5)
        self.action_selector.bind("<<ComboboxSelected>>", self.on_action_selected)  # 绑定选择事件

        # 控制按钮
        self.btn_start = ttk.Button(control_frame, text="开始检测", command=self.start_detection, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_upload = ttk.Button(control_frame, text="上传视频", command=self.upload_video, state=tk.DISABLED)
        self.btn_upload.pack(side=tk.LEFT, padx=5)

        # 新增“计数清零”按钮
        self.btn_reset = ttk.Button(control_frame, text="计数清零", command=self.reset_count)
        self.btn_reset.pack(side=tk.LEFT, padx=5)

        # 数据展示面板
        self.data_panel = tk.LabelFrame(root, text="实时数据")
        self.data_panel.pack(fill=tk.BOTH, padx=10, pady=5)

        self.count_label = tk.Label(self.data_panel, text="动作计数: 0")
        self.count_label.pack(side=tk.LEFT, padx=20)

        # 状态变量
        self.camera_active = False
        self.video_active = False
        self.cap = None
        self.video_path = None
        self.selected_action = None  # 当前选择的动作
        self.squat_count = 0  # 深蹲计数
        self.is_start_position = False  # 标记是否处于起始位置

    def on_action_selected(self, event=None):
        """当用户选择动作时触发"""
        selected_action = self.action_var.get()
        if selected_action in ["跳绳", "引体向上", "深蹲"]:
            self.selected_action = selected_action
            self.btn_start.config(state=tk.NORMAL)
            self.btn_upload.config(state=tk.NORMAL)
            self.update_data_panel(selected_action)
        else:
            self.selected_action = None
            self.btn_start.config(state=tk.DISABLED)
            self.btn_upload.config(state=tk.DISABLED)

    def update_data_panel(self, action):
        """根据选择的动作更新实时数据面板"""
        if action in ["跳绳", "引体向上", "深蹲"]:
            self.count_label.config(text="动作计数: 0")

    def start_detection(self):
        if not self.camera_active:
            if not self.selected_action:
                messagebox.showwarning("警告", "请先选择一个运动项目！")
                return
            self.initialize_camera()
            self.camera_active = True
            self.btn_start.config(text="停止检测")
            threading.Thread(target=self.update_frame, daemon=True).start()
        else:
            self.release_camera()

    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)  # 修改为实际视频源
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def update_frame(self):
        while self.camera_active:
            ret, frame = self.cap.read()
            if ret:
                # 调用 YOLOv8 模型进行姿态估计并绘制结果
                processed_frame = self.process_frame(frame)

                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

    def process_frame(self, frame):
        """使用 YOLOv8 模型进行姿态估计并绘制结果"""
        # 调用 YOLOv8 模型
        results = self.model(frame, imgsz=256)
        annotated_frame = results[0].plot()

        # 检查是否有检测结果
        if len(results[0].keypoints.xy) == 0:
            print("No detections in this frame.")
            return annotated_frame

        if self.selected_action == "深蹲":
            self.count_squats(results, annotated_frame)

        return annotated_frame

    def count_squats(self, results, frame):
        """检测深蹲动作并计数"""
        keypoints = results[0].keypoints.xy[0].cpu().numpy()  # 获取关键点坐标

        # 检查是否检测到关键点
        if len(keypoints) == 0:
            print("No keypoints detected in this frame.")
            return frame

        # 提取关键骨骼点
        try:
            left_hip = keypoints[11]  # 左髋关节
            left_knee = keypoints[13]  # 左膝
            left_ankle = keypoints[15]  # 左脚踝
            right_hip = keypoints[12]  # 右髋关节
            right_knee = keypoints[14]  # 右膝
            right_ankle = keypoints[16]  # 右脚踝
        except IndexError as e:
            print(f"Error accessing keypoints: {e}")
            print(f"Detected keypoints: {keypoints}")
            return frame  # 如果关键点不足，直接返回原帧

        # 计算髋关节、膝关节和踝关节之间的夹角
        left_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

        # 使用左右腿的平均角度
        avg_angle = (left_angle + right_angle) / 2

        # 判断动作
        if avg_angle < 95 and not self.is_start_position:
            self.squat_count += 1
            self.is_start_position = True  # 标记动作完成
        elif avg_angle > 120:
            self.is_start_position = False  # 重置状态

        # 更新计数显示
        self.count_label.config(text=f"动作计数: {self.squat_count}")

        # 在帧上绘制计数信息
        cv2.putText(frame, f"Squats: {self.squat_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def calculate_angle(self, a, b, c):
        """计算两个向量之间的角度"""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def release_camera(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.btn_start.config(text="开始检测")

    def upload_video(self):
        if not self.selected_action:
            messagebox.showwarning("警告", "请先选择一个运动项目！")
            return
        if self.camera_active:
            self.release_camera()
        self.video_path = filedialog.askopenfilename()
        if self.video_path:
            self.process_video()

    def process_video(self):
        self.video_active = True
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            self.video_active = False
            return

        while cap.isOpened() and self.video_active:
            success, frame = cap.read()
            if not success:
                print(f"Reached end of video or failed to read frame for {self.video_path}.")
                break

            # 调用 process_frame 方法处理每一帧
            processed_frame = self.process_frame(frame)

            # 显示处理后的帧
            cv2.imshow("YOLOv8 Pose Inference", processed_frame)

            # 按下 'q' 键退出视频播放
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 释放视频文件并关闭窗口
        cap.release()
        cv2.destroyAllWindows()
        self.video_active = False

    # 清零计数器的方法
    def reset_count(self):
        """清零计数器"""
        self.squat_count = 0
        self.count_label.config(text="动作计数: 0")
        print("计数器已清零")

# 主程序入口
if __name__ == "__main__":
    root = tk.Tk()
    app = SportsAnalysisApp(root)
    root.protocol("WM_DELETE_WINDOW", app.release_camera)  # 确保关闭窗口时释放摄像头
    root.mainloop()
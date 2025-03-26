import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from emulator import emulator  # 你的游戏环境

class AtariEDA:
    def __init__(self, rom_name='breakout.bin', max_frames=10000, output_dir="output", log_dir="log"):
        """
        初始化 EDA 分析类
        :param rom_name: 运行的游戏 ROM
        :param max_frames: 采集的最大帧数
        :param output_dir: 保存 EDA 图片的文件夹
        :param log_dir: 保存日志的文件夹
        """
        print("Initializing Atari Emulator for Full EDA...")
        self.engine = emulator(rom_name=rom_name, vis=False)
        self.max_frames = max_frames
        self.frames = []

        # 确保 output 和 log 目录存在
        self.output_dir = output_dir
        self.log_dir = log_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 生成带时间戳的日志文件
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file = os.path.join(self.log_dir, f"{timestamp}.log")

    def log(self, message):
        """
        记录日志到 log 文件，同时打印到终端
        """
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def preprocess_frame(self, frame):
        """
        对游戏帧进行预处理：
        1. 调整大小到 (84, 110)
        2. 转换为灰度图
        3. 裁剪 (去掉无关部分)
        4. 归一化到 [0,1]
        """
        frame_resized = cv2.resize(frame, (84, 110))
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_cropped = frame_gray[26:110, :]
        frame_normalized = frame_cropped / 255.0
        return frame_normalized

    def collect_samples(self):
        """
        采集完整游戏过程的帧，并进行预处理
        """
        self.log(f"Collecting up to {self.max_frames} frames...")
        state = self.engine.newGame()  
        frame_count = 0
        
        while frame_count < self.max_frames:
            processed_frame = self.preprocess_frame(state)
            self.frames.append(processed_frame)
            
            action = np.random.choice(self.engine.legal_actions)
            state, _, terminal = self.engine.next(action)
            
            frame_count += 1
            if terminal:  
                state = self.engine.newGame()
        
        self.log(f"Collected {frame_count} frames.")

    def plot_samples(self, num_samples=5):
        """
        显示采集到的样本图像，并保存
        """
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
        for i in range(num_samples):
            axes[i].imshow(self.frames[i], cmap='gray')
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis('off')
        output_path = os.path.join(self.output_dir, "sample_frames.png")
        plt.savefig(output_path, bbox_inches='tight')
        self.log(f"Sample frames saved to {output_path}")
        plt.close()

    def plot_histogram(self):
        """
        绘制灰度值分布直方图，并保存
        """
        all_pixels = np.concatenate([frame.flatten() for frame in self.frames])
        output_path = os.path.join(self.output_dir, "histogram.png")
        plt.hist(all_pixels, bins=50, color='blue', alpha=0.7)
        plt.title("Histogram of Grayscale Values (All Frames)")
        plt.xlabel("Gray Value")
        plt.ylabel("Frequency")
        plt.savefig(output_path, bbox_inches='tight')
        self.log(f"Histogram saved to {output_path}")
        plt.close()

    def plot_stats(self):
        """
        计算并绘制均值 & 标准差图，并保存
        """
        all_pixels = np.concatenate([frame.flatten() for frame in self.frames])
        mean_val = np.mean(all_pixels)
        std_val = np.std(all_pixels)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Mean", "Std Dev"], [mean_val, std_val], color=['green', 'red'])
        ax.set_title("Mean & Standard Deviation of Grayscale Values")
        output_path = os.path.join(self.output_dir, "stats.png")
        plt.savefig(output_path, bbox_inches='tight')
        self.log(f"Stats plot saved to {output_path}")
        plt.close()

    def show_stats(self):
        """
        计算并打印统计数据，同时保存到日志
        """
        all_pixels = np.concatenate([frame.flatten() for frame in self.frames])
        log_text = (
            f"Total Frames: {len(self.frames)}\n"
            f"Mean Pixel Value: {np.mean(all_pixels):.4f}\n"
            f"Std Dev: {np.std(all_pixels):.4f}\n"
            f"Min Pixel Value: {np.min(all_pixels)}\n"
            f"Max Pixel Value: {np.max(all_pixels)}\n"
        )
        self.log(log_text)

    def run_eda(self):
        """
        运行所有的 EDA 分析，并保存结果
        """
        self.log("Initializing Atari Emulator for Full EDA...")
        self.collect_samples()
        self.plot_samples(num_samples=10)  # 显示 & 保存 10 张样本
        self.plot_histogram()  # 保存直方图
        self.plot_stats()  # 保存均值 & 标准差图
        self.show_stats()  # 统计数据

if __name__ == "__main__":
    eda = AtariEDA(max_frames=5000)  # 采集 5000 帧数据
    eda.run_eda()

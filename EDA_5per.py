import numpy as np
import cv2
import matplotlib.pyplot as plt
from emulator import emulator  # 你的游戏环境

class AtariEDA:
    def __init__(self, rom_name='breakout.bin', sample_frames=5):
        """
        初始化 EDA 分析类
        :param rom_name: 运行的游戏 ROM
        :param sample_frames: 采样的帧数
        """
        print("Initializing Atari Emulator for EDA...")
        self.engine = emulator(rom_name=rom_name, vis=False)  # 关闭可视化，加速采样
        self.sample_frames = sample_frames
        self.frames = []
    
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
        frame_cropped = frame_gray[26:110, :]  # 裁剪掉顶部不相关区域
        frame_normalized = frame_cropped / 255.0  # 归一化
        return frame_normalized

    def collect_samples(self):
        """
        采集 `self.sample_frames` 张游戏帧，并进行预处理
        """
        print(f"Collecting {self.sample_frames} sample frames...")
        state = self.engine.newGame()  # 初始化游戏
        for _ in range(self.sample_frames):
            self.frames.append(self.preprocess_frame(state))  # 预处理后存入列表
            action = np.random.choice(self.engine.legal_actions)  # 随机选择动作
            state, _, _ = self.engine.next(action)  # 执行动作，获取下一帧
    
    def plot_samples(self):
        """
        显示采集到的样本图像
        """
        fig, axes = plt.subplots(1, self.sample_frames, figsize=(15, 5))
        for i in range(self.sample_frames):
            axes[i].imshow(self.frames[i], cmap='gray')
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis('off')
        plt.show()

    def plot_histogram(self):
        """
        绘制灰度值分布直方图
        """
        all_pixels = np.concatenate([frame.flatten() for frame in self.frames])
        plt.hist(all_pixels, bins=50, color='blue', alpha=0.7)
        plt.title("Histogram of Grayscale Values")
        plt.xlabel("Gray Value")
        plt.ylabel("Frequency")
        plt.show()

    def show_stats(self):
        """
        计算并输出帧的统计信息（均值、标准差、最大/最小值）
        """
        all_pixels = np.concatenate([frame.flatten() for frame in self.frames])
        print(f"Mean Pixel Value: {np.mean(all_pixels):.4f}")
        print(f"Std Dev: {np.std(all_pixels):.4f}")
        print(f"Min Pixel Value: {np.min(all_pixels)}")
        print(f"Max Pixel Value: {np.max(all_pixels)}")

    def run_eda(self):
        """
        运行所有的 EDA 分析
        """
        self.collect_samples()
        self.plot_samples()
        self.plot_histogram()
        self.show_stats()

if __name__ == "__main__":
    eda = AtariEDA(sample_frames=5) #统计5张
    eda.run_eda()

# Frequency-domain-deep-learning
## 频域深度学习入门教程

## 概述

这是一个用于在频域进行深度学习的入门教程，看完这个文档后，希望读者能够理解和掌握频域深度学习的核心概念、技术和应用。

## 📖 适用人群

- **初学者**: 对频域处理感兴趣的深度学习爱好者
- **研究人员**: 希望将频域方法应用到研究中的学者
- **工程师**: 需要在实际项目中使用频域技术的开发者
- **学生**: 学习信号处理和深度学习交叉领域的学生

## 🔧 主要技术

- **PyTorch FFT**: 高效的频域变换
- **傅里叶神经算子**: 现代频域深度学习
- **频谱卷积**: 全局感受野的卷积操作
- **频域可视化**: 直观的频域信号分析

## 📝 目录

1. [理论基础](#1-理论基础)
2. [环境设置和导入库](#2-环境设置和导入库)
3. [基础频域变换](#3-基础频域变换)
4. [频域信号分析与可视化](#4-频域信号分析与可视化)
5. [频域深度学习模型](#5-频域深度学习模型)
6. [时空域vs频域对比](#6-时空域vs频域对比)
7. [实际应用案例](#7-实际应用案例)
8. [高级技巧与优化](#8-高级技巧与优化)

---

## 1. 理论基础

### 什么是频域？

**时域（Time Domain）**: 信号随**时间**变化的表示方式。
**频域（Frequency Domain）**: 信号的**频率成分**表示方式。

### 傅里叶变换（Fourier Transform）

傅里叶变换是将时域信号转换为频域信号的数学工具：

$$
F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt
$$
离散傅里叶变换（DFT）：
$$
X_k = \sum_{n=0}^{N-1} x_n e^{-i2\pi kn/N}
$$


### 为什么要在频域做深度学习？

1. **计算效率提高**

   例如，对于时域/空间域（后续统称时域）上的卷积操作，利用傅里叶变换的性质，可以通过在频域中的逐点乘法来实现。相较于在时域上使用卷积核进行卷积操作，利用FFT卷积会提高计算性能，可以将复杂度从 $O(k^2N^2)$ 降低至 $O(N·logN)$，卷积核越大，优势越明显。

2. **具有全局感受野**

   时域网络，如 CNN / Transformer 的感受野较小，堆叠很多层后局部感受野才有可能达到全局感受野；而由于频域中的每个频率分量都包含整个时域信号的信息，因此，频域中的一次运算就可以达到影响时域信号全部范围的效果，使得 FNO 更适合捕获长程依赖和周期性的全局模式/周期性特征。FNO 在 PDE 求解中 4‒6 层即可达到比 30‒50 层 ResNet 更低的误差，且 **网格无关**（同一网络可推理 64×64 或 1024×1024）。

3. **多尺度信息解耦**

   许多时域信号通过FFT转换后，具有**稀疏性**，以及明确的物理意义，高频对应局部细节（细纹理），低频对应全局结构（全局形状）。因此，在频域中，可按频段**显式地剪枝、压缩或加权**，而无需像时域中那样，设计膨胀卷积、U-Net skip 之类复杂结构等，例如：剪枝操作可以去掉不重要而保留重要的频率成分，从而提升计算效率。当使用一些压缩算法后，还能达到参数量锐减的效果。此外，信息解耦还有其他的效果，如：

   - 一些扰动通常在时域表现为高频噪声，频域网络可在训练时直接 mask / 量化高频，天然对对抗扰动和 JPEG 量化更鲁棒（Fourier Adversarial Training）。

   - 频谱加权损失可轻松让网络先学低频形状、后补高频细节（curriculum）。

   - 正则化优化：添加L1正则迫使网络学习更紧凑的频域表示。

   - 动态计算：根据输入特征自适应分配计算资源到关键频段。

4. **物理一致性**

   对于很多科学计算（流体力学）问题，其边界本就是周期的；而频域天生满足周期假设或平移等变（平移等变性由卷积定理严格保证，不依赖近似），无需额外进行 padding 或 mask操作。

5. **与物理傅里叶算子无缝衔接**

   在科学机器学习（SciML）中，很多 PDE 的解算子本身就是频域乘法（如 Poisson 方程、Helmholtz）。用频域网络逼近这些算子时，**网络层与物理算子形式一致**，误差更小、可解释性更强。

## 2. 环境设置和导入库

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print(f"PyTorch版本: {torch.__version__}")
```

## 3. 基础频域变换

首先，让我们从基本的 FFT 操作开始，掌握如何在 PyTorch 中进行频域变换。

```python
class FrequencyTransforms:
    """频域变换工具类"""
    
    @staticmethod
    def fft1d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
        """一维快速傅里叶变换"""
        return torch.fft.fft(x, norm=norm)
    
    @staticmethod
    def ifft1d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
        """一维逆快速傅里叶变换"""
        return torch.fft.ifft(x, norm=norm)
    
    @staticmethod
    def fft2d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
        """二维快速傅里叶变换"""
        return torch.fft.fft2(x, norm=norm)
    
    @staticmethod
    def ifft2d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
        """二维逆快速傅里叶变换"""
        return torch.fft.ifft2(x, norm=norm)
    
    @staticmethod
    def rfft(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
        """实数快速傅里叶变换（更节省内存）"""
        return torch.fft.rfft(x, norm=norm)
    
    @staticmethod
    def irfft(x: torch.Tensor, n: int = None, norm: str = 'ortho') -> torch.Tensor:
        """实数逆快速傅里叶变换"""
        return torch.fft.irfft(x, n=n, norm=norm)

# 创建变换实例
ft 			  = FrequencyTransforms()

# 创建测试信号
t 		      = torch.linspace(0, 1, 128)
signal_test   = torch.sin(2 * np.pi * 5 * t) + 0.5 * torch.sin(2 * np.pi * 10 * t)

# 进行FFT变换
freq_signal   = ft.fft1d(signal_test)
reconstructed = ft.ifft1d(freq_signal).real 

print(f"原始信号形状: {signal_test.shape}")
print(f"频域信号形状: {freq_signal.shape}")
print(f"重构误差: {torch.mean((signal_test - reconstructed)**2):.8f}")
```

在对比中，若不将重构信号的实部提取出来（.real），PyTorch会自动将 实数张量 转换为 复数（虚部为0）后进行逐元素计算。 实际误差计算时通常取复数实部进行比较（需.real处理），否则会保留复数形式但虚部接近0（因数值精度误差）。

## 4. 频域信号分析与可视化

接下来，我们采用时域/频域可视化的方式来理解时域和频域之间的关系。首先，生成傅里叶变换后的频率坐标轴。

```python
freqs = np.fft.fftfreq(n, d=1/sample_rate) # sample_rate=1
```

该频率坐标轴是绘制频域图（幅度谱/相位谱）的横坐标基础，对应FFT变换后的各个频率分量。具体来说：

- n 是信号长度
- d 表示采样间隔
- 返回的数组包含从 -0.5 到 0.5 Hz 的归一化频率坐标
- 后续绘图时通过freqs[:n//2]取正频率部分用于可视化

当采样间隔d=1时，采样频率fs=1/d=1Hz。根据奈奎斯特采样定理，最大可表示频率为fs/2=0.5Hz。np.fft.fftfreq返回的归一化频率坐标范围是[-0.5, 0.5)，其中：
正频率部分[0, 0.5)对应实际物理频率；负频率部分[-0.5, 0)是FFT对称性的数学表达。当采样率不是1时（如d=0.5），实际频率范围会按fs=1/d比例缩放。

```python
plt.rcParams['font.family']        = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
class FrequencyVisualizer:
    """频域可视化工具"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
    
    def plot_time_frequency_analysis(self, signal: torch.Tensor, sample_rate: float = 1.0, title: str = "时频分析"):
        """绘制时域、频域和相位谱"""
        # 转换为numpy用于绘图
        if isinstance(signal, torch.Tensor):
            signal_np = signal.detach().cpu().numpy()
        else:
            signal_np = signal
            signal = torch.from_numpy(signal)
        
        # 计算频域表示
        freq_signal = torch.fft.fft(signal)
        freq_signal_np = freq_signal.detach().cpu().numpy()
        
        # 计算频率轴
        n = len(signal_np)
        freqs = np.fft.fftfreq(n, d=1/sample_rate) # 将采样率(Hz)转换为采样间隔(秒/样本)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 时域信号
        time_axis = np.linspace(0, len(signal_np)/sample_rate, len(signal_np))
        axes[0, 0].plot(time_axis, signal_np, 'b-', linewidth=2)
        axes[0, 0].set_title('时域信号')
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('幅度')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 频域幅度谱
        magnitude = np.abs(freq_signal_np)
        axes[0, 1].plot(freqs[:n//2], magnitude[:n//2], 'r-', linewidth=2)
        axes[0, 1].set_title('频域幅度谱')
        axes[0, 1].set_xlabel('频率 (Hz)')
        axes[0, 1].set_ylabel('幅度')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 相位谱
        phase = np.angle(freq_signal_np)
        axes[1, 0].plot(freqs[:n//2], phase[:n//2], 'g-', linewidth=2)
        axes[1, 0].set_title('相位谱')
        axes[1, 0].set_xlabel('频率 (Hz)')
        axes[1, 0].set_ylabel('相位 (弧度)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 功率谱密度
        power = magnitude**2
        axes[1, 1].semilogy(freqs[:n//2], power[:n//2], 'm-', linewidth=2)
        axes[1, 1].set_title('功率谱密度 (对数尺度)')
        axes[1, 1].set_xlabel('频率 (Hz)')
        axes[1, 1].set_ylabel('功率')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return freqs, magnitude, phase

# 创建可视化器
viz = FrequencyVisualizer()

# 示例1：复合正弦信号
print("=== 示例1: 复合正弦信号 ===")
t = torch.linspace(0, 2, 512)
signal1 = (torch.sin(2 * np.pi * 3 * t) + 
          0.7 * torch.sin(2 * np.pi * 7 * t) + 
          0.3 * torch.sin(2 * np.pi * 15 * t) +
          0.1 * torch.randn_like(t))  # 添加噪声

freqs, mag, phase = viz.plot_time_frequency_analysis(signal1, sample_rate=256, title="复合正弦信号频域分析")
```

![](D:\24bo\扩散模型\torch-cfd-main\show1.svg)

**子图分析**

左上绘制了原始信号，横坐标是时间，纵坐标是幅值。曲线呈不规则的正弦-like 振荡，幅度在 -2 到 2 之间波动，有多个峰谷，表明多频率叠加。宏观感受就是，具有主周期（较慢波动）和高频抖动（噪声）。除此之外，通过观察，很难获取原始信号的其他信息。这时候，傅里叶变换就派上用场了，可以用它来揭示隐藏的频率模式。右上绘制了频域信号的幅度谱，横轴为频率 (Hz)，纵轴为幅值。图中有一个高尖峰在低频（近0 Hz），然后迅速衰减到零，类似于指数衰减，且放眼望去，并没有明显的多个峰值，这表明信号主要是低频主导，或高频被噪声淹没。从图中可以看到，主峰在低频段（ 0-20 Hz），表示信号能量集中在此，多个峰（如3Hz、7Hz、15Hz），它们对应原始信号的正弦成分。此外，幅度谱量化了每个频率的“强度”。在后续深度学习中，可以用于特征提取，例如FNO模型通过截断高频模式（modes）来高效计算。

接下来，在类中添加 **2D** 可视化方法：

```python
# 添加2D分析方法
def plot_2d_frequency_analysis(self, image: torch.Tensor, title: str = "2D频域分析"):
    """绘制2D图像的频域分析"""
    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image
        image = torch.from_numpy(image)
    
    # 2D FFT
    freq_image = torch.fft.fft2(image)
    freq_image_shifted = torch.fft.fftshift(freq_image)
    
    # 转换为numpy
    freq_np = freq_image_shifted.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=self.figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 原图像
    im1 = axes[0, 0].imshow(image_np, cmap='viridis')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # 频域幅度谱
    magnitude_2d = np.abs(freq_np)
    im2 = axes[0, 1].imshow(np.log(magnitude_2d + 1), cmap='hot')
    axes[0, 1].set_title('频域幅度谱 (对数)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # 相位谱
    phase_2d = np.angle(freq_np)
    im3 = axes[1, 0].imshow(phase_2d, cmap='hsv')
    axes[1, 0].set_title('相位谱')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # 重构图像
    reconstructed = torch.fft.ifft2(freq_image).real
    reconstructed_np = reconstructed.detach().cpu().numpy()
    im4 = axes[1, 1].imshow(reconstructed_np, cmap='viridis')
    axes[1, 1].set_title('重构图像')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    plt.show()
    
    return freq_image, magnitude_2d, phase_2d

# 将方法添加到类中
FrequencyVisualizer.plot_2d_frequency_analysis = plot_2d_frequency_analysis

# 示例2：2D图像频域分析
print("\n=== 示例2: 2D图像频域分析 ===")

# 创建一个带有周期性模式的测试图像
x = torch.linspace(-5, 5, 128)
y = torch.linspace(-5, 5, 128)
X, Y = torch.meshgrid(x, y, indexing='ij')

# 创建具有多种频率成分的图像
test_image = (torch.sin(2 * X) * torch.cos(3 * Y) + 
              0.5 * torch.sin(5 * X + 2 * Y) +
              0.3 * torch.exp(-(X**2 + Y**2)/4))  # 高斯包络

freq_img, mag_2d, phase_2d = viz.plot_2d_frequency_analysis(test_image, title="2D图像频域分析")

```

![](D:\24bo\扩散模型\torch-cfd-main\show2.svg)

在分析2D 频谱图时，最重要的是理解它代表了原始图像的频率成分分解。由于零频已移到中心，所以中心点代表了低频（全局/平滑特征），周围是高频（细节/噪声）。分析时，应先从图像中心出发，因为中心点（或小区域）对应零频率（DC term），表示原始图像的平均亮度或整体偏移。中心可以看作频谱的“锚点”，携带了最多能量（通常是最亮点）。如果中心过亮，可能表示图像有强低频偏置（如均匀背景）；如果暗淡，则图像对比度高或有高频主导。接着，从中心向外扩展，检查径向分布，半径直接代表空间频率（spatial frequency）的幅度。低半径（20%）代表了低频区域，对应大尺度结构（如整体形状、渐变）。看是否有十字或斑点（表示方向性模式，如水平/垂直边缘）；中等半径代表了中频区域（50%）：对应纹理和中等细节；大半径代表了高频区域（100%）：对应边缘、噪声或细微纹理。通常能量低、随机分布。如果有亮斑，可能表示周期性噪声。

此外，可以对二维图像进行x和y方向的分解可视化：

```python
# 在类中添加切片方法
def plot_1d_slice_analysis(self, freq_image_shifted: torch.Tensor, 
                            slice_type: str = 'horizontal', title: str = "1D切片频域分析"):
    """从2D频谱切片到1D分析"""
    freq_np = freq_image_shifted.detach().cpu().numpy()
    magnitude_2d = np.abs(freq_np)
    center_y, center_x = magnitude_2d.shape[0]//2, magnitude_2d.shape[1]//2
    
    if slice_type == 'horizontal':  # 水平切片（分析x维度）
        magnitude_1d = magnitude_2d[center_y, :]  # 中心行
        freq_axis = np.fft.fftshift(np.fft.fftfreq(magnitude_2d.shape[1]))
        label = '水平频率'
    elif slice_type == 'vertical':  # 垂直切片（分析y维度）
        magnitude_1d = magnitude_2d[:, center_x]  # 中心列
        freq_axis = np.fft.fftshift(np.fft.fftfreq(magnitude_2d.shape[0]))
        label = '垂直频率'
    
    plt.figure(figsize=(10, 5))
    plt.plot(freq_axis, np.log(magnitude_1d + 1), 'g-', linewidth=2)
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel('幅度 (对数)')
    plt.grid(True)
    plt.savefig('show3.svg')
    plt.show()
    
    return freq_axis, magnitude_1d

# 添加方法并使用
FrequencyVisualizer.plot_1d_slice_analysis = plot_1d_slice_analysis
# 示例：垂直切片（分析y维度）
freq_axis_y, mag_1d_y = viz.plot_1d_slice_analysis(torch.tensor(freq_img), slice_type='vertical', title="y维度1D切片幅度谱")
```

![](D:\24bo\扩散模型\torch-cfd-main\show3.svg)

## 5. 频域深度学习模型

现在，让我们实现一个频域卷积模型。

```python
class SpectralConv1d(nn.Module):
    """一维频谱卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes        = modes  # 保留的频率模式数量
        
        # 复数权重
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat) * 0.02
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, length)
        batch_size = x.shape[0]
        
        # 转换到频域
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # 初始化输出
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        
        # 频谱卷积
        out_ft[:, :, :self.modes] = torch.einsum('bix,iox->box', x_ft[:, :, :self.modes], self.weights)
        
        # 转换回时域
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x


class FourierNeuralOperator1D(nn.Module):
    """一维傅里叶神经算子"""
    
    def __init__(self, modes: int, width: int, input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # 输入投影
        self.fc0   = nn.Linear(input_dim + 1, width)  # +1 for coordinate
        
        # 频谱卷积层
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        
        # 局部卷积
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        
        # 输出投影
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, length, input_dim)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        # 输入投影
        x = self.fc0(x)  # (batch, length, width)
        x = x.permute(0, 2, 1)  # (batch, width, length)
        
        # 频谱卷积层
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x  = x1 + x2
        x  = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x  = x1 + x2
        x  = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x  = x1 + x2
        x  = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x  = x1 + x2
        
        # 输出投影
        x = x.permute(0, 2, 1)  # (batch, length, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x
    
    def get_grid(self, shape, device):
        """生成坐标网格"""
        batch_size, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1).repeat([batch_size, 1, 1])
        return gridx.to(device)


# 测试模型
print("=== 测试频域深度学习模型 ===")

# 创建测试数据
batch_size, seq_len = 32, 256
test_input = torch.randn(batch_size, seq_len, 1)

# 测试FNO
fno        = FourierNeuralOperator1D(modes=16, width=64)
fno_output = fno(test_input)
print(f"FNO输出形状: {fno_output.shape}")

# 计算参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"FNO参数数量: {count_parameters(fno):,}")
```

## 6. 时空域vs频域对比

让我们通过实际例子来比较时域和频域方法的差异。

```python
class TimeDomainCNN(nn.Module):
    """时域卷积神经网络"""
    
    def __init__(self, input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, output_dim, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, length, channels)
        x = x.permute(0, 2, 1)  # (batch, channels, length)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # (batch, length, channels)
        return x


def generate_pde_data(n_samples: int = 1000, seq_len: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成简单PDE问题的数据：热方程"""
    # 空间坐标
    x = torch.linspace(0, 2*np.pi, seq_len)
    
    inputs = []
    outputs = []
    
    for _ in range(n_samples):
        # 随机初始条件（多个正弦波的组合）
        coeffs = torch.randn(5) * 0.5
        freqs = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        
        u0 = torch.sum(coeffs.unsqueeze(1) * torch.sin(freqs.unsqueeze(1) * x.unsqueeze(0)), dim=0)
        
        # 解析解（热方程简化版）
        t = 0.1  # 时间步长
        alpha = 0.1  # 扩散系数
        u_t = torch.sum(coeffs.unsqueeze(1) * torch.exp(-alpha * freqs.unsqueeze(1)**2 * t) * 
                       torch.sin(freqs.unsqueeze(1) * x.unsqueeze(0)), dim=0)
        
        inputs.append(u0.unsqueeze(-1))
        outputs.append(u_t.unsqueeze(-1))
    
    return torch.stack(inputs), torch.stack(outputs)


def train_and_compare_models():
    """训练并比较时域和频域模型"""
    # 生成数据
    print("生成训练数据...")
    X_train, y_train = generate_pde_data(1000, 128)
    X_test, y_test = generate_pde_data(200, 128)
    
    # 创建模型
    time_model = TimeDomainCNN()
    freq_model = FourierNeuralOperator1D(modes=16, width=32)
    
    models = {
        'Time Domain CNN': time_model,
        'Frequency Domain FNO': freq_model
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 简单训练循环
        train_losses = []
        model.train()
        
        for epoch in range(50):
            optimizer.zero_grad()
            
            # 批量训练
            batch_size = 32
            indices = torch.randperm(len(X_train))[:batch_size]
            x_batch = X_train[indices]
            y_batch = y_train[indices]
            
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test).item()
        
        results[name] = {
            'train_losses': train_losses,
            'test_loss': test_loss,
            'parameters': count_parameters(model),
            'predictions': test_pred[:5]  # 保存前5个预测用于可视化
        }
    
    return results, X_test[:5], y_test[:5]


# 执行比较
print("=== 时域vs频域模型对比 ===")
results, test_inputs, test_targets = train_and_compare_models()

# 打印结果
print("\n=== 训练结果对比 ===")
for name, result in results.items():
    print(f"{name}:")
    print(f"  参数数量: {result['parameters']:,}")
    print(f"  测试损失: {result['test_loss']:.6f}")
    print(f"  最终训练损失: {result['train_losses'][-1]:.6f}")

```

## 7. 实际应用案例

频域深度学习在实际问题中的应用：信号降噪。

```python
class SignalDenoiser(nn.Module):
    """频域信号降噪器"""
    
    def __init__(self, modes: int = 32):
        super().__init__()
        self.modes = modes
        
        # 频域特征提取
        self.freq_encoder = nn.Sequential(
            nn.Linear(modes * 2, 128),  # 实部+虚部
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, modes * 2)
        )
        
        # 注意力机制用于频率选择
        self.attention = nn.Sequential(
            nn.Linear(modes * 2, modes),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, length)
        
        # 转换到频域
        x_freq = torch.fft.fft(x)
        
        # 只保留低频部分进行处理
        x_freq_low = x_freq[:, :self.modes]
        
        # 分离实部和虚部
        x_complex = torch.cat([x_freq_low.real, x_freq_low.imag], dim=-1)
        
        # 频域特征提取
        freq_features = self.freq_encoder(x_complex)
        
        # 计算注意力权重
        attention_weights = self.attention(freq_features)
        
        # 重构复数频域信号
        real_part = freq_features[:, :self.modes]
        imag_part = freq_features[:, self.modes:]
        
        # 应用注意力权重
        real_part = real_part * attention_weights
        imag_part = imag_part * attention_weights
        
        # 重构频域信号
        x_freq_processed = torch.complex(real_part, imag_part)
        
        # 零填充到原始长度
        x_freq_full = torch.zeros_like(x_freq)
        x_freq_full[:, :self.modes] = x_freq_processed
        
        # 转换回时域
        x_denoised = torch.fft.ifft(x_freq_full).real
        
        return x_denoised


def create_noisy_signals(n_samples: int = 1000, seq_len: int = 256):
    """创建带噪声的测试信号"""
    t = torch.linspace(0, 4*np.pi, seq_len)
    
    clean_signals = []
    noisy_signals = []
    
    for _ in range(n_samples):
        # 创建清洁信号（多个正弦波组合）
        freq1 = np.random.uniform(0.5, 2.0)
        freq2 = np.random.uniform(2.0, 4.0)
        freq3 = np.random.uniform(0.1, 0.5)
        
        amp1 = np.random.uniform(0.5, 1.0)
        amp2 = np.random.uniform(0.3, 0.7)
        amp3 = np.random.uniform(0.2, 0.4)
        
        clean = (amp1 * torch.sin(freq1 * t) + 
                amp2 * torch.sin(freq2 * t) + 
                amp3 * torch.sin(freq3 * t))
        
        # 添加噪声
        noise_level = np.random.uniform(0.1, 0.3)
        noise = noise_level * torch.randn_like(clean)
        noisy = clean + noise
        
        clean_signals.append(clean)
        noisy_signals.append(noisy)
    
    return torch.stack(clean_signals), torch.stack(noisy_signals)


# 训练降噪器
print("=== 频域信号降噪应用 ===")
print("生成训练数据...")
clean_train, noisy_train = create_noisy_signals(1000, 256)
clean_test, noisy_test = create_noisy_signals(100, 256)

# 创建模型
denoiser = SignalDenoiser(modes=32)
optimizer = torch.optim.Adam(denoiser.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("训练降噪器...")
denoiser.train()
train_losses = []

for epoch in range(100):
    # 批量训练
    batch_size = 32
    indices = torch.randperm(len(noisy_train))[:batch_size]
    noisy_batch = noisy_train[indices]
    clean_batch = clean_train[indices]
    
    optimizer.zero_grad()
    denoised = denoiser(noisy_batch)
    loss = criterion(denoised, clean_batch)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# 测试降噪效果
denoiser.eval()
with torch.no_grad():
    test_denoised = denoiser(noisy_test)
    test_loss = criterion(test_denoised, clean_test)
    
    # 计算信噪比改善
    def snr_db(signal, noise):
        signal_power = torch.mean(signal**2)
        noise_power = torch.mean(noise**2)
        return 10 * torch.log10(signal_power / noise_power)
    
    original_snr = snr_db(clean_test, noisy_test - clean_test)
    denoised_snr = snr_db(clean_test, test_denoised - clean_test)
    
    print(f"\n测试结果:")
    print(f"测试损失: {test_loss:.6f}")
    print(f"原始信噪比: {original_snr:.2f} dB")
    print(f"降噪后信噪比: {denoised_snr:.2f} dB")
    print(f"信噪比改善: {denoised_snr - original_snr:.2f} dB")

```

## 8. 高级技巧与优化

在这一节中，我们将介绍一些高级的频域深度学习技巧。

这里实现了一个通过可学习的阈值和权重参数，实现动态的、数据驱动的频率双重调控机制，相比固定滤波器能更好地适应不同任务需求。这样的好处有两个，首先可以引入结构先验， 阈值参数提供了频率分界点的初始假设（如0.1对应低频），通过sigmoid的陡峭系数（10）形成明确的保留/抑制区划分。这为模型提供了物理意义明确的频率选择先验，比纯权重学习收敛更快；其次，提供了解耦不同维度的控制，阈值控制频率选择的宏观范围（选择哪些频段），权重控制微观调整（选中频段内的相对重要性）。这种解耦使网络可以分别优化频率选择的整体策略和局部调整。这种设计类似于CNN中同时使用卷积核（局部特征提取）和注意力机制（特征重要性调整）的互补思路。

```python
# 高级技巧1：自适应频率滤波
class AdaptiveFrequencyFilter(nn.Module):
    """自适应频率滤波器"""
    
    def __init__(self, seq_len: int, learnable_threshold: bool = True):
        super().__init__()
        self.seq_len = seq_len
        
        if learnable_threshold:
            # 可学习的频率阈值
            self.freq_threshold = nn.Parameter(torch.tensor(0.1))
            # 可学习的频率权重
            self.freq_weights = nn.Parameter(torch.ones(seq_len // 2 + 1))
        else:
            self.register_buffer('freq_threshold', torch.tensor(0.1))
            self.register_buffer('freq_weights', torch.ones(seq_len // 2 + 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 转换到频域
        x_freq = torch.fft.rfft(x, dim=-1)
        
        # 计算频率掩码
        freqs = torch.fft.rfftfreq(self.seq_len, device=x.device)
        mask = torch.sigmoid((self.freq_threshold - freqs) * 10)
        
        # 应用自适应权重
        weighted_mask = mask * torch.sigmoid(self.freq_weights)
        
        # 滤波
        x_filtered = x_freq * weighted_mask.unsqueeze(0)
        
        # 转换回时域
        x_out = torch.fft.irfft(x_filtered, n=self.seq_len, dim=-1)
        
        return x_out
```

接下来将介绍频域损失函数。该损失函数具有特征解耦的特点，采用时域损失和频域损失双管齐下的特点，时域损失主要用于捕捉整体波形差异，频域损失用于独立优化幅度（能量分布）和相位（时序关系）这两个正交特征。在物理意义上，也能解释它的优点，许多实际应用中，人类感知对频域特征更敏感（如特定频率成分的保留），把信号高频部分整体衰减 10 %，时域上的逐点差异可能很小，人耳/人眼却会立刻注意到“闷”或“糊”，频域损失会直接惩罚这种频带能量错误，而 时域对此不敏感。；此外，当一些任务涉及频域特性时（如去噪需保持干净频率成分），或者频域特征对任务至关重要时，纯时域损失可能无法有效传递梯度到频域相关参数，采用这种混合损失能显著提升模型表现。

还可以从**相位偏差**、**能量分布偏差**和**跨尺度误差**入手。一个纯正弦向前平移几个采样点，时域 MSE 立即变大，但在频域里仅仅表现为 phase 变化、幅度不变。仅用**时域损失**会把“轻微时移”当成巨大误差，模型会花力气去拟合这种其实无关紧要的位移；而**频域损失**能告诉网络“形状对了，只是时间没对齐”，从而避免过拟合局部时移。时域 MSE 把每个采样点权重等同，10 kHz 误差和 100 Hz 误差被平等对待；频域损失天然按频段归一化，更容易让网络在不同尺度上均衡学习。

```python
# 高级技巧2：频域损失函数
class FrequencyDomainLoss(nn.Module):
    """频域损失函数"""
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 时域损失
        time_loss = self.mse(pred, target)
        
        # 频域损失
        pred_freq = torch.fft.fft(pred)
        target_freq = torch.fft.fft(target)
        
        # 幅度损失
        magnitude_loss = self.mse(torch.abs(pred_freq), torch.abs(target_freq))
        
        # 相位损失
        phase_loss = self.mse(torch.angle(pred_freq), torch.angle(target_freq))
        
        freq_loss = magnitude_loss + phase_loss
        
        # 组合损失
        total_loss = (1 - self.alpha) * time_loss + self.alpha * freq_loss
        
        return total_loss


# 测试高级模块
print("=== 测试高级频域模块 ===")

# 创建测试数据
test_seq_len = 256
test_batch = 16
test_data = torch.randn(test_batch, test_seq_len)

# 测试自适应滤波器
adaptive_filter = AdaptiveFrequencyFilter(test_seq_len, learnable_threshold=True)
filtered_output = adaptive_filter(test_data)
print(f"自适应滤波器输出形状: {filtered_output.shape}")
print(f"可学习阈值: {adaptive_filter.freq_threshold.item():.4f}")

# 测试频域损失
freq_loss = FrequencyDomainLoss(alpha=0.3)
dummy_pred = torch.randn(10, 128)
dummy_target = torch.randn(10, 128)
loss_value = freq_loss(dummy_pred, dummy_target)
print(f"频域损失值: {loss_value.item():.6f}")
```

### 🌟 进阶学习

1. **更复杂的PDE问题**：Navier-Stokes方程、Maxwell方程、高维频域方法等
2. **图像和视频处理**：2D/3D频域方法、图神经网络与频域
3. **多物理场耦合**：复杂系统建模
4. **实时应用**：边缘计算和推理优化
5. **理论分析**：频域网络的泛化能力和表达能力

## 📝 贡献和反馈

如果您有任何建议或发现问题，欢迎：

- 提出改进建议
- 报告错误或问题
- 贡献新的应用案例
- 分享学习心得

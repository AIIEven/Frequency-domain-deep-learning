# 时域与频域操作的对应关系

## 理论基础

根据傅里叶变换的性质，时域和频域操作有以下对应关系：

### 1. 卷积定理 (Convolution Theorem)
**时域卷积 ↔ 频域乘法**

- 时域 
  $$
  (f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t-\tau) d\tau
  $$
  
- 频域
  $$
  \mathcal{F}(f * g) = \mathcal{F}(f) \cdot \mathcal{F}(g)
  $$
  

### 2. 微分性质 (Differentiation Property)
**时域微分 ↔ 频域乘以 $j\omega$**
$$
\frac{df(t)}{dt} \buildrel {} \over \longleftrightarrow  j\omega F(\omega)
$$
其中 $F(\omega) = \mathcal{F}(f(t))$

### 3. 积分性质 (Integration Property)
**时域积分 ↔ 频域除以 $j\omega$**
$$
\int_{-\infty}^{t} f(\tau) d\tau \buildrel {} \over \longleftrightarrow  \frac{F(\omega)}{j\omega} + \pi F(0)\delta(\omega)
$$


### 4. 时移性质 (Time Shift)
**时域时移 ↔ 频域相位调制**
$$
f(t-t_0) \buildrel {} \over \longleftrightarrow F(\omega)e^{-j\omega t_0}
$$
下面我们用 `PyTorch` 来验证这些性质。

## 实验验证

验证卷积定理: 时域卷积 = 频域乘法的逆变换

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 设置随机种子确保结果可重现
torch.manual_seed(42)

# 创建时域信号
fs = 1000  # 采样频率
t  = torch.linspace(0, 1, fs, dtype=torch.float32)  # 时间轴
dt = t[1] - t[0]  # 时间间隔

# 创建两个测试信号
# 正弦波公式：sin(2πft)，其中f即频率值
f1 = torch.sin(2 * np.pi * 5 * t)  # 5Hz正弦波
f2 = torch.exp(-t) * torch.sin(2 * np.pi * 10 * t)  # 衰减10Hz正弦波

print(f"信号长度: {len(f1)}")
print(f"采样间隔: {dt:.4f} s")
print(f"频率分辨率: {1/(len(f1)*dt):.4f} Hz")


print("=" * 50)
print("1. 验证卷积定理")
print("=" * 50)

# 时域卷积 (使用PyTorch的conv1d，需要调整维度)
f1_padded = F.pad(f1, (len(f2)-1, 0))  # 零填充避免边界效应
f2_flipped = torch.flip(f2, dims=[0])  # 翻转f2

# 手动实现卷积
conv_time = F.conv1d(f1_padded.unsqueeze(0).unsqueeze(0), 
                    f2_flipped.unsqueeze(0).unsqueeze(0), 
                    padding=0).squeeze()

# 截取与原信号相同长度
conv_time = conv_time[:len(f1)]

# 频域乘法
F1 = torch.fft.fft(f1)
F2 = torch.fft.fft(f2)
conv_freq = torch.fft.ifft(F1 * F2).real

# 比较结果
error_conv = torch.mean((conv_time - conv_freq)**2)
print(f"卷积定理验证:")
print(f"  时域卷积与频域乘法的均方误差: {error_conv:.8f}")
print(f"  最大绝对误差: {torch.max(torch.abs(conv_time - conv_freq)):.8f}")

# 可视化前100个点
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(conv_time[:100].numpy(), 'b-', label='时域卷积', linewidth=2)
plt.plot(conv_freq[:100].numpy(), 'r--', label='频域乘法(IFFT)', linewidth=2)
plt.legend()
plt.title('卷积定理验证')
plt.xlabel('样本点')
plt.ylabel('幅值')

plt.subplot(1, 2, 2)
plt.plot((conv_time - conv_freq)[:100].numpy(), 'g-', linewidth=2)
plt.title('误差 (时域 - 频域)')
plt.xlabel('样本点')
plt.ylabel('误差')
plt.tight_layout()
plt.show()
```

![](D:\24bo\扩散模型\AFNO-transformer-master\figures\conv_time_freq.png)

从上图中可以看出，误差很大，这是因为对概念的错误理解造成的，此外本文档中的案例都存在边界处理问题，个人感觉除了卷积可以通过填充处理外，其他的都很难完美的解决。

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)

# 创建时域信号
fs = 1000  # 采样频率
t  = torch.linspace(0, 1, fs, dtype=torch.float32)  # 时间轴
dt = t[1] - t[0]  # 时间间隔

# 创建两个时域信号
f1 = torch.sin(2 * np.pi * 5 * t)  # 5Hz正弦波
f2 = torch.exp(-t) * torch.sin(2 * np.pi * 10 * t)  # 衰减10Hz正弦波

print(f"信号长度: {len(f1)}")
print(f"采样间隔: {dt:.4f} s")
print(f"频率分辨率: {1/(len(f1)*dt):.4f} Hz")

print("=" * 50)
print("1. 验证卷积定理")
print("=" * 50)

# 时域卷积 (使用PyTorch的conv1d，需要调整维度)
# 为线性卷积添加零填充
pad_length = len(f2) - 1

# f1_padded  = F.pad(f1, (pad_length, pad_length))
f1_padded  = F.pad(f1, (pad_length, pad_length))  # 零填充避免边界效应
f2_flipped = torch.flip(f2, dims=[0])  # Conv1d中的卷积实际上是相关,所以需要翻转f2

# 手动实现卷积 (full mode)
conv_time  = F.conv1d(
    f1_padded.unsqueeze(0).unsqueeze(0), 
    f2_flipped.unsqueeze(0).unsqueeze(0), 
    padding=0
).squeeze()


# conv_time  = conv_time[:len(f1)]  # 截取与原信号相同长度

# 频域乘法

# TODO
N  = len(f1)
M  = len(f2)
padded_length = N + M - 1
f1_padded  = F.pad(f1, (0, padded_length - N)) # ADDED
f2_padded  = F.pad(f2, (0, padded_length - M)) # ADDED

F1 = torch.fft.fft(f1_padded)
F2 = torch.fft.fft(f2_padded)
conv_freq = torch.fft.ifft(F1 * F2).real

# 比较结果
error_conv = torch.mean((conv_time - conv_freq)**2)
print(f"卷积定理验证:")
print(f"  时域卷积与频域乘法的均方误差: {error_conv:.8f}")
print(f"  最大绝对误差: {torch.max(torch.abs(conv_time - conv_freq)):.8f}")

# 可视化前100个点
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(conv_time[:100].numpy(), 'b-', label='时域卷积', linewidth=2)
plt.plot(conv_freq[:100].numpy(), 'r--', label='频域乘法(IFFT)', linewidth=2)
plt.legend()
plt.title('卷积定理验证')
plt.xlabel('样本点')
plt.ylabel('幅值')

plt.subplot(1, 2, 2)
plt.plot((conv_time - conv_freq)[:100].numpy(), 'g-', linewidth=2)
plt.title('误差 (时域 - 频域)')
plt.xlabel('样本点')
plt.ylabel('误差')
plt.tight_layout()
plt.show()
```

![](D:\24bo\扩散模型\AFNO-transformer-master\figures\conv.png)

验证微分性质: 时域微分 = 频域乘以 jω 的逆变换

```python
print("\n" + "=" * 50)
print("2. 验证微分性质")
print("=" * 50)

# 时域数值微分 (前向差分)
df_time = torch.zeros_like(f1)
df_time[:-1] = (f1[1:] - f1[:-1]) / dt
df_time[-1] = df_time[-2]  # 边界处理

# 频域微分: F(ω) * jω
# 构造频率轴
freqs = torch.fft.fftfreq(len(f1), dt)
omega = 2 * np.pi * freqs
jw = 1j * omega

# 频域微分
F1 = torch.fft.fft(f1)
df_freq = torch.fft.ifft(F1 * jw).real

# 比较结果 (忽略边界效应，只比较中间部分)
start, end = 50, -50
error_diff = torch.mean((df_time[start:end] - df_freq[start:end])**2)
print(f"微分性质验证:")
print(f"  时域微分与频域微分的均方误差: {error_diff:.8f}")
print(f"  最大绝对误差: {torch.max(torch.abs(df_time[start:end] - df_freq[start:end])):.8f}")

# 可视化
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t[start:end], df_time[start:end], 'b-', label='时域数值微分', linewidth=2)
plt.plot(t[start:end], df_freq[start:end], 'r--', label='频域微分(IFFT)', linewidth=2)
plt.legend()
plt.title('微分性质验证')
plt.xlabel('时间 (s)')
plt.ylabel('微分值')

plt.subplot(1, 2, 2)
plt.plot(t[start:end], (df_time - df_freq)[start:end], 'g-', linewidth=2)
plt.title('误差 (时域 - 频域)')
plt.xlabel('时间 (s)')
plt.ylabel('误差')
plt.tight_layout()
plt.show()
```

![](D:\24bo\扩散模型\AFNO-transformer-master\figures\diff_time_freq.png)

验证积分性质: 时域积分 = 频域除以 jω 的逆变换

```python
print("\n" + "=" * 50)
print("3. 验证积分性质")
print("=" * 50)

# 时域数值积分 (累积求和)
integral_time = torch.cumsum(f1 * dt, dim=0)

# 频域积分: F(ω) / jω
# 注意: ω=0时需要特殊处理 (直流分量)
F1 = torch.fft.fft(f1)
jw_inv = torch.zeros_like(jw, dtype=torch.complex64)

# 避免除零，对于ω≠0的分量
nonzero_mask = omega != 0
jw_inv[nonzero_mask] = 1.0 / (1j * omega[nonzero_mask])

# 对于ω=0的分量，设为0 (假设原信号无直流偏移)
jw_inv[~nonzero_mask] = 0

# 频域积分
integral_freq = torch.fft.ifft(F1 * jw_inv).real

# 由于积分有任意常数，我们对齐两个结果的起始点
integral_freq = integral_freq - integral_freq[0] + integral_time[0]

# 比较结果
error_int = torch.mean((integral_time - integral_freq)**2)
print(f"积分性质验证:")
print(f"  时域积分与频域积分的均方误差: {error_int:.8f}")
print(f"  最大绝对误差: {torch.max(torch.abs(integral_time - integral_freq)):.8f}")

# 可视化
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t[:200], integral_time[:200], 'b-', label='时域数值积分', linewidth=2)
plt.plot(t[:200], integral_freq[:200], 'r--', label='频域积分(IFFT)', linewidth=2)
plt.legend()
plt.title('积分性质验证')
plt.xlabel('时间 (s)')
plt.ylabel('积分值')

plt.subplot(1, 2, 2)
plt.plot(t[:200], (integral_time - integral_freq)[:200], 'g-', linewidth=2)
plt.title('误差 (时域 - 频域)')
plt.xlabel('时间 (s)')
plt.ylabel('误差')
plt.tight_layout()
plt.show()
```

![](D:\24bo\扩散模型\AFNO-transformer-master\figures\integral.png)

验证时移性质: 时域时移 = 频域相位调制的逆变换

```python
print("\n" + "=" * 50)
print("4. 验证时移性质")
print("=" * 50)

# 时移量
shift_samples = 100  # 时移样本数
t0 = shift_samples * dt  # 时移时间

# 时域时移 (循环移位)
f1_shifted_time = torch.roll(f1, shift_samples)

# 频域时移: F(ω) * exp(-jωt₀)
F1 = torch.fft.fft(f1)
phase_shift = torch.exp(-1j * omega * t0)
f1_shifted_freq = torch.fft.ifft(F1 * phase_shift).real

# 比较结果
error_shift = torch.mean((f1_shifted_time - f1_shifted_freq)**2)
print(f"时移性质验证:")
print(f"  时域时移与频域相位调制的均方误差: {error_shift:.8f}")
print(f"  最大绝对误差: {torch.max(torch.abs(f1_shifted_time - f1_shifted_freq)):.8f}")

# 可视化
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(t[:300], f1[:300], 'k-', label='原信号', linewidth=2)
plt.plot(t[:300], f1_shifted_time[:300], 'b-', label='时域时移', linewidth=2)
plt.plot(t[:300], f1_shifted_freq[:300], 'r--', label='频域相位调制', linewidth=2)
plt.legend()
plt.title('时移性质验证')
plt.xlabel('时间 (s)')
plt.ylabel('幅值')

plt.subplot(1, 3, 2)
plt.plot(t[:300], (f1_shifted_time - f1_shifted_freq)[:300], 'g-', linewidth=2)
plt.title('误差 (时域 - 频域)')
plt.xlabel('时间 (s)')
plt.ylabel('误差')

# 显示频域相位差
plt.subplot(1, 3, 3)
phase_diff = torch.angle(phase_shift)
freqs_plot = freqs[:len(freqs)//2]  # 只显示正频率
plt.plot(freqs_plot, phase_diff[:len(freqs)//2], 'm-', linewidth=2)
plt.title(f'相位调制 (时移 {t0:.3f}s)')
plt.xlabel('频率 (Hz)')
plt.ylabel('相位 (rad)')
plt.tight_layout()
plt.show()
```

![](D:\24bo\扩散模型\AFNO-transformer-master\figures\shift.png)

性能对比：时域 vs 频域操作

```python
print("\n" + "=" * 50)
print("5. 性能对比")
print("=" * 50)

import time

# 测试不同长度的信号
signal_lengths = [1024, 2048, 4096, 8192]
time_conv_results = []
freq_conv_results = []

for N in signal_lengths:
    # 生成测试信号
    x = torch.randn(N)
    h = torch.randn(N//4)  # 卷积核
    
    # 时域卷积
    start_time = time.time()
    for _ in range(10):  # 重复10次取平均
        conv_time = F.conv1d(x.unsqueeze(0).unsqueeze(0), 
                           h.flip(0).unsqueeze(0).unsqueeze(0), 
                           padding=N//4-1).squeeze()
    time_conv = (time.time() - start_time) / 10
    time_conv_results.append(time_conv)
    
    # 频域卷积
    start_time = time.time()
    for _ in range(10):  # 重复10次取平均
        X = torch.fft.fft(x)
        H = torch.fft.fft(F.pad(h, (0, N-len(h))))  # 零填充到相同长度
        conv_freq = torch.fft.ifft(X * H).real
    freq_conv = (time.time() - start_time) / 10
    freq_conv_results.append(freq_conv)
    
    print(f"信号长度 {N}: 时域卷积 {time_conv:.6f}s, 频域卷积 {freq_conv:.6f}s, "
          f"加速比 {time_conv/freq_conv:.2f}x")

# 可视化性能对比
plt.figure(figsize=(10, 6))
plt.loglog(signal_lengths, time_conv_results, 'bo-', label='时域卷积', linewidth=2, markersize=8)
plt.loglog(signal_lengths, freq_conv_results, 'ro-', label='频域卷积', linewidth=2, markersize=8)
plt.xlabel('信号长度')
plt.ylabel('计算时间 (秒)')
plt.title('时域 vs 频域卷积性能对比')
plt.legend()
plt.grid(True)
plt.show()
```

![](D:\24bo\扩散模型\AFNO-transformer-master\figures\ratio.png)

import numpy as np

def generate_bluetooth_frequencies_1mhz():
    """
    生成1MHz间隔的蓝牙频率列表
    
    返回:
    frequencies -- 所有蓝牙信道频率 [Hz]
    """
    # 蓝牙BLE信道参数 (1MHz间隔)
    n_channels = 80  # 1MHz间隔下80个信道覆盖2402-2482MHz
    channel_spacing = 1e6  # 1MHz信道间隔
    start_frequency = 2402e6  # 起始频率 2402MHz
    
    # 生成所有蓝牙信道频率
    frequencies = np.array([start_frequency + i * channel_spacing for i in range(n_channels)])
    
    return frequencies

def generate_mask_list(unused_channels, total_channels=80):
    """
    生成掩码列表
    
    参数:
    unused_channels -- 未被使用的信道索引列表
    total_channels -- 总信道数
    
    返回:
    mask_list -- 掩码列表，True表示该信道未被使用，False表示被使用
    """
    mask_list = np.zeros(total_channels, dtype=bool)  # 初始化为全部使用
    
    # 将未使用的信道设为True
    mask_list[unused_channels] = True
    
    return mask_list

def compensate_cable_phase(full_spectrum, cable_length=6.0, velocity_factor=0.66):
    """
    补偿线缆在不同频率上引入的相位偏移。

    参数:
    full_spectrum (np.array): 测量的复数频谱，从蓝牙信道0开始。
    cable_length (float): 单根线缆的物理长度，单位：米。
    velocity_factor (float): 线缆的速度因子。

    返回:
    np.array: 补偿线缆相位影响后的复数频谱。
    """
    num_channels = len(full_spectrum)

    electrical_length = cable_length / velocity_factor  # 单位：米
    
    bluetooth_ch0_center_mhz = 2402
    frequencies = (bluetooth_ch0_center_mhz + np.arange(num_channels)) * 1e6
    
    # 3. 计算线缆引入的相位偏移 (单位: 弧度)
    # 时延 tau = electrical_length / c
    # 相位偏移 phi_cable = -2 * pi * f * tau
    c = 299792458
    time_delay = electrical_length / c
    phase_offsets = -2 * np.pi * frequencies * time_delay
    
    # 4. 构建补偿因子并应用
    # 补偿因子 = cos(phi) + j * sin(phi) = exp(j * (-phi_cable))
    # 因为我们要减去线缆的影响
    compensation_factors = np.exp(1j * (-phase_offsets))
    
    # 应用补偿
    compensated_spectrum = full_spectrum * compensation_factors
    
    return compensated_spectrum

def combine_iq_data(i_data, q_data):
    """
    将分开的I/Q数据组合成复数形式
    
    参数:
    i_data -- 同相分量数据
    q_data -- 正交分量数据
    
    返回:
    iq_data -- 复数形式的I/Q数据
    """
    return np.array(i_data) + 1j * np.array(q_data)

def reconstruct_full_spectrum(i_data, q_data, mask_list):
    """
    将部分信道的I/Q数据重建为完整频谱
    
    参数:
    i_data -- 同相分量数据
    q_data -- 正交分量数据
    mask_list -- 掩码列表，True表示未使用，False表示使用
    
    返回:
    full_spectrum -- 完整的频谱（未使用的信道填充为0）
    """
    n_channels = len(mask_list)
    full_spectrum = np.zeros(n_channels, dtype=complex)
    
    # 将I/Q数据组合成复数
    iq_data_used = combine_iq_data(i_data, q_data)
    
    # 找到被使用的信道索引 (mask_list中为False的位置)
    used_channel_indices = np.where(~mask_list)[0]
    
    # 将使用的信道数据填入对应位置
    full_spectrum[used_channel_indices] = iq_data_used
    full_spectrum = compensate_cable_phase(full_spectrum)

    return full_spectrum

def bluetooth_range_estimation(i_data, q_data, mask_list, c=3e8):
    """
    基于蓝牙信道的多载波测距
    
    参数:
    i_data -- 同相分量数据
    q_data -- 正交分量数据
    mask_list -- 掩码列表
    c -- 光速 [m/s]
    
    返回:
    range_estimate -- 距离估计 [m]
    ifft_result -- IFFT结果
    full_spectrum -- 完整频谱
    peak_index -- 峰值索引
    """
    # 重建完整频谱
    full_spectrum = reconstruct_full_spectrum(i_data, q_data, mask_list)
    
    # 执行IFFT
    ifft_result = np.fft.ifft(full_spectrum)
    
    # 计算幅度谱
    magnitude = np.abs(ifft_result)
    
    # 找到峰值（忽略第一个点，可能是DC分量）
    peak_index = np.argmax(magnitude[1:]) + 1
    
    # 计算距离参数
    n_channels = len(mask_list)
    start_freq = 2402e6
    end_freq = start_freq + (n_channels - 1) * 1e6
    bandwidth = end_freq - start_freq
    
    # 距离分辨率
    range_resolution = c / (2 * bandwidth)
    
    # 距离估计
    range_estimate = peak_index * range_resolution
    
    return range_estimate, ifft_result, full_spectrum, peak_index

def advanced_bluetooth_range_estimation(i_data, q_data, mask_list, c=3e8, zero_padding_factor=4):
    """
    增强版的蓝牙测距算法，使用零填充和窗函数
    
    参数:
    i_data -- 同相分量数据
    q_data -- 正交分量数据
    mask_list -- 掩码列表
    c -- 光速 [m/s]
    zero_padding_factor -- 零填充倍数
    
    返回:
    range_estimate -- 距离估计 [m]
    ifft_result -- IFFT结果
    full_spectrum -- 完整频谱
    """
    n_channels = len(mask_list)
    
    # 重建完整频谱
    full_spectrum = reconstruct_full_spectrum(i_data, q_data, mask_list)
    
    # 零填充以提高分辨率
    spectrum_padded = np.zeros(n_channels * zero_padding_factor, dtype=complex)
    spectrum_padded[:n_channels] = full_spectrum
    
    # 应用窗函数减少频谱泄漏
    window = np.hanning(n_channels)
    spectrum_padded[:n_channels] *= window
    
    # 执行IFFT
    ifft_result = np.fft.ifft(spectrum_padded)
    
    # 计算幅度谱
    magnitude = np.abs(ifft_result)
    
    # 找到峰值（在前半部分搜索，避免镜像）
    search_range = len(magnitude) // 2
    peak_index = np.argmax(magnitude[1:search_range]) + 1
    
    # 计算距离参数
    start_freq = 2402e6
    end_freq = start_freq + (n_channels - 1) * 1e6
    bandwidth = end_freq - start_freq
    
    # 考虑零填充的距离分辨率
    range_resolution = c / (2 * bandwidth * zero_padding_factor)
    
    # 距离估计
    range_estimate = peak_index * range_resolution
    
    return range_estimate, ifft_result, full_spectrum, peak_index

def generate_range_resolution(mask_list, c=3e8, zero_padding_factor=4):
    n_channels = len(mask_list)
    start_freq = 2402e6
    end_freq = start_freq + (n_channels - 1) * 1e6
    bandwidth = end_freq - start_freq
    # 考虑零填充的距离分辨率
    range_resolution = c / (2 * bandwidth * zero_padding_factor)
    return range_resolution
def quick_sinc_interp(magnitude_spectrum, coarse_index, window_size=6, range_resolution=0.5):
    """
    快速Sinc插值版本，返回精确索引（不返回插值曲线）
    
    参数:
    magnitude_spectrum -- IFFT后的幅度谱
    coarse_index -- 粗略的峰值索引位置
    window_size -- 用于插值的窗口大小
    
    返回:
    fine_index -- 精确的峰值位置（小数索引）
    """
    
    start_idx = max(0, coarse_index - window_size)
    end_idx = min(len(magnitude_spectrum), coarse_index + window_size + 1)
    
    window_indices = np.arange(start_idx, end_idx)
    window_data = magnitude_spectrum[start_idx:end_idx]
    
    # 在粗略峰值附近进行密集插值
    interp_range = 2.0  # 插值范围
    n_points = 50       # 插值点数
    
    interp_x = np.linspace(coarse_index - interp_range, coarse_index + interp_range, n_points)
    interp_y = np.zeros_like(interp_x)
    
    for i, x in enumerate(interp_x):
        sinc_sum = 0.0
        for j, orig_x in enumerate(window_indices):
            dx = x - orig_x
            if abs(dx) < 1e-10:
                sinc_val = 1.0
            else:
                sinc_val = np.sin(np.pi * dx) / (np.pi * dx)
            sinc_sum += window_data[j] * sinc_val
        interp_y[i] = sinc_sum
    
    peak_index = interp_x[np.argmax(interp_y)]
    return peak_index
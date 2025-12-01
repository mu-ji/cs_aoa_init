import os
import numpy as np
import matplotlib.pyplot as plt

def get_data_by_address_and_index(address, index=0, output_dir="iq_data"):
    """
    根据设备地址和索引获取特定设备的数据
    
    参数:
    address: 设备地址
    index: 数据索引 (默认为0，即第一条数据)
    output_dir: 数据文件目录
    """
    filename = address + ".npy"
    filepath = os.path.join(output_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"设备 {address} 的数据文件不存在")
        return None
    
    try:
        # 加载numpy数据
        data_array = np.load(filepath, allow_pickle=True)
        
        if len(data_array) == 0:
            print(f"设备 {address} 的数据文件为空")
            return None
        
        if index >= len(data_array):
            print(f"索引 {index} 超出范围，设备 {address} 只有 {len(data_array)} 条数据")
            return None
        
        # 返回指定索引的数据
        selected_data = data_array[index]
        return {
            'address': address,
            'i_local': selected_data['i_local'],
            'q_local': selected_data['q_local'],
            'i_remote': selected_data['i_remote'],
            'q_remote': selected_data['q_remote'],
            'distance_data': selected_data['distance_data'],
            'timestamp': selected_data['timestamp'],
            'index': index,
            'total_records': len(data_array)
        }
        
    except Exception as e:
        print(f"读取设备 {address} 的数据时出错: {e}")
        return None

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def linear_fit_phase_data(phase_data, outlier_threshold=2.0):
    """
    对相位数据进行线性回归拟合，使用Z-score方法去除离群值
    
    参数:
    phase_data: 相位数据数组 (local phase + remote phase, 已经unwrap)
    outlier_threshold: 离群值检测阈值
    
    返回:
    slope: 拟合直线的斜率 (单位: radians/MHz)
    intercept: 拟合直线的截距 (单位: radians)
    inlier_mask: 内点掩码 (True表示内点，False表示离群值)
    """
    
    # 准备数据 - 将索引映射到蓝牙信道频率 (2404MHz 到 2479MHz)
    x_indices = np.arange(len(phase_data))
    x_freq = 2404 + x_indices  # 映射到频率: 2404, 2405, ..., 2479 MHz
    y = phase_data
    
    # 初始线性拟合
    slope_init, intercept_init, _, _, _ = stats.linregress(x_freq, y)
    y_pred_init = slope_init * x_freq + intercept_init
    residuals = y - y_pred_init
    
    # 使用Z-score方法检测离群值
    z_scores = np.abs(stats.zscore(residuals))
    inlier_mask = z_scores < outlier_threshold
    
    # 使用内点重新拟合
    if np.sum(inlier_mask) > 2:  # 至少需要3个点才能拟合
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_freq[inlier_mask], y[inlier_mask])
    else:
        slope, intercept, r_value, p_value, std_err = slope_init, intercept_init, 0, 0, 0
        print("警告：内点数量不足，使用初始拟合结果")
    
    # 计算拟合值
    y_fit = slope * x_freq + intercept
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    
    # 绘制原始数据和拟合直线
    plt.scatter(x_freq[inlier_mask], y[inlier_mask], color='blue', alpha=0.7, s=50, 
                label=f'内点 ({np.sum(inlier_mask)}个)')
    plt.scatter(x_freq[~inlier_mask], y[~inlier_mask], color='red', alpha=0.7, s=50, 
                label=f'离群值 ({np.sum(~inlier_mask)}个)', marker='x')
    
    plt.plot(x_freq, y_fit, 'r-', linewidth=2, 
             label=f'拟合直线: y = {slope:.6f}x + {intercept:.3f}')
    
    plt.xlabel('频率 (MHz)')
    plt.ylabel('相位 (radians)')
    plt.title('相位数据线性拟合 - 按蓝牙信道频率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度，每10MHz显示一个刻度
    plt.xticks(np.arange(2400, 2481, 10))
    
    # 添加统计信息文本框
    textstr = '\n'.join((
        f'斜率: {slope:.8f} radians/MHz',
        f'截距: {intercept:.6f} radians',
        f'相关系数 R²: {r_value**2:.4f}',
        f'标准误差: {std_err:.6f}',
        f'内点比例: {np.sum(inlier_mask)}/{len(phase_data)}',
        f'频率范围: {x_freq[0]}-{x_freq[-1]} MHz'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    # 打印结果
    print(f"拟合结果:")
    print(f"  斜率: {slope:.8f} radians/MHz")
    print(f"  截距: {intercept:.6f} radians")
    print(f"  相关系数 R²: {r_value**2:.4f}")
    print(f"  标准误差: {std_err:.6f}")
    print(f"  内点数量: {np.sum(inlier_mask)}/{len(phase_data)}")
    print(f"  频率范围: {x_freq[0]}-{x_freq[-1]} MHz")
    
    return slope, intercept, inlier_mask


# 获取数据
rx1_data = get_data_by_address_and_index('DB_59_E3_17_FC_79', index=2)  # 使用第一条数据
rx2_data = get_data_by_address_and_index('E9_D2_FF_FF_96_E8', index=2)  # 使用第一条数据

if rx1_data is not None and rx2_data is not None:
    print(f"RX1: 设备 {rx1_data['address']}, 索引 {rx1_data['index']}/{rx1_data['total_records']-1}, 时间 {rx1_data['timestamp']}")
    print(f"RX2: 设备 {rx2_data['address']}, 索引 {rx2_data['index']}/{rx2_data['total_records']-1}, 时间 {rx2_data['timestamp']}")
    
    # 提取数据
    rx1_i_local = rx1_data['i_local']
    rx1_q_local = rx1_data['q_local']
    rx1_i_remote = rx1_data['i_remote']
    rx1_q_remote = rx1_data['q_remote']

    rx2_i_local = rx2_data['i_local']
    rx2_q_local = rx2_data['q_local']
    rx2_i_remote = rx2_data['i_remote']
    rx2_q_remote = rx2_data['q_remote']

    # 计算相位
    rx1_local_phase = np.arctan2(rx1_q_local, rx1_i_local)
    rx1_remote_phase = np.arctan2(rx1_q_remote, rx1_i_remote)

    rx2_local_phase = np.arctan2(rx2_q_local, rx2_i_local)
    rx2_remote_phase = np.arctan2(rx2_q_remote, rx2_i_remote)

    # 绘制图形
    plt.figure(figsize=(12, 8))
    
    # 相位和
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(rx1_local_phase)), np.unwrap(rx1_local_phase + rx1_remote_phase), 
                label='RX1 Phase Sum', alpha=0.7, s=30)
    plt.scatter(range(len(rx2_local_phase)), np.unwrap(rx2_local_phase + rx2_remote_phase), 
                label='RX2 Phase Sum', alpha=0.7, s=30)
    plt.xlabel('Sample Index')
    plt.ylabel('Unwrapped Phase (radians)')
    plt.title('Phase Sum Comparison (Local + Remote)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 分别显示本地和远程相位
    plt.subplot(2, 1, 2)
    plt.scatter(range(len(rx1_local_phase)), np.unwrap(rx1_local_phase), 
                label='RX1 Local Phase', alpha=0.7, s=30)
    plt.scatter(range(len(rx1_remote_phase)), np.unwrap(rx1_remote_phase), 
                label='RX1 Remote Phase', alpha=0.7, s=30)
    plt.scatter(range(len(rx2_local_phase)), np.unwrap(rx2_local_phase), 
                label='RX2 Local Phase', alpha=0.7, s=30)
    plt.scatter(range(len(rx2_remote_phase)), np.unwrap(rx2_remote_phase), 
                label='RX2 Remote Phase', alpha=0.7, s=30)
    plt.xlabel('Sample Index')
    plt.ylabel('Unwrapped Phase (radians)')
    plt.title('Individual Phase Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

else:
    print("无法加载一个或两个设备的数据，请检查设备地址是否正确")

slope, intercept, inlier_mask = linear_fit_phase_data(np.unwrap(rx1_local_phase + rx1_remote_phase), outlier_threshold=2.0)
print(rx1_data['distance_data']['phase_slope'])
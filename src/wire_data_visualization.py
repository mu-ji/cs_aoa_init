import os
import numpy as np
import matplotlib.pyplot as plt

def load_numpy_data(filepath):
    """
    加载并查看numpy文件中的数据
    """
    try:
        data = np.load(filepath, allow_pickle=True).item()
        print(f"文件 {filepath} 中包含 {len(data)} 个设备地址的数据")
        total_records = 0
        for address, records in data.items():
            print(f"\n设备地址: {address}")
            print(f"  记录数量: {len(records)}")
            total_records += len(records)
            if records:
                print(f"  最后一条记录时间: {records[-1]['timestamp']}")
                print(f"  距离数据: ifft={records[-1]['distance_data']['ifft']:.3f}, "
                      f"phase_slope={records[-1]['distance_data']['phase_slope']:.3f}")
        print(f"\n总记录数: {total_records}")
        return data
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None


def iq_pre_process(i_local, q_local, i_remote, q_remote):
    # iq_result = [0]*2*len(i_local)
    i_result = [0]*len(i_local)
    q_result = [0]*len(i_local)
    for i in range(len(i_local)):
        i_result[i] = i_local[i]*i_remote[i] - q_local[i]*q_remote[i]
        q_result[i] = i_local[i]*q_remote[i] + q_local[i]*i_remote[i]
        # iq_result[2*i] = i_local[i]*i_remote[i] - q_local[i]*q_remote[i]
        # iq_result[2*i+1] = i_local[i]*q_remote[i] + q_local[i]*i_remote[i]
    
    return i_result, q_result

def plot_phase_data(plt, data_array, data_length, color, labels):
    for i in range(data_length):  # 遍历所有数据
        data = data_array[i]

        rx2_i_local = data['i_local']
        rx2_q_local = data['q_local']
        rx2_i_remote = data['i_remote']
        rx2_q_remote = data['q_remote']

        rx2_local_phase = np.arctan2(rx2_q_local, rx2_i_local)
        rx2_remote_phase = np.arctan2(rx2_q_remote, rx2_i_remote)
        phase = (rx2_local_phase + rx2_remote_phase)
        phase = np.arctan2(np.sin(phase), np.cos(phase))
        phase = np.unwrap(phase)
        
        # i_result, q_result = iq_pre_process(rx2_i_local, rx2_q_local, rx2_i_remote, rx2_q_remote)
        # phase = np.arctan2(q_result, i_result)
        # phase = np.unwrap(phase)

        # 根据基础颜色生成渐变色
        if color == 'red':
            # 红色系渐变：从浅红到深红
            red_intensity = 0.3 + 0.7 * (i / max(1, data_length-1))
            plot_color = (1.0, 0.2 * red_intensity, 0.2 * red_intensity)
        elif color == 'blue':
            # 蓝色系渐变：从浅蓝到深蓝
            blue_intensity = 0.3 + 0.7 * (i / max(1, data_length-1))
            plot_color = (0.2 * blue_intensity, 0.2 * blue_intensity, 1.0)
        else:
            # 其他颜色使用原色
            plot_color = color
            
        plt.scatter(range(len(phase)), phase, alpha=0.7, s=30, c=[plot_color], label='{} index {}'.format(labels, i))
    return

# 创建2行4列的子图，返回figure和axes数组
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

# 将二维的axs数组展平为一维，方便按索引访问
axs_flat = axs.flatten()

for index in range(6):
    data = load_numpy_data('wire_data/data_{}.npy'.format(index))
    rx1_data_array = data['E9:D2:FF:FF:96:E8']
    rx2_data_array = data['DB:59:E3:17:FC:79']

    rx1_data_length = len(rx1_data_array)
    rx2_data_length = len(rx2_data_array)
    
    # 在对应的子图中绘图
    plot_phase_data(axs_flat[index], rx1_data_array, rx1_data_length, 'red', 'E9:D2:FF:FF:96:E8')
    plot_phase_data(axs_flat[index], rx2_data_array, rx2_data_length, 'blue', 'DB:59:E3:17:FC:79')
    
    # 为每个子图添加标题（可选）
    axs_flat[index].set_title(f'Subplot {index+1}')
    
    # 为每个子图添加图例（可选）
    axs_flat[index].legend()

# 调整子图之间的间距
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2)
axs_flat = axs.flatten()
data = load_numpy_data('wire_data/data_0_0.npy')

rx1_data_array = data['E9:D2:FF:FF:96:E8']
rx2_data_array = data['DB:59:E3:17:FC:79']

rx1_data_length = len(rx1_data_array)
rx2_data_length = len(rx2_data_array)

# 在对应的子图中绘图
plot_phase_data(axs_flat[0], rx1_data_array, rx1_data_length, 'red', 'E9:D2:FF:FF:96:E8')
plot_phase_data(axs_flat[0], rx2_data_array, rx2_data_length, 'blue', 'DB:59:E3:17:FC:79')

data = load_numpy_data('wire_data/data_0_1.npy')

rx1_data_array = data['E9:D2:FF:FF:96:E8']
rx2_data_array = data['DB:59:E3:17:FC:79']

rx1_data_length = len(rx1_data_array)
rx2_data_length = len(rx2_data_array)

# 在对应的子图中绘图
plot_phase_data(axs_flat[1], rx1_data_array, rx1_data_length, 'red', 'E9:D2:FF:FF:96:E8')
plot_phase_data(axs_flat[1], rx2_data_array, rx2_data_length, 'blue', 'DB:59:E3:17:FC:79')

plt.legend()
plt.tight_layout()
plt.show()
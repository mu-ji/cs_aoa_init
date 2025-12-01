import os
import numpy as np
import matplotlib.pyplot as plt
import my_slope

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
                # print(f"  距离数据: ifft={records[-1]['distance_data']['ifft']:.3f}, "
                #       f"phase_slope={records[-1]['distance_data']['phase_slope']:.3f}")
        print(f"\n总记录数: {total_records}")
        return data
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

# data = load_numpy_data('iq_data/data_0.npy')
# data = load_numpy_data('wire_data/data_0.npy')
data = load_numpy_data('antenna_equipment_data/data_1.npy')
print(data)
rx1_data_array = data['E9:D2:FF:FF:96:E8']
rx2_data_array = data['DB:59:E3:17:FC:79']
# rx1_data_array = np.load('iq_data/E9_D2_FF_FF_96_E8.npy', allow_pickle=True)
rx1_data_length = len(rx1_data_array)
print(rx1_data_length)
# rx2_data_array = np.load('iq_data/DB_59_E3_17_FC_79.npy', allow_pickle=True)
rx2_data_length = len(rx2_data_array)


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
    # for i in range(1):
        data = data_array[i]

        rx2_i_local = data['i_local']
        rx2_q_local = data['q_local']
        rx2_i_remote = data['i_remote']
        rx2_q_remote = data['q_remote']

        # rx2_local_phase = np.arctan2(rx2_q_local, rx2_i_local)
        # rx2_remote_phase = np.arctan2(rx2_q_remote, rx2_i_remote)
        # phase = (rx2_local_phase + rx2_remote_phase)
        # phase = np.arctan2(np.sin(phase), np.cos(phase))

        i_result, q_result = iq_pre_process(rx2_i_local, rx2_q_local, rx2_i_remote, rx2_q_remote)
        phase = np.arctan2(q_result, i_result)
        phase = np.unwrap(phase)

        
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

plt.figure(figsize=(12, 6))
plot_phase_data(plt,rx1_data_array, rx1_data_length, 'red', 'E9:D2:FF:FF:96:E8')
plot_phase_data(plt,rx2_data_array, rx2_data_length, 'blue', 'DB:59:E3:17:FC:79')

plt.legend()
plt.show()

def ifft_from_iq(plt, data_array, data_length):

    for i in range(data_length):
        data = data_array[i]

        rx2_i_local = data['i_local']
        rx2_q_local = data['q_local']
        rx2_i_remote = data['i_remote']
        rx2_q_remote = data['q_remote']

        i_result, q_result = iq_pre_process(rx2_i_local, rx2_q_local, rx2_i_remote, rx2_q_remote)
        complex_data = np.array(i_result) + 1j * np.array(q_result)
    # 执行逆傅里叶变换
        ifft_result = np.fft.ifft(complex_data)
        plt.plot(np.abs(ifft_result))
    return 

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes = axes.flatten()
plot_phase_data(axes[0],rx1_data_array, rx1_data_length, 'red', 'E9:D2:FF:FF:96:E8')
plot_phase_data(axes[0],rx2_data_array, rx2_data_length, 'blue', 'DB:59:E3:17:FC:79')

ifft_from_iq(axes[1], rx1_data_array, rx1_data_length)

plt.show()


import my_ifft

frequencies = my_ifft.generate_bluetooth_frequencies_1mhz()

unused_channels = [0, 1, 77, 78, 79]

mask_list = my_ifft.generate_mask_list(unused_channels, len(frequencies))

used_channel_indices = np.where(~mask_list)[0]
print(used_channel_indices)


LIGHTSPEED = 299792458

for i in range(rx1_data_length):
    data = rx2_data_array[i]

    rx1_i_local = data['i_local'] 
    rx1_q_local = data['q_local']
    rx1_i_remote = data['i_remote']
    rx1_q_remote = data['q_remote']

    i_result, q_result = iq_pre_process(rx1_i_local, rx1_q_local, rx1_i_remote, rx1_q_remote)

    phase = my_slope.unwrapped_phase(i_result, q_result)
    slope = my_slope.cal_slope(phase)
    distance = my_slope.cal_distance(slope)

    range_basic, ifft_basic, full_spectrum, peak_index = my_ifft.bluetooth_range_estimation(i_result, q_result, mask_list, LIGHTSPEED)

    range_advanced, ifft_advanced, full_spectrum, peak_index = my_ifft.advanced_bluetooth_range_estimation(i_result, q_result, mask_list, LIGHTSPEED, 4)
    # plt.figure()
    # plt.plot(np.abs(ifft_basic))
    # plt.plot(np.abs(ifft_advanced))
    # plt.show()
    sinc_index = my_ifft.quick_sinc_interp(np.abs(ifft_advanced), peak_index, window_size=6)
    resolution = my_ifft.generate_range_resolution(mask_list, 4)

    print(data['distance_data'])
    print(range_basic/2)
    print(range_advanced/2)
    print(distance/2)
    print(sinc_index*resolution/2)


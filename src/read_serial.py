import re
import os
import serial
import serial.tools.list_ports
import numpy as np
import json
from datetime import datetime

def sanitize_filename(filename):
    """
    清理文件名，移除无效字符
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    filename = filename.strip().rstrip('.')
    return filename

def extract_data_from_buffer(buffer):
    """
    从缓冲区中提取数据块
    """
    data_blocks = []
    
    # 使用正则表达式匹配数据块
    pattern = r"-----------------------------------(.*?)-----------------------------------"
    matches = re.findall(pattern, buffer, re.DOTALL)
    
    for match in matches:
        lines = match.strip().split('\n')
        
        # 提取数组数据
        i_local = []
        q_local = []
        i_remote = []
        q_remote = []
        distance_data = {}
        current_address = "unknown"
        
        for line in lines:
            line = line.strip()
            
            # 匹配数组数据行
            if "i_local :" in line:
                match_obj = re.search(r"i_local\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if match_obj:
                    value = float(match_obj.group(1))
                    i_local.append(value)
            elif "q_local :" in line:
                match_obj = re.search(r"q_local\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if match_obj:
                    value = float(match_obj.group(1))
                    q_local.append(value)
            elif "i_remote :" in line:
                match_obj = re.search(r"i_remote\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if match_obj:
                    value = float(match_obj.group(1))
                    i_remote.append(value)
            elif "q_remote :" in line:
                match_obj = re.search(r"q_remote\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if match_obj:
                    value = float(match_obj.group(1))
                    q_remote.append(value)
            
            # 匹配距离数据行 - 关键修改：处理nan值
            elif "Distance estimates" in line:
                # 匹配地址（可能包含冒号）
                address_match = re.search(r"Address:\s*([^\s,]+)", line)
                if address_match:
                    current_address = address_match.group(1)
                
                # 匹配ifft值（可能为nan）
                ifft_match = re.search(r"ifft:\s*(nan|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line, re.IGNORECASE)
                # 匹配phase_slope值（可能为nan）
                phase_slope_match = re.search(r"phase_slope:\s*(nan|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line, re.IGNORECASE)
                # 匹配rtt值（可能为nan）
                rtt_match = re.search(r"rtt:\s*(nan|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line, re.IGNORECASE)
                
                # 构建distance_data字典，允许值为nan
                distance_data = {}
                if current_address != "unknown":
                    distance_data['address'] = current_address
                
                # 处理ifft值
                if ifft_match:
                    ifft_str = ifft_match.group(1).lower()
                    if ifft_str == 'nan':
                        distance_data['ifft'] = float('nan')
                    else:
                        try:
                            distance_data['ifft'] = float(ifft_str)
                        except ValueError:
                            distance_data['ifft'] = float('nan')
                else:
                    distance_data['ifft'] = float('nan')
                
                # 处理phase_slope值
                if phase_slope_match:
                    phase_slope_str = phase_slope_match.group(1).lower()
                    if phase_slope_str == 'nan':
                        distance_data['phase_slope'] = float('nan')
                    else:
                        try:
                            distance_data['phase_slope'] = float(phase_slope_str)
                        except ValueError:
                            distance_data['phase_slope'] = float('nan')
                else:
                    distance_data['phase_slope'] = float('nan')
                
                # 处理rtt值
                if rtt_match:
                    rtt_str = rtt_match.group(1).lower()
                    if rtt_str == 'nan':
                        distance_data['rtt'] = float('nan')
                    else:
                        try:
                            distance_data['rtt'] = float(rtt_str)
                        except ValueError:
                            distance_data['rtt'] = float('nan')
                else:
                    distance_data['rtt'] = float('nan')
        
        # 只有当数组长度正确时才保存（75个元素），即使distance_data中有nan值
        if len(i_local) == 75 and len(q_local) == 75 and len(i_remote) == 75 and len(q_remote) == 75:
            data_block = {
                'address': current_address,
                'i_local': np.array(i_local, dtype=np.float64),
                'q_local': np.array(q_local, dtype=np.float64),
                'i_remote': np.array(i_remote, dtype=np.float64),
                'q_remote': np.array(q_remote, dtype=np.float64),
                'distance_data': distance_data,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            data_blocks.append(data_block)
        elif current_address != "unknown":
            # 如果数组长度不正确，但至少有地址信息，可以保存一个警告记录
            print(f"警告：数据块地址 {current_address} 的数组长度不正确（local_I={len(i_local)}, local_Q={len(q_local)}, remote_I={len(i_remote)}, remote_Q={len(q_remote)}）")
    
    return data_blocks

def initialize_data_file(output_dir="iq_data", filename_prefix="combined_data"):
    """
    初始化数据文件，如果文件存在则清空
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = sanitize_filename(filename_prefix) + ".npy"
    filepath = os.path.join(output_dir, filename)
    
    # 如果文件存在，删除它
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"已清空现有文件: {filename}")
    
    # 创建空的字典结构
    empty_data = {}
    np.save(filepath, empty_data)
    
    return filepath

def save_data_to_numpy_file(data_block, filepath, max_records=10):
    """
    将不同address的数据保存到同一个numpy文件中，按address分类
    当达到最大记录数时返回True
    """
    # 准备要保存的数据结构 - 使用address作为key
    address = data_block['address']
    data_dict = {
        'timestamp': data_block['timestamp'],
        'i_local': data_block['i_local'],
        'q_local': data_block['q_local'],
        'i_remote': data_block['i_remote'],
        'q_remote': data_block['q_remote'],
        'distance_data': data_block['distance_data']
    }
    
    # 读取现有数据
    try:
        existing_data = np.load(filepath, allow_pickle=True).item()
    except Exception as e:
        print(f"读取文件时出错，创建新数据结构: {e}")
        existing_data = {}
    
    # 如果该address不存在，初始化列表
    if address not in existing_data:
        existing_data[address] = []
    
    # 将新数据追加到对应address的列表中
    existing_data[address].append(data_dict)
    
    # 检查是否达到最大记录数
    total_records = sum(len(records) for records in existing_data.values())
    reached_max = total_records >= max_records
    
    # 保存为numpy文件
    np.save(filepath, existing_data)
    
    return existing_data, reached_max

def read_and_process_serial(port=None,out_folder='iq_data', baudrate=115200, timeout=1, filename_prefix="combined_data", max_records=10):
    """
    从串口读取数据并实时处理保存，所有address的数据保存到同一个numpy文件中
    当保存了指定数量的数据后停止程序
    """
    # 自动检测串口
    if port is None:
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("未找到可用的串口")
            return
        port = ports[0].device
        print(f"自动选择串口: {port}")
    
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"已连接串口 {port}，波特率 {baudrate}")
        print(f"等待数据... (将在收集 {max_records} 组数据后自动停止)")
        print(f"数据将保存到: {filename_prefix}.npy")
        
        # 初始化数据文件（清空现有文件）
        filepath = initialize_data_file(output_dir=out_folder, filename_prefix=filename_prefix)
        print("数据文件已初始化，开始记录...")
        
        buffer = ""
        data_count = 0
        address_counts = {}  # 记录每个address的数据计数
        should_stop = False
        
        while not should_stop:
            if ser.in_waiting > 0:
                # 读取数据并解码
                data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                print(data, end='')  # 实时显示串口输出
                buffer += data
                
                # 检查是否包含完整的数据块
                if "-----------------------------------" in buffer:
                    # 提取数据
                    data_blocks = extract_data_from_buffer(buffer)
                    
                    for data_block in data_blocks:
                        data_count += 1
                        address = data_block['address']
                        
                        # 更新address计数
                        if address not in address_counts:
                            address_counts[address] = 0
                        address_counts[address] += 1
                        
                        # 保存数据到numpy文件
                        all_data, reached_max = save_data_to_numpy_file(
                            data_block, 
                            filepath=filepath,
                            max_records=max_records
                        )
                        
                        print(f"\n[数据块 #{data_count}] 地址: {address}")
                        print(f"  距离估计 - ifft: {data_block['distance_data']['ifft']:.3f}, "
                              f"phase_slope: {data_block['distance_data']['phase_slope']:.3f}")
                        
                        # 显示当前数据统计
                        total_current = sum(len(records) for records in all_data.values())
                        print(f"  当前数据统计: 总记录数 {total_current}/{max_records}")
                        for addr, records in all_data.items():
                            print(f"    {addr}: {len(records)} 条记录")
                        
                        # 检查是否达到停止条件
                        if reached_max:
                            should_stop = True
                            print(f"\n已达到最大记录数 {max_records}，停止采集")
                            break
                        
                        # 从缓冲区中移除已处理的数据
                        start_index = buffer.find("-----------------------------------")
                        end_index = buffer.find("-----------------------------------", start_index + 1)
                        if end_index != -1:
                            buffer = buffer[end_index + len("-----------------------------------"):]
                        else:
                            buffer = ""
                
                # 防止缓冲区过大
                if len(buffer) > 10000:
                    buffer = buffer[-5000:]
                    
    except serial.SerialException as e:
        print(f"串口错误: {e}")
    except KeyboardInterrupt:
        print(f"\n\n用户中断程序")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("串口已关闭")
        
        # 最终统计
        print(f"\n最终统计:")
        print(f"总共处理了 {data_count} 个数据块")
        print(f"记录了 {len(address_counts)} 个不同设备地址的数据")
        for address, count in address_counts.items():
            print(f"  {address}: {count} 个数据块")
        if 'filepath' in locals():
            print(f"所有数据已保存到: {filepath}")

# 使用示例
if __name__ == "__main__":
    # 参数说明:
    # port: 串口名称，None为自动检测
    # filename_prefix: 输出文件名前缀
    # max_records: 最大记录数，达到后自动停止
    read_and_process_serial(
        port='COM20',
        out_folder = 'antenna_equipment_angle_data', 
        filename_prefix="data_30",  # 自定义文件名
        max_records=10  # 收集10组数据后停止
    )
    
    # 示例：如何加载数据
    # data = load_numpy_data("iq_data/my_iq_data.npy")
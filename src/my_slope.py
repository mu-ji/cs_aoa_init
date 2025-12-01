import numpy as np

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

def unwrapped_phase(i_result, q_result):
    phase = np.arctan2(q_result, i_result)
    return np.unwrap(phase)

def cal_slope(phase):
    diff = np.diff(phase)
    # 计算 Z-score
    mean_val = np.mean(diff)
    std_val = np.std(diff)
    # 3σ 过滤
    mask = np.abs(diff - mean_val) < 3 * std_val
    
    mean_diff = np.mean(diff[mask])
    return mean_diff/1e6

def cal_distance(slope):
    LIGHTSPEED = 299792458
    return (LIGHTSPEED / (4 * np.pi))*slope




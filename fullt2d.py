import numpy as np
import os
import json

from blf2dic import blf2dic_main

class SIMULATIONINFO:
    def __init__(self) -> None:
        pass

class FULLDATAINFO:
    
    def __init__(self) -> None:
        self.data = blf2dic_main()
        self.composed_data = self.compose_data(self.data)

    @staticmethod
    def compose_data(data:dict)->dict:
        return FULLDATAINFO.interpolation(data)

    @staticmethod
    def interpolation(data :dict) -> dict:
        # 读取仿真参数
        config_path = os.path.join(os.path.dirname(__file__), "vehicle_param.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        sim_param = config["simulation_param"]
        simulation_time = sim_param["simulation_time"]
        simulation_cycle = sim_param["simulation_cycle"]
        
        # 生成仿真时间数组
        sim_times = np.arange(0, simulation_time + simulation_cycle, simulation_cycle)
        
        signals = data["signals"]
        composed_data = {}
        
        for signal_name, series in signals.items():
            t = np.array(series["t"])
            v = np.array(series["v"])
            
            # 对每个sim_time进行插值
            interpolated_values = []
            for sim_t in sim_times:
                if sim_t <= t[0]:
                    interpolated_values.append(float(v[0]))
                elif sim_t >= t[-1]:
                    interpolated_values.append(float(v[-1]))
                else:
                    # 找到第一个大于等于sim_t的索引
                    idx = np.searchsorted(t, sim_t)
                    if t[idx] == sim_t:
                        interpolated_values.append(float(v[idx]))
                    else:
                        # 线性插值
                        t1, t2 = t[idx-1], t[idx]
                        v1, v2 = v[idx-1], v[idx]
                        interpolated_value = v1 + (v2 - v1) * (sim_t - t1) / (t2 - t1)
                        interpolated_values.append(float(interpolated_value))
            
            composed_data[signal_name] = interpolated_values
        
        return composed_data
    

if __name__ == "__main__":
    test = FULLDATAINFO()
    print(test.composed_data)

import numpy as np
import os
import json
from typing import Any, Dict, Set

from blf2dic import blf2dic_main


def _to_real(x: Any) -> float:
    """解码值可能是数值，或 cantools 的 NamedSignalValue（枚举，需取 .value）。"""
    if isinstance(x, (bool, np.bool_)):
        return float(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    val = getattr(x, "value", None)
    if val is not None and not isinstance(x, (str, bytes)):
        return float(val)
    return float(x)


def _signal_keys_for_bus(signals_cfg: Dict[str, Any], bus: str) -> Set[str]:
    block = signals_cfg.get(bus)
    if not isinstance(block, dict):
        return set()
    if "signals" in block and isinstance(block.get("signals"), dict):
        return {str(k) for k in block["signals"].keys()}
    return {str(k) for k in block.keys() if k != "dbc"}


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
        # zoh：与多数 CAN 工具（含 TSMaster）一致——帧间保持上一值；linear：两点间线性（易与实车显示不一致）
        interp = str(config.get("signal_interpolation", "zoh")).strip().lower()
        if interp in ("linear", "lerp"):
            use_zoh_default = False
        else:
            use_zoh_default = True
        
        # 生成仿真时间数组
        sim_times = np.arange(0, simulation_time + simulation_cycle, simulation_cycle)
        
        signals = data["signals"]
        composed_data = {}

        signals_cfg = config.get("signals") or {}
        discrete_signal_names = _signal_keys_for_bus(signals_cfg, "Tcs_signal")

        for signal_name, series in signals.items():
            t = np.array(series["t"], dtype=float)
            v = np.array(series["v"], dtype=object)
            if len(t) == 0:
                composed_data[signal_name] = [0.0] * len(sim_times)
                continue

            use_hold = use_zoh_default or (signal_name in discrete_signal_names)

            interpolated_values = []
            for sim_t in sim_times:
                if use_hold:
                    if sim_t <= t[0]:
                        interpolated_values.append(_to_real(v[0]))
                    elif sim_t >= t[-1]:
                        interpolated_values.append(_to_real(v[-1]))
                    else:
                        idx = int(np.searchsorted(t, sim_t, side="right") - 1)
                        if idx < 0:
                            idx = 0
                        interpolated_values.append(_to_real(v[idx]))
                else:
                    if sim_t <= t[0]:
                        interpolated_values.append(_to_real(v[0]))
                    elif sim_t >= t[-1]:
                        interpolated_values.append(_to_real(v[-1]))
                    else:
                        idx = int(np.searchsorted(t, sim_t))
                        if t[idx] == sim_t:
                            interpolated_values.append(_to_real(v[idx]))
                        else:
                            t1, t2 = float(t[idx - 1]), float(t[idx])
                            v1, v2 = _to_real(v[idx - 1]), _to_real(v[idx])
                            interpolated_value = v1 + (v2 - v1) * (sim_t - t1) / (t2 - t1)
                            interpolated_values.append(float(interpolated_value))

            composed_data[signal_name] = interpolated_values
        
        return composed_data
    

if __name__ == "__main__":
    test = FULLDATAINFO()
    print(test.composed_data)

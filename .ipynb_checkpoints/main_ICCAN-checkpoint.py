import json
import os
import re
import numpy as np

from fullt2d import FULLDATAINFO
from iccan_bokeh_report import save_iccan_interactive_html


class ICCAN:
    def __init__(self) -> None:
        self.intpl_data = FULLDATAINFO().composed_data
        self.full_data = self.cal_data(self.intpl_data)
        self.generate_csv(self.full_data)
        self.generate_plot(self.full_data)

    @staticmethod
    def getblf_time() -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(script_dir, "vehicle_param.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        blf_path = cfg.get("blf_path", "")
        if not blf_path:
            raise FileNotFoundError("vehicle_param.json 未配置 blf_path")
        base = os.path.basename(blf_path)
        name, _ext = os.path.splitext(base)
        m = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", name)
        if not m:
            raise ValueError(f"从BLF文件名中无法解析时间戳: {name}")
        t = m.group(1)  # 默认的
        return f"{t[0:4]}{t[5:7]}{t[8:10]}_{t[11:13]}{t[14:16]}{t[17:19]}"

    @staticmethod
    def cal_data(composed_data:dict|None)->dict:
        if composed_data is None:
            return {}

        # 读取参数
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(script_dir, "vehicle_param.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        sim_cycle = cfg.get("simulation_param", {}).get("simulation_cycle", 0.01)
        vp = cfg.get("vehicle_param", {})
        m = float(vp.get("vehicle_mass", 2300))
        g = float(vp.get("g_acc", vp.get("gravity", 9.81)))
        r = float(vp.get("wheel_radius", 0.32))
        J = float(vp.get("J_whe", 1.2))
        motor2wheel_ratio = float(vp.get("motor2wheel_ratio", 10.685))
        motor_eff = float(vp.get("motor_eff", 0.95))
        ff_cfg = vp.get("Cal_Ff", {})
        ff_cof = float(ff_cfg.get("cof", 0.0))
        ff_const = float(ff_cfg.get("const", 0.0))

        full_data = {}

        # 统一长度为最短列长度 minus 1（因为diff产生序列少1个点）
        n = min(len(v) for v in composed_data.values() if isinstance(v, (list, tuple, np.ndarray)))
        if n <= 1:
            return composed_data

        # 时间轴
        full_data["time"] = [round(i * sim_cycle, 6) for i in range(n - 1)]

        # 原始数据截断
        for k, v in composed_data.items():
            arr = np.array(v, dtype=float)
            if k == "MCU_Torque":
                full_data[k] = (arr[: n-1]).tolist()
            else:
                full_data[k] = arr[:n - 1].tolist()

        # 轮速角加速度计算
        wheel_keys = ["WhlSpdFL", "WhlSpdFR", "WhlSpdRL", "WhlSpdRR"]
        ang_acc = {}
        for key in wheel_keys:
            if key in composed_data:
                w = np.array(composed_data[key], dtype=float)
                da = np.diff(w) / sim_cycle
                ang_acc_key = f"{key}_ang_acc"
                ang_acc[ang_acc_key] = da.tolist()
                full_data[ang_acc_key] = da.tolist()

        # 驱动轮、非驱动轮力与纵向力
        torque = np.array(full_data.get("MCU_Torque", [0.0] * (n - 1)), dtype=float)

        F_fl = np.zeros(n - 1)
        F_fr = np.zeros(n - 1)
        F_rl = np.zeros(n - 1)
        F_rr = np.zeros(n - 1)

        for wheel, initial in [("RL", "WhlSpdRL_ang_acc"), ("RR", "WhlSpdRR_ang_acc")]:
            if initial in full_data:
                dom = np.array(full_data[initial], dtype=float)
                td = torque * motor2wheel_ratio * motor_eff/ 2.0
                Ff = ff_cof * np.abs(full_data.get("MCU_Torque" ,[])) + ff_const
                Fx = (td - J * dom - Ff ) / r
                if wheel == "RL":
                    F_rl = Fx
                else:
                    F_rr = Fx

        for wheel, initial in [("FL", "WhlSpdFL_ang_acc"), ("FR", "WhlSpdFR_ang_acc")]:
            if initial in full_data:
                dom = np.array(full_data[initial], dtype=float)#轮加速度
                Ff = ff_cof * np.abs(full_data.get("MCU_Torque" ,[])) + ff_const
                Fx = -(J * dom / r) - Ff
                if wheel == "FL":
                    F_fl = Fx
                else:
                    F_fr = Fx

        full_data["Fx_RL"] = F_rl.tolist()
        full_data["Fx_RR"] = F_rr.tolist()
        full_data["Fx_FL"] = F_fl.tolist()
        full_data["Fx_FR"] = F_fr.tolist()

        total_Fx = F_rl + F_rr + F_fl + F_fr
        full_data["Fx_by_tyre"] = total_Fx.tolist()

        # 参考 LongAcc 计算纵向力：F = m * g * a
        if "LongAcc" in full_data:
            long_acc = np.array(full_data.get("LongAcc", []), dtype=float)
            Fx_by_acc = m * g * long_acc
            full_data["Fx_by_acc"] = Fx_by_acc.tolist()
        else:
            full_data["Fx_by_acc"] = [0.0] * (n - 1)
        return full_data

    @staticmethod
    def generate_csv(full_data:dict):
        timestamp = ICCAN.getblf_time()
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", timestamp)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "blf_result.csv")

        keys = sorted(full_data.keys())
        if not keys:
            return

        n = min(len(full_data[k]) for k in keys if isinstance(full_data[k], (list, tuple)) and len(full_data[k]) > 0)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for i in range(n):
                row = []
                for k in keys:
                    v = full_data.get(k, [])
                    if i < len(v):
                        row.append(str(v[i]))
                    else:
                        row.append("")
                f.write(",".join(row) + "\n")

        print(f"CSV生成: {csv_path}")

    @staticmethod
    def generate_plot(full_data:dict):
        timestamp = ICCAN.getblf_time()
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", timestamp)
        os.makedirs(out_dir, exist_ok=True)
        html_path = os.path.join(out_dir, "blf_result.html")
        save_iccan_interactive_html(full_data, html_path, title="ICCAN 仿真结果")
        print(f"HTML生成: {html_path}")


# 外部接口
def mainshow():
    ICCAN()


if __name__ == "__main__":
    mainshow()    
    
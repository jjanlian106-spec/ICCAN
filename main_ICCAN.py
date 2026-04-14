import json
import os
import re
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column , row

from fullt2d import FULLDATAINFO


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
        t = np.array(full_data.get("time", []), dtype=float)

        # VehSpd-fig1
        VehSpd = np.array(full_data["VehSpd"], dtype=float)
        WhlSpdRL = np.array(full_data["WhlSpdRL"], dtype=float)
        WhlSpdRR = np.array(full_data["WhlSpdRR"], dtype=float)
        WhlSpdFL = np.array(full_data["WhlSpdFL"], dtype=float)
        WhlSpdFR = np.array(full_data["WhlSpdFR"], dtype=float)      
        source = ColumnDataSource(data={"time": t, "VehSpd": VehSpd,
                                        "WhlSpdRL": WhlSpdRL,
                                        "WhlSpdRR": WhlSpdRR,
                                        "WhlSpdFL": WhlSpdFL,
                                        "WhlSpdFR": WhlSpdFR,
                                        })
        fig1 = figure(
            x_axis_label="time (s)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=600,
            height=280,
        )
        fig1.line("time", "VehSpd", source=source, line_width=2, color="blue" , legend_label="VehSpd")
        fig1.line("time", "WhlSpdRL", source=source, line_width=2, color="red" , legend_label="WhlRLSpd")
        fig1.line("time", "WhlSpdRR", source=source, line_width=2, color="green" , legend_label="WhlRRSpd")
        fig1.line("time", "WhlSpdFL", source=source, line_width=2, color="orange" , legend_label="WhlFLSpd")
        fig1.line("time", "WhlSpdFR", source=source, line_width=2, color="black" , legend_label="WhlFRSpd")
        fig1.add_tools(HoverTool(tooltips=[("time", "@time{0.000}"), ("VehSpd", "@value{0.000}")]))

        # LongAcc-fig2
        LongAcc = np.array(full_data["LongAcc"], dtype=float)
        source = ColumnDataSource(data={"time": t, "value": LongAcc})
        fig2 = figure(
            x_axis_label="time (s)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=600,
            height=280,
        )
        fig2.line("time", "value", source=source, line_width=2, color="green" , legend_label="LongAcc")
        fig2.add_tools(HoverTool(tooltips=[("time", "@time{0.000}"), ("LongAcc", "@value{0.000}")]))

        # MCU_Torque-fig3
        MCU_Torque = np.array(full_data["MCU_Torque"], dtype=float)
        VCU2MCU_MotorTorque_cmd = np.array(full_data["VCU2MCU_MotorTorque_cmd"], dtype=float)
        source = ColumnDataSource(data={"time": t, "MCU_Torque": MCU_Torque,
                                        "VCU2MCU_MotorTorque_cmd": VCU2MCU_MotorTorque_cmd})
        fig3 = figure(
            x_axis_label="time (s)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=600,
            height=280,
        )
        fig3.line("time", "MCU_Torque", source=source, line_width=2, color="red" , legend_label="MCU_Torque")
        fig3.line("time", "VCU2MCU_MotorTorque_cmd", source=source, line_width=2, color="blue" , legend_label="VCU2MCU_MotorTorque_cmd")
        fig3.add_tools(HoverTool(tooltips=[("time", "@time{0.000}"), ("MCU_Torque", "@value{0.000}")]))

        # MCU_MotorSpeed-fig4
        MCU_MotorSpeed = np.array(full_data["MCU_MotorSpeed"], dtype=float)
        source = ColumnDataSource(data={"time": t, "value": MCU_MotorSpeed})
        fig4 = figure(
            x_axis_label="time (s)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=600,
            height=280,
        )
        fig4.line("time", "value", source=source, line_width=2, color="orange", legend_label="MCU_MotorSpeed")
        fig4.add_tools(HoverTool(tooltips=[("time", "@time{0.000}"), ("MCU_MotorSpeed", "@value{0.000}")]))

        # Fx_by_tyre
        Fx_by_tyre = np.array(full_data["Fx_by_tyre"], dtype=float)
        Fx_by_acc = np.array(full_data["Fx_by_acc"], dtype=float)
        source1 = ColumnDataSource(data={"time": t, "value": Fx_by_tyre})
        source2 = ColumnDataSource(data={"time": t, "value": Fx_by_acc})
        fig5 = figure(
            x_axis_label="time (s)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=600,
            height=280,
        )
        fig5.line("time", "value", source=source1, line_width=2, color="red", legend_label="Fx_by_tyre")
        fig5.line("time", "value", source=source2, line_width=2, color="blue", legend_label="Fx_by_acc")
        fig5.add_tools(HoverTool(tooltips=[("time", "@time{0.000}"), ("Fx_by_tyre", "@value{0.000}")]))

        #GasPdl
        GasPdlPsnRaw = np.array(full_data["GasPdlPsnRaw"], dtype=float)
        source = ColumnDataSource(data={"time": t, "value": GasPdlPsnRaw})
        fig6 = figure(
            x_axis_label="time (s)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=600,
            height=280,
        )
        fig6.line("time", "value", source=source, line_width=2, color="black" , legend_label="GasPdlPsnRaw")
        fig6.add_tools(HoverTool(tooltips=[("time", "@time{0.000}"), ("GasPdlPsnRaw", "@value{0.000}")]))  
        
        #WheelSpdPulse
        RRWheelSpdPulse = np.array(full_data["RRWheelSpdPulse"], dtype=float)
        RLWheelSpdPulse = np.array(full_data["RLWheelSpdPulse"], dtype=float)
        FLWheelSpdPulse = np.array(full_data["FLWheelSpdPulse"], dtype=float)
        source = ColumnDataSource(data={"time": t, "RRWheelSpdPulse": RRWheelSpdPulse ,
                                        "RLWheelSpdPulse": RLWheelSpdPulse,
                                        "FLWheelSpdPulse": FLWheelSpdPulse,
                                        })
        fig7 = figure(
            x_axis_label="time (s)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=600,
            height=280,
        )
        fig7.line("time", "RRWheelSpdPulse", source=source, line_width=2, color="black" , legend_label="RRWheelSpdPulse")
        fig7.line("time", "RLWheelSpdPulse", source=source, line_width=2, color="blue" , legend_label="RLWheelSpdPulse")
        fig7.line("time", "FLWheelSpdPulse", source=source, line_width=2, color="red" , legend_label="FLWheelSpdPulse")
        fig7.add_tools(HoverTool(tooltips=[("time", "@time{0.000}"), ("RRWheelSpdPulse", "@value{0.000}")])) 
        
        
        #可视化排布
        r1 = row(fig1 , fig2)
        r2 = row(fig3 , fig4)
        r3 = row(fig5 , fig6)
        r4 = row(fig7)
        layout = column(r1 , r2 , r3 , r4)
        output_file(html_path, title="ICCAN 仿真结果")
        save(layout)
        print(f"HTML生成: {html_path}")


# 外部接口
def mainshow():
    ICCAN()


if __name__ == "__main__":
    mainshow()    
    
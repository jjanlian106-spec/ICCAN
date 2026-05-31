# -*- coding: utf-8 -*-
"""
ICCAN Bokeh HTML: one time-index slider, vertical spans on all plots, tcs_enable_code bit table (CustomJS).
Add more code maps (disable/fault/kickdown) in this module later.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, DataTable, Div, HoverTool, Slider, Span, TableColumn
from bokeh.plotting import figure, output_file, save

# tcs_enable_code bit0..bit8; \\u strings keep Chinese stable under flaky UTF-8 saves.
TCS_ENABLE_CODE_BIT_DEFS: List[Tuple[str, str]] = [
    ("bit0", "\u56db\u95e8\u4e24\u76d6\u672a\u5168\u90e8\u5173\u95ed"),
    ("bit1", "\u6321\u4f4d\u4e0d\u662fD\u6863"),
    ("bit2", "\u5b89\u5168\u5e26\u672a\u7cfb\u7d27"),
    ("bit3", "\u5236\u52a8\u8e29\u677f\u8e29\u4e0b"),
    ("bit4", "\u4fa7\u5411\u52a0\u901f\u5ea6\u5927\u4e8e2m/ss"),
    ("bit5", "\u6a2a\u6446\u89d2\u901f\u5ea6\u5927\u4e8e10deg/s"),
    ("bit6", "\u65b9\u5411\u76d8\u8f6c\u89d2\u5927\u4e8e40deg"),
    ("bit7", "\u65b9\u5411\u76d8\u8f6c\u89d2\u901f\u5ea6\u5927\u4e8e100deg"),
    ("bit8", "\u52a0\u901f\u8e29\u677f\u5f00\u5ea6\u5c0f\u4e8e30%"),
]


def _as_float1d(full_data: Dict[str, Any], key: str, n: int) -> np.ndarray:
    if key not in full_data:
        return np.zeros(n, dtype=float)
    return np.asarray(full_data[key], dtype=float)[:n]


def _condition_texts_only(defs: List[Tuple[str, str]]) -> List[str]:
    return [ds for _, ds in defs]


def _enable_table_columns(code: float, defs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Three columns: bit name, condition text, active_bit as '0'/'1'."""
    c = int(round(code))
    bit_col: List[str] = []
    cond_col: List[str] = []
    act_col: List[str] = []
    for i, (bk, ds) in enumerate(defs):
        on = (c >> i) & 1
        bit_col.append(bk)
        cond_col.append(ds)
        act_col.append("1" if on else "0")
    return {"bit": bit_col, "cond": cond_col, "active_bit": act_col}


def _header_html(t_val: float) -> str:
    return f"<b>t</b> = {t_val:.4f} s"


def save_iccan_interactive_html(full_data: Dict[str, Any], html_path: str, title: str = "ICCAN \u4eff\u771f\u7ed3\u679c") -> None:
    """Interactive HTML: top slider, light time header, compact enable table under fig_en."""
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)

    t = np.asarray(full_data["time"], dtype=float)
    n = len(t)
    if n == 0:
        raise ValueError("full_data['time'] \u4e3a\u7a7a")

    x_range = None

    def _fig(**kw: Any) -> Any:
        nonlocal x_range
        if x_range is None:
            f = figure(**kw)
            x_range = f.x_range
            return f
        return figure(x_range=x_range, **kw)

    VehSpd = _as_float1d(full_data, "VehSpd", n)
    WhlSpdRL = _as_float1d(full_data, "WhlSpdRL", n)
    WhlSpdRR = _as_float1d(full_data, "WhlSpdRR", n)
    WhlSpdFL = _as_float1d(full_data, "WhlSpdFL", n)
    WhlSpdFR = _as_float1d(full_data, "WhlSpdFR", n)
    src1 = ColumnDataSource(
        data={"time": t, "VehSpd": VehSpd, "WhlSpdRL": WhlSpdRL, "WhlSpdRR": WhlSpdRR, "WhlSpdFL": WhlSpdFL, "WhlSpdFR": WhlSpdFR}
    )
    fig1 = _fig(x_axis_label="time (s)", tools="pan,wheel_zoom,box_zoom,reset,save", width=600, height=280)
    fig1.line("time", "VehSpd", source=src1, line_width=2, color="blue", legend_label="VehSpd")
    fig1.line("time", "WhlSpdRL", source=src1, line_width=2, color="red", legend_label="WhlRLSpd")
    fig1.line("time", "WhlSpdRR", source=src1, line_width=2, color="green", legend_label="WhlRRSpd")
    fig1.line("time", "WhlSpdFL", source=src1, line_width=2, color="orange", legend_label="WhlFLSpd")
    fig1.line("time", "WhlSpdFR", source=src1, line_width=2, color="black", legend_label="WhlFRSpd")
    fig1.yaxis.axis_label = "speed (km/h)"
    fig1.legend.click_policy = "hide"
    fig1.add_tools(HoverTool(tooltips=[("time", "@time{0.000}")]))

    LongAcc = _as_float1d(full_data, "LongAcc", n)
    src2 = ColumnDataSource(data={"time": t, "value": LongAcc})
    fig2 = _fig(x_axis_label="time (s)", tools="pan,wheel_zoom,box_zoom,reset,save", width=600, height=280)
    fig2.line("time", "value", source=src2, line_width=2, color="green", legend_label="LongAcc")
    fig2.yaxis.axis_label = "acceleration (m/s^2)"
    fig2.legend.click_policy = "hide"
    fig2.add_tools(HoverTool(tooltips=[("time", "@time{0.000}")]))

    MCU_Torque = _as_float1d(full_data, "MCU_Torque", n)
    VCU2MCU_MotorTorque_cmd = _as_float1d(full_data, "VCU2MCU_MotorTorque_cmd", n)
    src3 = ColumnDataSource(data={"time": t, "MCU_Torque": MCU_Torque, "VCU2MCU_MotorTorque_cmd": VCU2MCU_MotorTorque_cmd})
    fig3 = _fig(x_axis_label="time (s)", tools="pan,wheel_zoom,box_zoom,reset,save", width=600, height=280)
    fig3.line("time", "MCU_Torque", source=src3, line_width=2, color="red", legend_label="MCU_Torque")
    fig3.line("time", "VCU2MCU_MotorTorque_cmd", source=src3, line_width=2, color="blue", legend_label="VCU2MCU_MotorTorque_cmd")
    fig3.yaxis.axis_label = "torque (Nm)"
    fig3.legend.click_policy = "hide"
    fig3.add_tools(HoverTool(tooltips=[("time", "@time{0.000}")]))

    MCU_MotorSpeed = _as_float1d(full_data, "MCU_MotorSpeed", n)
    src4 = ColumnDataSource(data={"time": t, "value": MCU_MotorSpeed})
    fig4 = _fig(x_axis_label="time (s)", tools="pan,wheel_zoom,box_zoom,reset,save", width=600, height=280)
    fig4.line("time", "value", source=src4, line_width=2, color="orange", legend_label="MCU_MotorSpeed")
    fig4.yaxis.axis_label = "motor speed (rpm)"
    fig4.legend.click_policy = "hide"
    fig4.add_tools(HoverTool(tooltips=[("time", "@time{0.000}")]))

    Fx_by_tyre = _as_float1d(full_data, "Fx_by_tyre", n)
    Fx_by_acc = _as_float1d(full_data, "Fx_by_acc", n)
    src5a = ColumnDataSource(data={"time": t, "value": Fx_by_tyre})
    src5b = ColumnDataSource(data={"time": t, "value": Fx_by_acc})
    fig5 = _fig(x_axis_label="time (s)", tools="pan,wheel_zoom,box_zoom,reset,save", width=600, height=280)
    fig5.line("time", "value", source=src5a, line_width=2, color="red", legend_label="Fx_by_tyre")
    fig5.line("time", "value", source=src5b, line_width=2, color="blue", legend_label="Fx_by_acc")
    fig5.yaxis.axis_label = "force (N)"
    fig5.legend.click_policy = "hide"
    fig5.add_tools(HoverTool(tooltips=[("time", "@time{0.000}")]))

    GasPdlPsnRaw = _as_float1d(full_data, "GasPdlPsnRaw", n)
    src6 = ColumnDataSource(data={"time": t, "value": GasPdlPsnRaw})
    fig6 = _fig(x_axis_label="time (s)", tools="pan,wheel_zoom,box_zoom,reset,save", width=600, height=280)
    fig6.line("time", "value", source=src6, line_width=2, color="black", legend_label="GasPdlPsnRaw")
    fig6.yaxis.axis_label = "pedal position (%)"
    fig6.legend.click_policy = "hide"
    fig6.add_tools(HoverTool(tooltips=[("time", "@time{0.000}")]))

    RRWheelSpdPulse = _as_float1d(full_data, "RRWheelSpdPulse", n)
    RLWheelSpdPulse = _as_float1d(full_data, "RLWheelSpdPulse", n)
    FLWheelSpdPulse = _as_float1d(full_data, "FLWheelSpdPulse", n)
    src7 = ColumnDataSource(
        data={"time": t, "RRWheelSpdPulse": RRWheelSpdPulse, "RLWheelSpdPulse": RLWheelSpdPulse, "FLWheelSpdPulse": FLWheelSpdPulse}
    )
    fig7 = _fig(x_axis_label="time (s)", tools="pan,wheel_zoom,box_zoom,reset,save", width=600, height=280)
    fig7.line("time", "RRWheelSpdPulse", source=src7, line_width=2, color="black", legend_label="RRWheelSpdPulse")
    fig7.line("time", "RLWheelSpdPulse", source=src7, line_width=2, color="blue", legend_label="RLWheelSpdPulse")
    fig7.line("time", "FLWheelSpdPulse", source=src7, line_width=2, color="red", legend_label="FLWheelSpdPulse")
    fig7.yaxis.axis_label = "pulse count"
    fig7.legend.click_policy = "hide"
    fig7.add_tools(HoverTool(tooltips=[("time", "@time{0.000}")]))

    enable = _as_float1d(full_data, "tcs_enable_code", n)
    src_en = ColumnDataSource(data={"time": t, "enable": enable})
    fig_en = _fig(x_axis_label="time (s)", tools="pan,wheel_zoom,box_zoom,reset,save", width=600, height=260)
    fig_en.step("time", "enable", source=src_en, mode="after", line_width=2, color="darkviolet", legend_label="tcs_enable_code")
    fig_en.yaxis.axis_label = "code"
    fig_en.legend.click_policy = "hide"
    fig_en.add_tools(HoverTool(tooltips=[("time", "@time{0.000}"), ("enable", "@enable{0}")], mode="vline"))

    figures: List[Any] = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig_en]
    spans: List[Span] = []
    x0 = float(t[0])
    for f in figures:
        sp = Span(location=x0, dimension="height", line_color="crimson", line_width=1.5, line_alpha=0.85)
        f.add_layout(sp)
        spans.append(sp)

    plot_w = 600
    n_bits = len(TCS_ENABLE_CODE_BIT_DEFS)
    idx_src = ColumnDataSource(data={"time": t, "enable": enable})
    table_src = ColumnDataSource(data=_enable_table_columns(float(enable[0]), TCS_ENABLE_CODE_BIT_DEFS))
    header_div = Div(text=_header_html(float(t[0])), width=plot_w * 2 + 40, height=28)

    _col_cond = "\u4f7f\u80fd\u6761\u4ef6"
    columns = [
        TableColumn(field="bit", title="bit", width=72),
        TableColumn(field="cond", title=_col_cond, width=360),
        TableColumn(field="active_bit", title="active_bit", width=88),
    ]
    data_table = DataTable(
        source=table_src,
        columns=columns,
        width=plot_w,
        height=248,
        index_position=None,
        row_height=26,
    )

    _slider_title = (
        "\u65f6\u95f4\u7d22\u5f15\uff08\u62d6\u52a8\uff1a\u540c\u6b65\u5404\u56fe\u7ad6\u7ebf\u4e0e\u4e0b\u65b9\u4f7f\u80fd\u6761\u4ef6\u8868\uff09"
    )
    slider = Slider(
        start=0,
        end=max(0, n - 1),
        value=0,
        step=1,
        title=_slider_title,
        width=plot_w * 2 + 40,
    )

    bit_key_json = json.dumps([bk for bk, _ in TCS_ENABLE_CODE_BIT_DEFS], ensure_ascii=False)
    cond_json = json.dumps(_condition_texts_only(TCS_ENABLE_CODE_BIT_DEFS), ensure_ascii=False)

    span_args = {f"sp{i}": sp for i, sp in enumerate(spans)}
    span_lines = "\n".join(f"    sp{i}.location = x;" for i in range(len(spans)))
    cb_code = f"""
    const i = Math.round(cb_obj.value);
    const tm = idx_src.data['time'];
    const en = idx_src.data['enable'];
    if (i < 0 || i >= tm.length) return;
    const x = tm[i];
    let code = Math.round(en[i]);
    const nBits = {n_bits};
    const mask = (1 << nBits) - 1;
    code = code & mask;

    const bitLabels = {bit_key_json};
    const condTexts = {cond_json};

{span_lines}

    const actCol = [];
    for (let b = 0; b < bitLabels.length; b++) {{
        const on = (code >> b) & 1;
        actCol.push(String(on));
    }}
    tbl.data = {{
        bit: bitLabels,
        cond: condTexts,
        active_bit: actCol
    }};
    tbl.change.emit();

    hdr.text = "<b>t</b> = " + x.toFixed(4) + " s";
    """

    slider_cb = CustomJS(args=dict(idx_src=idx_src, tbl=table_src, hdr=header_div, **span_args), code=cb_code)
    slider.js_on_change("value", slider_cb)

    top_bar = column(slider, header_div, sizing_mode="stretch_width")
    _cap = "<small>tcs_enable_code</small>"
    enable_caption = Div(text=_cap, width=plot_w, height=20)
    enable_col = column(fig_en, enable_caption, data_table, width=plot_w)
    r1 = row(fig1, fig2)
    r2 = row(fig3, fig4)
    r3 = row(fig5, fig6)
    r4 = row(fig7, enable_col)
    layout = column(top_bar, r1, r2, r3, r4, sizing_mode="stretch_width")

    output_file(html_path, title=title)
    save(layout)

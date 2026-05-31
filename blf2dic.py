
from __future__ import annotations

import argparse
import json
import os
import pickle
import can
import cantools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

def _default_out_path(blf_path: str, fmt: str) -> str:
    base, _ext = os.path.splitext(blf_path)
    return f"{base}_dic.{fmt}"

def _load_vehicle_param_json(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(base_dir: str, p: str) -> str:
    # 允许 json 里写相对路径（相对脚本目录）
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


def _default_dbc_ref_dir(script_dir: str) -> str:
    p = os.path.join(script_dir, "dbc_ref")
    if not os.path.isdir(p):
        raise FileNotFoundError(f"找不到 dbc_ref 目录：{p}")
    return p


def _pick_dbc_by_bus(dbc_dir: str, bus: str) -> str:
    bus_u = bus.upper()
    candidates: List[str] = []
    for name in os.listdir(dbc_dir):
        if not name.lower().endswith(".dbc"):
            continue
        if bus_u in name.upper():
            candidates.append(os.path.join(dbc_dir, name))
    if not candidates:
        raise FileNotFoundError(f"在 {dbc_dir} 未找到包含 {bus} 的 DBC")
    # 对 CCAN 固定优先 AEBS 新版
    if bus_u == "CCAN":
        aebs = [p for p in candidates if "AEBS" in os.path.basename(p).upper()]
        if aebs:
            candidates = aebs
    candidates.sort()
    return os.path.abspath(candidates[0])


def _build_frame_id_sets(db: Any, wanted_signals: Sequence[str]) -> set[int]:
    wanted = set(wanted_signals)
    frame_ids: set[int] = set()
    for msg in db.messages:
        for sig in msg.signals:
            if sig.name in wanted:
                frame_ids.add(int(msg.frame_id))
                break
    return frame_ids


def _coerce_channel_map(raw: Any) -> Dict[str, Union[int, List[int], Tuple[int, ...]]]:
    """vehicle_param.json 里 blf_channel_map：{ \"CCAN\": 1, \"EVCAN\": [0,2] }"""
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Union[int, List[int], Tuple[int, ...]]] = {}
    for bus, ch in raw.items():
        b = str(bus)
        if isinstance(ch, (list, tuple)):
            out[b] = [int(x) for x in ch]
        elif ch is None:
            continue
        else:
            out[b] = int(ch)
    return out


def _channel_allows(msg: can.Message, bus: str, channel_by_bus: Dict[str, Union[int, List[int], Tuple[int, ...]]]) -> bool:
    """未配置该 bus 的通道 = 不限制；配置了则必须匹配（多通道录制时避免误用其它总线 DBC 解码）。"""
    if bus not in channel_by_bus:
        return True
    allowed = channel_by_bus[bus]
    if msg.channel is None:
        return False
    if isinstance(allowed, (list, tuple)):
        return int(msg.channel) in [int(x) for x in allowed]
    return int(msg.channel) == int(allowed)


def blf_to_dict_by_config(
    blf_path: str,
    bus_to_dbc_path: Dict[str, str],
    signals_by_bus: Dict[str, Sequence[str]],
    channel_by_bus: Optional[Dict[str, Union[int, List[int], Tuple[int, ...]]]] = None,
) -> Dict[str, Any]:
    if not os.path.exists(blf_path):
        raise FileNotFoundError(f"找不到 BLF 文件：{blf_path}")

    # 加载每个 bus 的 DBC，并为该 bus 的信号构建 frame_id set
    bus_to_db: Dict[str, Any] = {}
    bus_to_frame_ids: Dict[str, set[int]] = {}
    for bus, dbc_path in bus_to_dbc_path.items():
        if not os.path.exists(dbc_path):
            raise FileNotFoundError(f"找不到 {bus} 的 DBC：{dbc_path}")
        db = cantools.database.load_file(dbc_path)
        bus_to_db[bus] = db
        wanted = list(signals_by_bus.get(bus, []))
        bus_to_frame_ids[bus] = _build_frame_id_sets(db, wanted) if wanted else set()

    signals: Dict[str, Dict[str, list]] = {}
    # 快速过滤：signal -> (bus, display_name?) 这里先只做集合，显示名留给 meta
    bus_signal_sets: Dict[str, set[str]] = {bus: set(sigs) for bus, sigs in signals_by_bus.items()}
    t_min: Optional[float] = None
    t_max: Optional[float] = None

    ch_map = channel_by_bus or {}

    reader = can.BLFReader(blf_path)
    for msg in reader:
        t = float(msg.timestamp)
        if t_min is None or t < t_min:
            t_min = t
        if t_max is None or t > t_max:
            t_max = t

        arb_id = int(msg.arbitration_id)
        msg_data = bytes(msg.data)

        # 对每个 bus：先用 frame_id set 过滤，再解码
        for bus, frame_ids in bus_to_frame_ids.items():
            if not _channel_allows(msg, bus, ch_map):
                continue
            if arb_id not in frame_ids:
                continue
            db = bus_to_db[bus]
            try:
                decoded = db.decode_message(arb_id, msg_data)
            except Exception:
                continue
            wanted_set = bus_signal_sets.get(bus, set())
            for sig_name, sig_value in decoded.items():
                if sig_name not in wanted_set:
                    continue
                # 输出 key 仍用“信号名本身”，由你在 json 中确保不冲突；若以后冲突再改成 bus 前缀即可
                bucket = signals.get(sig_name)
                if bucket is None:
                    bucket = {"t": [], "v": []}
                    signals[sig_name] = bucket
                bucket["t"].append(t)
                bucket["v"].append(sig_value)

    meta = {
        "source_blf": os.path.abspath(blf_path),
        "decode_strategy": "by_vehicle_param_json_signals",
        "t_min": t_min,
        "t_max": t_max,
        "num_signals": len(signals),
        "bus_to_dbc": {k: os.path.abspath(v) for k, v in bus_to_dbc_path.items()},
        "signals_by_bus": {k: list(v) for k, v in signals_by_bus.items()},
        "blf_channel_map": ch_map if ch_map else None,
    }
    return {"meta": meta, "signals": signals}

def unify_data_time(data: Dict[str, Any], time_base: str = "first_sample") -> Dict[str, Any]:
    """
    时间戳归一化：
    - time_base=\"first_sample\"（默认）：t0 = 各已解码信号首点时间的最小值（原行为）
    - time_base=\"file_start\"：t0 = 整条 BLF 最早时间 meta[\"t_min\"]，便于与 TSMaster 等按录制起点对齐
    - 将所有信号时间减去 t0，写入 meta[\"t0\"]，并更新 meta[\"t_min\"]/meta[\"t_max\"]
    """
    if not isinstance(data, dict):
        raise TypeError("data 必须是 dict")
    signals = data.get("signals")
    if not isinstance(signals, dict):
        raise ValueError("data 缺少 signals 字段或类型不正确")

    meta = data.get("meta")
    meta_dict = meta if isinstance(meta, dict) else {}

    t0: Optional[float] = None
    if time_base == "file_start" and meta_dict.get("t_min") is not None:
        t0 = float(meta_dict["t_min"])
    else:
        for _sig_name, series in signals.items():
            if not isinstance(series, dict):
                continue
            t_list = series.get("t")
            if not isinstance(t_list, list) or len(t_list) == 0:
                continue
            first_t = float(t_list[0])
            if t0 is None or first_t < t0:
                t0 = first_t

    if t0 is None:
        raise ValueError("signals 中没有找到任何可用时间戳（t 列表为空或缺失）")

    for _sig_name, series in signals.items():
        if not isinstance(series, dict):
            continue
        t_list = series.get("t")
        if not isinstance(t_list, list) or len(t_list) == 0:
            continue
        series["t"] = [float(t) - t0 for t in t_list]

    if isinstance(meta, dict):
        meta["t0"] = t0
        meta["time_base"] = time_base
        if "t_min" in meta and meta["t_min"] is not None:
            meta["t_min"] = float(meta["t_min"]) - t0
        if "t_max" in meta and meta["t_max"] is not None:
            meta["t_max"] = float(meta["t_max"]) - t0

    return data


def blf2dic_main() -> dict:
    parser = argparse.ArgumentParser(description="BLF -> dict（按时间戳对应）")
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认：脚本同目录 vehicle_param.json）",
    )
    parser.add_argument("--blf", default=None, help="可选覆盖：输入 BLF 文件路径（否则从 config 的 blf_path 读取）")
    parser.add_argument("--out", default=None, help="输出文件路径（默认：同目录/同名生成 *_dic.pkl 或 *_dic.json）")
    parser.add_argument("--format", choices=["pkl", "json"], default="pkl", help="输出格式")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if args.config else os.path.join(script_dir, "vehicle_param.json")
    cfg = _load_vehicle_param_json(config_path)

    blf_path = args.blf if args.blf else cfg.get("blf_path")
    if not blf_path:
        raise ValueError("未提供 BLF 路径：请在 vehicle_param.json 里设置 blf_path，或用 --blf 覆盖。")

    out_path = args.out if args.out else _default_out_path(blf_path, args.format)

    # signals: JSON 里配置为 {BUS: {signal_name: display_name, ...}, ...}
    cfg_signals = cfg.get("signals") or {}
    if not isinstance(cfg_signals, dict) or not cfg_signals:
        raise ValueError("vehicle_param.json 的 signals 必须是非空 dict（例如包含 EVCAN/CCAN 分组）。")

    signals_by_bus: Dict[str, List[str]] = {}
    bus_dbc_inline: Dict[str, str] = {}
    display_name_map: Dict[str, str] = {}
    for bus, mapping in cfg_signals.items():
        if not isinstance(mapping, dict):
            raise ValueError(f"signals[{bus}] 必须是 dict")

        # 兼容两种写法：
        # 1) signals.EVCAN = {sig: display, ...}
        # 2) signals.EVCAN = {"dbc": "...", "signals": {sig: display, ...}}
        if "signals" in mapping and isinstance(mapping.get("signals"), dict):
            sig_map = mapping["signals"]
            dbc_inline = mapping.get("dbc")
            if isinstance(dbc_inline, str) and dbc_inline.strip():
                bus_dbc_inline[str(bus)] = dbc_inline.strip()
        else:
            sig_map = mapping

        signals_by_bus[str(bus)] = [str(k) for k in sig_map.keys()]
        for sig_name, display in sig_map.items():
            display_name_map[str(sig_name)] = str(display)

    # 优先使用顶层 dbc_path：键与 signals 分组一致（如 EVCAN/CCAN/Tcs_signal）
    dbc_map = cfg.get("dbc_path") or cfg.get("dbc_map") or cfg.get("dbc_files") or cfg.get("dbc")
    bus_to_dbc_path: Dict[str, str] = {}
    for bus in signals_by_bus.keys():
        try:
            if bus in bus_dbc_inline:
                bus_to_dbc_path[bus] = _resolve_path(script_dir, bus_dbc_inline[bus])
                continue
            if isinstance(dbc_map, dict) and isinstance(dbc_map.get(bus), str) and dbc_map.get(bus).strip():
                bus_to_dbc_path[bus] = _resolve_path(script_dir, dbc_map[bus].strip())
                continue
            dbc_dir = cfg.get("dbc_dir")
            dbc_dir_path = (
                _resolve_path(script_dir, dbc_dir)
                if isinstance(dbc_dir, str) and dbc_dir.strip()
                else _default_dbc_ref_dir(script_dir)
            )
            bus_to_dbc_path[bus] = _pick_dbc_by_bus(dbc_dir_path, bus)
        except Exception:
            # 没有找到对应 DBC 的分组直接忽略，不再读取
            continue

    if not bus_to_dbc_path:
        raise ValueError("未能为任何 signals 分组找到对应 DBC，请检查 vehicle_param.json 中的 DBC 配置或 dbc_ref 目录。")

    channel_raw = cfg.get("blf_channel_map")
    if channel_raw is None and isinstance(cfg.get("blf_decode"), dict):
        channel_raw = cfg["blf_decode"].get("channel_map")
    channel_by_bus = _coerce_channel_map(channel_raw)

    data = blf_to_dict_by_config(
        blf_path=blf_path,
        bus_to_dbc_path=bus_to_dbc_path,
        signals_by_bus=signals_by_bus,
        channel_by_bus=channel_by_bus or None,
    )
    data["meta"]["signals_config"] = cfg_signals
    data["meta"]["display_names"] = display_name_map

    time_base = str(cfg.get("blf_time_base", "first_sample"))
    if time_base not in ("first_sample", "file_start"):
        time_base = "first_sample"
    unify_data_time(data, time_base=time_base)

    return data


if __name__ == "__main__":
    blf2dic_main()

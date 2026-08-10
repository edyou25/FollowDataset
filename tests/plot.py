#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


LABEL_TO_VALUE = {
    "guide": 0,
    "tether": 1,
}


def load_interaction_data(jsonl_path: Path):
    times = []
    labels = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[Warning] 第 {line_no} 行不是有效 JSON，已跳过: {exc}")
                continue

            label = str(record.get("interaction_label", "")).strip().lower()
            if label not in LABEL_TO_VALUE:
                print(
                    f"[Warning] 第 {line_no} 行 interaction_label={label!r}，"
                    "不是 guide/tether，已跳过"
                )
                continue

            # 优先使用 relative_time，没有时使用 stamp
            time_value = record.get("relative_time")
            if time_value is None:
                time_value = record.get("stamp")

            if time_value is None:
                print(f"[Warning] 第 {line_no} 行没有 relative_time 或 stamp，已跳过")
                continue

            try:
                time_value = float(time_value)
            except (TypeError, ValueError):
                print(f"[Warning] 第 {line_no} 行时间无效，已跳过")
                continue

            times.append(time_value)
            labels.append(label)

    if not times:
        raise RuntimeError("没有读取到有效的 guide/tether 数据。")

    times = np.asarray(times, dtype=float)
    labels = np.asarray(labels)

    # 按时间排序
    order = np.argsort(times)
    times = times[order]
    labels = labels[order]

    # 无论输入的是 stamp 还是 relative_time，都让时间严格从 0 开始
    times = times - times[0]

    values = np.asarray([LABEL_TO_VALUE[label] for label in labels], dtype=int)

    return times, values, labels


def find_switches(times, labels):
    """返回状态切换的位置。"""
    switch_indices = np.where(labels[1:] != labels[:-1])[0] + 1
    return [
        {
            "time": float(times[index]),
            "from": str(labels[index - 1]),
            "to": str(labels[index]),
        }
        for index in switch_indices
    ]


def plot_interaction_curve(
    times,
    values,
    labels,
    output_path: Path,
    title: str,
    show: bool,
):
    fig, ax = plt.subplots(figsize=(12, 3.8))

    # where="post" 表示当前状态保持到下一个采样时刻
    ax.step(
        times,
        values,
        where="post",
        linewidth=2.2,
    )

    switches = find_switches(times, labels)

    # 标记状态切换时刻
    for switch in switches:
        ax.axvline(
            switch["time"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
        )

        ax.text(
            switch["time"],
            0.5,
            f'{switch["from"]} → {switch["to"]}\n'
            f'{switch["time"]:.2f} s',
            rotation=90,
            ha="right",
            va="center",
            fontsize=8,
        )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Guide", "Tether"])
    ax.set_ylim(-0.25, 1.25)

    ax.set_xlim(left=0)
    ax.set_xlabel("Time (s)")

    # 主刻度每 1 秒一个，并显示数字
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # 次刻度每 0.1 秒一个
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    # 主、次网格
    ax.grid(
        True,
        which="major",
        axis="x",
        linestyle="-",
        linewidth=0.8,
        alpha=0.55,
    )

    ax.grid(
        True,
        which="minor",
        axis="x",
        linestyle=":",
        linewidth=0.5,
        alpha=0.35,
    )

    # 刻度线设置
    ax.tick_params(
        axis="x",
        which="major",
        length=6,
        labelsize=9,
    )

    ax.tick_params(
        axis="x",
        which="minor",
        length=3,
    )

    ax.set_ylabel("Interaction state")
    ax.set_title(title)

    ax.grid(
        True,
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=0.5,
    )

    fig.tight_layout()
    fig.show()
    # fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"[Saved] 曲线已保存到: {output_path}")
    print(f"[Info] 总帧数: {len(times)}")
    print(f"[Info] 总时长: {times[-1]:.3f} s")
    print(f"[Info] 状态切换次数: {len(switches)}")

    for index, switch in enumerate(switches, start=1):
        print(
            f"  Switch {index}: "
            f'{switch["from"]} -> {switch["to"]}, '
            f't={switch["time"]:.3f} s'
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="绘制 guide/tether 随时间变化的阶梯曲线"
    )
    parser.add_argument(
        "-input",
        default="/home/yyf/IROS2026/FollowDataset/tests/artifacts/2026-08-02-18-18-16.bag/planning_details.jsonl",
        type=Path,
        help="输入 JSONL 文件，每行包含一个 JSON 对象",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("interaction_state_timeline.png"),
        help="输出图片路径，默认 interaction_state_timeline.png",
    )
    parser.add_argument(
        "--title",
        default="Guide–Tether Interaction State",
        help="图片标题",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存图片，不弹出窗口",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    times, values, labels = load_interaction_data(args.input)

    plot_interaction_curve(
        times=times,
        values=values,
        labels=labels,
        output_path=args.output,
        title=args.title,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
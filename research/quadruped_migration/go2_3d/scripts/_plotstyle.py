"""共享 matplotlib 中文字体设置（修 CJK 字形缺失 → 方块）。

matplotlib 3.9 默认 DejaVu Sans 无 CJK 字形，且未把系统 .ttc 集合按 CJK 名索引。
这里显式 addfont 系统已装的 CJK 字体文件，并设 axes.unicode_minus=False（正确负号）。
所有 E3D 出图脚本在 import pyplot 后调用 use_cjk() 一次。
"""
from __future__ import annotations

import matplotlib
import matplotlib.font_manager as fm

_CJK_FILES = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",       # WenQuanYi Micro Hei（实测可注册）
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


def use_cjk() -> str | None:
    name = None
    for path in _CJK_FILES:
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            break
        except Exception:
            continue
    if name:
        matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return name

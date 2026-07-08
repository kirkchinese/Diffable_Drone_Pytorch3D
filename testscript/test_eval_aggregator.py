"""WP0.2 自检：聚合器 load_metrics 优先读 metrics.json，回退到 stdout-regex。

验证 batch_eval_all / multi_seed_eval 的 load_metrics：
  1. 有 metrics.json 时，取 JSON 值（即便日志里是另一套数）——去脆弱 stdout-regex。
  2. 无 metrics.json 时，回退到 parse_summary 正则（兼容旧 DONE run）。
纯 stdlib/CPU，不需 GPU/torch。
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_eval_all
import multi_seed_eval

# metrics.json 与日志故意给不同的数，用于判定 load_metrics 取了哪个来源
JSON_METRICS = {"SR": 83.13, "SR_n": "27/32", "RR": 93.75, "RR_n": "30/32",
                "CFR": 88.13, "CFR_n": "28/32", "avg_speed": 1.09,
                "final_target_dist": 0.42, "progress": 97.5}
LOG_TEXT = (
    "  严格成功率 SR: 5/10 (50.00%)\n"
    "  抵达率: 6/10 (60.00%)\n"
    "  全程无碰撞率: 7/10 (70.00%)\n"
)


def _check_json_primary(mod):
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(JSON_METRICS, f)
        log = os.path.join(d, "eval.log")
        with open(log, "w") as f:
            f.write(LOG_TEXT)  # 冲突来源：若走 regex 会得 50/60/70
        m = mod.load_metrics(d, log)
    assert abs(m["SR"] - 83.13) < 1e-6, f"{mod.__name__}: 未取 JSON 的 SR，得 {m.get('SR')}"
    assert abs(m["RR"] - 93.75) < 1e-6, f"{mod.__name__}: 未取 JSON 的 RR"
    assert abs(m["CFR"] - 88.13) < 1e-6, f"{mod.__name__}: 未取 JSON 的 CFR"
    assert abs(m["avg_speed"] - 1.09) < 1e-6, f"{mod.__name__}: 缺 avg_speed"
    print(f"  [PASS] {mod.__name__}.load_metrics 优先读 metrics.json (SR=83.13 而非 log 的 50)")


def _check_regex_fallback(mod):
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "eval.log")
        with open(log, "w") as f:
            f.write(LOG_TEXT)  # 无 metrics.json → 回退 regex
        m = mod.load_metrics(d, log)
    assert m and abs(float(m["SR"]) - 50.0) < 1e-6, \
        f"{mod.__name__}: 无 JSON 时未回退 regex，得 {m.get('SR') if m else None}"
    print(f"  [PASS] {mod.__name__}.load_metrics 回退 stdout-regex (SR=50)")


def main():
    failed = 0
    for mod in (batch_eval_all, multi_seed_eval):
        for check in (_check_json_primary, _check_regex_fallback):
            try:
                check(mod)
            except AssertionError as e:
                failed += 1
                print(f"  [FAIL] {e}")
    print("=== EVAL AGGREGATOR SELF-CHECK "
          + ("PASSED ===" if failed == 0 else f"{failed} FAILED ==="))
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)

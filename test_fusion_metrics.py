"""Testes do summarize_fusion_log (sem hardware)."""
from __future__ import annotations
import csv
import tempfile
from pathlib import Path

from fusion.metrics import summarize_fusion_log, print_summary
from fusion.logging import CSV_HEADERS


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fusion.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADERS)
            def row(hand, source, valid, err, cal):
                r = [""] * len(CSV_HEADERS)
                r[0] = "1.0"; r[1] = hand; r[2] = source; r[3] = "0.8"
                r[4] = valid; r[5] = "10.0"; r[19] = str(err); r[20] = str(cal)
                r[21] = "5.0"
                return r
            w.writerow(row("right", "fused", "1", 2.0, 12.0))
            w.writerow(row("right", "fused", "1", 4.0, 14.0))
            w.writerow(row("left", "imu_only", "1", "", ""))
            w.writerow(row("left", "none", "0", "", ""))
        s = summarize_fusion_log(path)
        assert s["total_rows"] == 4
        assert set(s["hands"]) == {"right", "left"}
        assert s["per_hand"]["right"]["fused_count"] == 2
        assert abs(s["per_hand"]["right"]["err_fused_vis"]["mean_deg"] - 3.0) < 1e-9
        assert s["per_hand"]["left"]["err_fused_vis"]["n"] == 0
        print_summary(s)
        print("METRICS_OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

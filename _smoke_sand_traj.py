"""Smoke test do Programa 2: treino sintetico + tempo real headless.

Passos:
1. sand_traj_treino.main (--selftest, 2 epocas) -> checkpoints temporarios;
2. sand_traj_tempo_real.main (--headless-test) com log CSV;
3. valida que o log tem >= 3 previsoes.

Rode dentro do terminal do VS Code (o g.Pype exige IDE suportada).
"""
import csv
import os
import sys


def run(argv_modulename, module_name, argv):
    sys.argv = argv
    __import__(module_name).main()


print("== [1/3] treino sintetico ==", flush=True)
run("sand_traj_treino.main", "sand_traj_treino", [
    "sand_traj_treino.py", "--selftest",
    "--epochs", "2", "--selftest-samples", "32",
    "--model-out", "_smoke_sand_traj_model.pt",
    "--results-out", "_smoke_sand_traj_results.csv",
    "--seed", "5",
])

print("== [2/3] tempo real headless (4 s) ==", flush=True)
run("sand_traj_tempo_real.main", "sand_traj_tempo_real", [
    "sand_traj_tempo_real.py", "--model", "_smoke_sand_traj_model.pt",
    "--headless-test", "4", "--interval", "0.5",
    "--log-csv", "_smoke_sand_traj_log.csv",
])

print("== [3/3] validacao ==", flush=True)
if not os.path.exists("_smoke_sand_traj_log.csv"):
    raise SystemExit("ERRO: log de previsoes nao foi criado.")
with open("_smoke_sand_traj_log.csv", "r", encoding="utf-8") as handle:
    rows = list(csv.reader(handle))
assert len(rows) >= 4, f"esperado >= 3 previsoes, achei {len(rows) - 1}"
header, *data = rows
assert header[1] == "movimento_m" and header[2] == "extensao_m"
for row in data:
    assert float(row[1]) >= 0.0 and int(row[3]) == 30
print(f"SMOKE_SAND_TRAJ_OK: {len(data)} previsoes validas | "
      f"movimento medio {sum(float(r[1]) for r in data) / len(data):.4f} m")
sys.exit(0)
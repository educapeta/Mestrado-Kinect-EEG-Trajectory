import csv, math, statistics
from pathlib import Path
p = Path(r"c:\Users\elgut\OneDrive\Desktop\Mestrado_Projeto\Mestradopy\overlay_deviation_log.csv")
rows = list(csv.DictReader(p.open(encoding='utf-8')))
print("nrows", len(rows))
def col(i, s): return f"n{i}_{s}"
def med(v):
    v=[x for x in v if math.isfinite(x)]
    return statistics.median(v) if v else float('nan')
def mean(v):
    v=[x for x in v if math.isfinite(x)]
    return statistics.mean(v) if v else float('nan')
# per node median (robust) + p90
for i in range(21):
    vals=[]
    for r in rows:
        try: vals.append(float(r[col(i,'dev_px')]))
        except: pass
    vals=[v for v in vals if math.isfinite(v)]
    vals.sort()
    if vals:
        p90=vals[int(0.9*(len(vals)-1))]
        print(f"n{i:02d} med {statistics.median(vals):6.1f} p90 {p90:8.1f} max {vals[-1]:10.1f}")
# segments by index: user said: start center static, then edges, then fast near aux
segs = {"inicio_centro_parado": rows[:60], "meio_bordas": rows[150:260], "fim_rapido_aux": rows[260:340], "resto": rows[340:]}
for label, sl in segs.items():
    d9=[]; dx=[]; dy=[]; vel=[]; b=[]
    for r in sl:
        try:
            d9.append(float(r[col(9,'dev_px')])); dx.append(float(r[col(9,'dx')])); dy.append(float(r[col(9,'dy')]))
            v=r['vel_px_s']; vel.append(float(v) if v else float('nan'))
            b.append(float(r['borda_0c_1b']))
        except: pass
    print(f"== {label} n={len(sl)} n9_med={med(d9):.1f} n9_p90={sorted(d9)[int(0.9*(len(d9)-1))]:.1f} dx_med={med(dx):+.1f} dy_med={med(dy):+.1f} vel_med={med(vel):.0f} borda_med={med(b):.2f}")
# slow vs fast on n9 (clip absurd >500)
pairs=[]
for r in rows:
    try:
        v=float(r['vel_px_s']); d=float(r[col(9,'dev_px')])
        if math.isfinite(v) and math.isfinite(d) and d<500: pairs.append((v,d))
    except: pass
slow=[d for v,d in pairs if v<50]; fast=[d for v,d in pairs if v>200]
print("n9 clip500: slow n",len(slow),"med",med(slow),"| fast n",len(fast),"med",med(fast))
# palm dx/dy sign stability: fraction with dx>0
for label, sl in segs.items():
    s=[]
    for r in sl:
        try:
            s.append((float(r[col(9,'dx')]), float(r[col(9,'dy')])))
        except: pass
    if s:
        print(label, "frac dx>0", sum(1 for x,y in s if x>0)/len(s), "frac dy>0", sum(1 for x,y in s if y>0)/len(s))
# depth usado vs bruto
for label, sl in [("ini",rows[:40]),("fim",rows[-40:])]:
    zu=[float(r['z_usado_m']) for r in sl]; zb=[float(r['z_bruto_m']) for r in sl]
    print(label, "z_usado med", med(zu), "z_bruto med", med(zb))

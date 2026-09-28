"""Testes da bateria estatistica (`estat_bci.py`) -- sem hardware, sem dados.

Valida o que pode ser conferido a mao ou por definicao:
1. nivel de acaso EXATO (ancoras 10->9/10 e 20->15/20; critico justo);
2. p-valor binomial e poder;
3. kappa / acuracia balanceada;
4. PCC (perfeito, invertido, constante -> NaN, NaN no meio);
5. baseline do alvo medio (a armadilha do PCC alto sem decodificacao);
6. IC por Fisher e bootstrap;
7. permutacao do PCC: caso informativo (pareamento importa) x caso nulo, com
   p = (1 + #{nulo >= obs}) / (1 + n_perm) conferido no minimo;
8. comparacao pareada (teste de sinais) + tamanho de efeito;
9. d de Cohen / d_z / g de Hedges conferidos por formula;
10. BH-FDR e Holm com exemplos calculados a mao;
11. ITR (bits/trial e bits/min) com valores exatos;
12. cluster-based permutation acha um efeito plantado e NAO acha efeito nenhum;
13. comparacao de grupos independentes e de metodos (Friedman + Holm);
14. determinismo (mesma semente -> mesmo resultado).
"""
import numpy as np

import estat_bci as eb

casos = 0

# ---------------------------------------------------------------- 1) acaso
info10 = eb.acerto_minimo(10, n_classes=2)
assert info10["acertos_criticos"] == 9, info10          # 9/10 = 90% (mao)
info20 = eb.acerto_minimo(20, n_classes=2)
assert info20["acertos_criticos"] == 15, info20         # 15/20 = 75% (mao)
assert info20["acerto_critico_pct"] == 75.0, info20
# critico JUSTO: no ponto, p <= alfa; um acerto a menos, p > alfa
for n_trials in (10, 20, 30, 60, 100, 200):
    info = eb.acerto_minimo(n_trials, n_classes=2)
    critico = info["acertos_criticos"]
    assert eb.p_valor_acerto(critico, n_trials, 2) <= eb.ALFA, (n_trials, info)
    assert eb.p_valor_acerto(critico - 1, n_trials, 2) > eb.ALFA, (n_trials, info)
# mais trials -> critico percentual menor (a distribuicao afina)
percentuais = [eb.acerto_minimo(n, n_classes=2)["acerto_critico_pct"]
               for n in (10, 20, 50, 100, 400)]
assert all(a > b for a, b in zip(percentuais, percentuais[1:])), percentuais
# 3 classes: base 1/3 e critico acima disso
info3 = eb.acerto_minimo(60, n_classes=3)
assert 33.0 < info3["acerto_critico_pct"] < 60.0, info3
# a aproximacao normal fica perto do exato com N grande (e longe com N pequeno)
aprox = 100.0 * eb.acaso_aproximado(400, n_classes=2)
exato = eb.acerto_minimo(400, n_classes=2)["acerto_critico_pct"]
assert abs(aprox - exato) < 1.0, (aprox, exato)
assert abs(100.0 * eb.acaso_aproximado(10, n_classes=2) - 90.0) > 1.0
casos += 1

# ------------------------------------------------- 2) p-valor binomial/poder
p15 = eb.p_valor_acerto(15, 20, 2)
assert abs(p15 - 0.0207) < 0.001, p15                   # mao: 0,0207
assert eb.p_valor_acerto(14, 20, 2) > eb.ALFA
assert eb.p_valor_acerto(5, 20, 2) > 0.99               # 5/20 esta' na media
assert eb.p_valor_acerto(10, 10, 2) < 0.01              # 10/10
# poder: com acuracia = acaso (0,5) o poder e' exatamente alfa (~0,0207 aqui);
# com 75% de acerto verdadeiro e SO' 20 trials o poder fica em ~0,62 -- e' o
# numero que explica por que 20 trials nao bastam para achar efeito moderado.
poder_nulo = eb.poder_binomial(20, 0.50, n_classes=2)
assert 0.01 < poder_nulo < 0.03, poder_nulo
poder_75 = eb.poder_binomial(20, 0.75, n_classes=2)
assert 0.50 < poder_75 < 0.75, poder_75
assert eb.poder_binomial(20, 0.90, n_classes=2) > 0.90
assert eb.poder_binomial(100, 0.59, n_classes=2) > 0.40
casos += 1

# ----------------------------------------------------- 3) kappa/acerto balanceado
assert abs(eb.kappa_de_cohen([[10, 0], [0, 10]]) - 1.0) < 1e-12
assert abs(eb.kappa_de_cohen([[5, 5], [5, 5]])) < 1e-12          # acaso puro
assert abs(eb.acuracia_balanceada([[9, 1], [5, 5]]) - 0.70) < 1e-12
assert np.isnan(eb.kappa_de_cohen([[0, 0], [0, 0]]))
casos += 1

# ------------------------------------------------------------------- 4) PCC
x = np.linspace(-1.0, 1.0, 21)
assert abs(eb.pcc(x, x) - 1.0) < 1e-12
assert abs(eb.pcc(x, -x) + 1.0) < 1e-12
assert np.isnan(eb.pcc(x, np.full(21, 3.0)))            # constante -> indefinido
com_nan = x.copy()
com_nan[5] = np.nan
assert abs(eb.pcc(com_nan, x) - 1.0) < 1e-12            # NaN ignorado
assert np.isnan(eb.pcc([1.0, np.nan], [1.0, 2.0]))      # < 3 pontos
rng = np.random.default_rng(7)
previsao = rng.normal(size=(6, 30, 3))
alvo = previsao + rng.normal(scale=0.5, size=(6, 30, 3))
por_trial = eb.pcc_por_trial(previsao, alvo)
assert por_trial.shape == (6, 3), por_trial.shape
assert abs(por_trial[0, 0]
           - np.corrcoef(previsao[0, :, 0], alvo[0, :, 0])[0, 1]) < 1e-12
assert (por_trial > 0.5).all(), por_trial
casos += 1

# ------------------------------------------- 5) baseline do alvo medio (armadilha)
base = np.sin(np.linspace(0.0, 3.0, 30))
trajetorias = [np.concatenate([base + rng.normal(scale=0.05, size=30),
                               np.zeros(0)])[:, None] for _ in range(12)]
alvo_parecido = np.stack(trajetorias, axis=0)           # (12, 30, 1)
assert alvo_parecido.shape == (12, 30, 1), alvo_parecido.shape
baseline = eb.pcc_alvo_medio(alvo_parecido)
assert baseline.shape == (12, 1), baseline.shape
assert np.nanmin(baseline) > 0.95, np.nanmin(baseline)  # chutar a media "acerta"
casos += 1

# ------------------------------------------------------------ 6) IC de PCC
lo, hi = eb.ic_pcc_fisher(0.6, 30)
assert lo < 0.6 < hi, (lo, hi)
lo2, hi2 = eb.ic_pcc_fisher(0.6, 300)
assert (hi2 - lo2) < (hi - lo), (lo, hi, lo2, hi2)       # mais pontos, IC menor
assert np.isnan(eb.ic_pcc_fisher(0.6, 3)[0])
amostras = rng.normal(loc=0.7, scale=0.1, size=200)
ic = eb.ic_media_bootstrap(amostras, n_boot=2000, rng=rng)
assert ic["lo"][0] < ic["media"][0] < ic["hi"][0], ic
assert abs(ic["media"][0] - 0.7) < 0.05, ic
casos += 1

# -------------------------------------------- 7) permutacao do PCC (pareamento)
# Caso INFORMATIVO: cada trial tem uma trajetoria BEM distinta e o modelo a
# reproduz -> se o pareamento for destruido (permutacao), o PCC desaba.
rng = np.random.default_rng(11)
n_trials, n_pontos = 24, 40
t_base = np.linspace(0.0, 1.0, n_pontos)
alvos = np.array([np.stack([np.sin((3.0 + 0.7 * i) * t_base),
                            np.cos((2.0 + 0.5 * i) * t_base),
                            0.5 * np.sin((1.5 + 0.4 * i) * t_base)], axis=1)
                  for i in range(n_trials)])              # (24, 40, 3)
prev = alvos + rng.normal(scale=0.02, size=alvos.shape)
inform = eb.teste_permutacao_pcc(prev, alvos, n_perm=400, rng=rng)
assert inform["observado_medio"] > 0.9, inform["observado_medio"]
assert inform["p_agregado"] <= 1.0 / 401.0 + 1e-12, inform["p_agregado"]
assert (inform["p_por_eixo"] <= 1.0 / 401.0 + 1e-12).all(), inform["p_por_eixo"]
# o nulo (pareamento embaralhado) fica MUITO abaixo do observado
assert float(np.nanmax(inform["nulo_media"])) < 0.5, inform["nulo_media"]
# p = (1 + #{nulo >= obs}) / (1 + n_perm): conferido pela contagem direta
contagem = np.mean(inform["nulo_media"] >= inform["observado_medio"])
assert abs(contagem - inform["p_agregado"]) < 0.05, (contagem,
                                                     inform["p_agregado"])
# Caso NULO: previsao independente do alvo -> observado da ordem do nulo
prev_nulo = rng.normal(size=alvos.shape)
nulo = eb.teste_permutacao_pcc(prev_nulo, alvos, n_perm=400, rng=rng)
assert nulo["p_agregado"] > 0.10, nulo["p_agregado"]
# permutacao DENTRO de blocos (2 sessoes) continua detectando o efeito
blocos = np.array([0] * (n_trials // 2) + [1] * (n_trials - n_trials // 2))
em_blocos = eb.teste_permutacao_pcc(prev, alvos, n_perm=300, bloco=blocos,
                                    rng=rng)
assert em_blocos["p_agregado"] < 0.05, em_blocos["p_agregado"]
casos += 1

# ------------------------------------------------- 8) comparacao pareada/efeito
rng = np.random.default_rng(13)
modelo = rng.normal(loc=0.70, scale=0.10, size=80)
baseline = modelo - 0.12 + rng.normal(scale=0.05, size=80)
teste = eb.comparacao_pareada(modelo, baseline, n_perm=2000, rng=rng)
assert teste["p_perm"] <= 1.0 / 2001.0 + 1e-12, teste["p_perm"]
assert teste["p_wilcoxon"] < 0.001, teste["p_wilcoxon"]
assert teste["d_z"] > 1.0, teste["d_z"]
assert teste["ic"][0] > 0.05, teste["ic"]
igual = eb.comparacao_pareada(modelo, modelo, n_perm=500, rng=rng)
assert np.isnan(igual["d_z"]) or abs(igual["d_z"]) < 1e-9, igual["d_z"]
casos += 1

# --------------------------------------------------- 9) tamanhos de efeito
a = np.array([2.0, 3.0, 4.0, 5.0])
b = np.array([1.0, 2.0, 3.0, 4.0])
d = eb.cohens_d(a, b)
assert abs(d - 0.7746) < 0.001, d                       # mao: 1/sqrt(1,6667)
assert abs(eb.hedges_g(a, b) - d * (1.0 - 3.0 / 23.0)) < 1e-12
assert np.isnan(eb.cohens_dz_pareado(a, b))             # variancia das dif = 0
c = a + np.array([0.1, -0.1, 0.2, -0.2])
dz = eb.cohens_dz_pareado(c, b)
assert dz > 5.0, dz
casos += 1

# --------------------------------------------------- 10) FDR (BH) e Holm
# Exemplo CANONICO (Benjamini & Hochberg 1995, m=15): limiares i*0,05/15 ->
# 0,0033 / 0,0067 / 0,01 / 0,0133 / 0,02 ... => so' os QUATRO primeiros passam.
canonico = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
            0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]
bh = eb.bh_fdr(canonico, alfa=0.05)
assert bh["rejeita"].tolist() == [True] * 4 + [False] * 11, bh["rejeita"]
assert bh["n_rejeitados"] == 4, bh["n_rejeitados"]
assert bh["p_ajustado"][3] < 0.05 < bh["p_ajustado"][4], bh["p_ajustado"][:5]
# O BH toma o MAIOR i que satisfaz p_(i) <= i*alfa/m -- entao um p grande pode
# "puxar" os menores junto (aqui os quatro viram ajustado 0,04 <= 0,05). E' uma
# propriedade do metodo, nao um bug: o doc usa este caso como exemplo.
quatro = eb.bh_fdr([0.01, 0.02, 0.03, 0.04], alfa=0.05)
assert quatro["rejeita"].all(), quatro
puxado = eb.bh_fdr([0.01, 0.04, 0.04, 0.04], alfa=0.05)
assert puxado["rejeita"].all(), puxado
assert abs(puxado["p_ajustado"][0] - 0.04) < 1e-12, puxado["p_ajustado"]
assert eb.bh_fdr([0.01, 0.04, 0.04, 0.04], alfa=0.03)["n_rejeitados"] == 0
assert eb.bh_fdr([np.nan, 0.01, np.nan])["n_rejeitados"] == 1
assert (bh["p_ajustado"] >= np.asarray(canonico) - 1e-12).all()
# Holm (FWER) no MESMO exemplo: p_(i) <= alfa/(m-i+1) -> 0,05/15, 0,05/14,
# 0,05/13 passam (0,0001 / 0,0004 / 0,0019) e 0,0095 > 0,05/12 PAROU -> 3 rejeicoes.
# Note: "mais conservador" no Holm quer dizer criterio/limiar por passo, nao
# necessariamente MENOS rejeicoes que o BH neste exemplo.
h_canonico = eb.holm(canonico, alfa=0.05)
assert h_canonico["n_rejeitados"] == 3, h_canonico["n_rejeitados"]
assert abs(h_canonico["p_ajustado"][0] - 15 * 0.0001) < 1e-12
assert abs(h_canonico["p_ajustado"][1] - 14 * 0.0004) < 1e-12
assert abs(h_canonico["p_ajustado"][3] - 12 * 0.0095) < 1e-12
h = eb.holm([0.01, 0.03, 0.04], alfa=0.05)
assert abs(h["p_ajustado"][0] - 0.03) < 1e-12, h         # 0,01*3
assert abs(h["p_ajustado"][1] - 0.06) < 1e-12, h         # 0,03*2
assert abs(h["p_ajustado"][2] - 0.06) < 1e-12, h         # monotono (0,06)
assert h["rejeita"].tolist() == [True, False, False], h
casos += 1

# ------------------------------------------------------------------- 11) ITR
assert abs(eb.itr_bits(1.0, 2) - 1.0) < 1e-12
assert abs(eb.itr_bits(1.0, 4) - 2.0) < 1e-12
assert eb.itr_bits(0.5, 2) == 0.0
assert eb.itr_bits(0.25, 4) == 0.0
assert abs(eb.itr_bits(0.75, 2) - 0.1887) < 0.001       # mao: 1 - 0,3113 - 0,5
assert abs(eb.itr_por_minuto(1.0, 2, 2.0) - 30.0) < 1e-12
assert abs(eb.itr_por_minuto(1.0, 2, 2.0, corrigido=False) - 60.0) < 1e-12
assert eb.itr_por_minuto(0.5, 2, 2.0) == 0.0
casos += 1

# --------------------------------------------------- 12) cluster-based (1D)
rng = np.random.default_rng(17)
n_unidades, n_pontos = 16, 100
cond_a = rng.normal(size=(n_unidades, n_pontos))
cond_b = rng.normal(size=(n_unidades, n_pontos))
cond_b[:, 40:60] -= 1.0                     # efeito plantado (t ~ +2,8 por ponto)
achado = eb.teste_cluster_permutacao(cond_a, cond_b, limiar=2.0, n_perm=400,
                                     rng=rng)
assert achado["p_menor"] < 0.05, achado["clusters"]
# o cluster do efeito e' POSITIVO (a - b > 0), comeca depois do ponto 30 e
# termina antes do 70: a borda exata depende do desvio padrao DE CADA PONTO
# (por isso nao se afirma latencia a partir de cluster -- ver o docstring).
do_efeito = [c for c in achado["clusters"] if c["soma"] > 0]
assert do_efeito, achado["clusters"]
maior = max(do_efeito, key=lambda c: c["n_pontos"])
# SOBREPOSICAO com a janela plantada (40..59) -- nao igualdade: o |t| de cada
# ponto depende do desvio padrao DAQUELE ponto, entao a borda do cluster varia
# (e' exatamente por isso que cluster NAO serve para afirmar latencia/onset).
assert maior["inicio"] < 60 and maior["fim"] >= 45, maior
assert maior["n_pontos"] >= 6, maior
assert maior["p"] < 0.05, maior
assert achado["p_menor"] >= 1.0 / (achado["n_perm"] + 1.0), achado["p_menor"]
# sem efeito: nenhum cluster sobrevive a correcao
nada = eb.teste_cluster_permutacao(rng.normal(size=(n_unidades, n_pontos)),
                                   rng.normal(size=(n_unidades, n_pontos)),
                                   limiar=2.0, n_perm=400, rng=rng)
assert np.isnan(nada["p_menor"]) or nada["p_menor"] > 0.05, nada["p_menor"]
# e a estatistica de cluster e' a do MAIOR cluster de cada permutacao (nula > 0)
assert (nada["nulo_max"] >= 0.0).all()
casos += 1

# ------------------------------------------- 13) grupos independentes e metodos
rng = np.random.default_rng(19)
grupo_a = rng.normal(loc=1.00, scale=0.30, size=60)
grupo_b = rng.normal(loc=0.60, scale=0.30, size=60)
comparado = eb.comparar_grupos(grupo_a, grupo_b, pareado=False, n_perm=2000,
                               rng=rng)
assert comparado["p_perm"] < 0.001, comparado["p_perm"]
assert comparado["d"] > 1.0, comparado["d"]
assert abs(comparado["diferenca_media"] - 0.40) < 0.15, comparado
# Calibracao do teste: com dados EXATAMENTE intercambiaveis, a taxa de falso
# positivo tem de ficar perto de alfa (40 comparacoes -> esperado ~2; aqui o
# limite 5 e' folgado de proposito, para nao depender de sorte de uma semente).
falsos = 0
for _ in range(40):
    juntos = rng.normal(loc=0.5, scale=1.0, size=120)
    res = eb.comparar_grupos(juntos[:60], juntos[60:], pareado=False,
                             n_perm=500, rng=rng)
    falsos += int(res["p_perm"] < eb.ALFA)
assert falsos <= 5, falsos
# metodos medidos nos MESMOS 8 sujeitos: A > B > C (com ruido pequeno)
sujeitos = rng.normal(loc=0.60, scale=0.05, size=8)
tabela = {"A": sujeitos + 0.10 + rng.normal(scale=0.01, size=8),
          "B": sujeitos + 0.02 + rng.normal(scale=0.01, size=8),
          "C": sujeitos - 0.05 + rng.normal(scale=0.01, size=8)}
resultado = eb.comparar_metodos(tabela, rng=rng)
assert resultado["vitorias"]["A"] == 8, resultado["vitorias"]
assert resultado["friedman"]["p"] < 0.01, resultado["friedman"]
assert len(resultado["pares"]) == 3, resultado["pares"]
for par in resultado["pares"]:
    assert par["p_ajustado"] >= par["p"] - 1e-12, par
    assert par["significante"], par
assert resultado["n_unidades"] == 8
casos += 1

# ------------------------------------------------------------ 14) determinismo
r1 = eb.teste_permutacao_pcc(prev, alvos, n_perm=100,
                             rng=np.random.default_rng(5))
r2 = eb.teste_permutacao_pcc(prev, alvos, n_perm=100,
                             rng=np.random.default_rng(5))
assert r1["p_agregado"] == r2["p_agregado"], (r1["p_agregado"],
                                              r2["p_agregado"])
assert np.allclose(r1["nulo_media"], r2["nulo_media"], equal_nan=True)
casos += 1

assert casos == 14, casos
print("ESTAT_BCI_OK: %d/14 (acaso EXATO 10->9/10 e 20->15/20; p de permutacao "
      "= (1+#{nulo>=obs})/(1+n_perm); cluster-based acha o efeito plantado e nao "
      "acha efeito nenhum; BH/Holm conferidos a mao; ITR exato: 1 bit/trial = 30 "
      "bits/min)" % casos)

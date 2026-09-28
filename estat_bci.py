"""Bateria estatistica do projeto: acaso, permutacao, IC, FDR, ITR, clusters.

Por que este modulo existe (ver `docs/ESTADO_DA_ARTE_ESTATISTICA.md`): o projeto
media ate' aqui PCC por eixo, desvio mediano do overlay e taxa de falso movimento
em scripts ad-hoc, sem teste de significancia, sem intervalo de confianca e sem
correcao de comparacoes multiplas. Sem esses instrumentos NAO ha' como separar
efeito de ruido -- e as duas armadilhas classicas da area (esquema de validacao
que vaza informacao e "acaso teorico" com poucos trials) produzem justamente os
resultados mais bonitos.

Convencoes
  - SO' numpy + scipy (o projeto nao tem statsmodels; nada aqui depende dele);
  - nenhuma funcao imprime: devolve dicts com nomes ESTAVEIS (o CLI formata);
  - `rng` (np.random.Generator) sempre explicito quando ha' aleatoriedade:
    reprodutibilidade e' requisito de relatorio (COBIDAS MEEG);
  - NaN e' ignorado onde faz sentido (nunca convertido em 0).

Referencias das escolhas (ver o doc para o texto completo):
  - nivel de acaso: Mueller-Putz 2008 (IJBEM 10(1):52-55), Combrisson & Jerbi
    2015 (J Neurosci Methods 245:1-9);
  - permutacao/cluster: Maris & Oostenveld 2007 (J Neurosci Methods 164:177-190);
    limites de interpretacao: Sassenhagen & Draschkow 2019 (Psychophysiology);
  - FDR: Benjamini & Hochberg 1995; Holm 1979;
  - validacao sem vazamento: White & Power 2023 (Sensors 23:6077);
    Schroeder 2025 (Front Neuroergon 6:1582724);
  - ITR: Wolpaw 2000/2002; armadilhas: Yuan 2013 (J Neural Eng 10:026012).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

#: Alfa padrao do projeto (dois lados quando o teste for bicaudal).
ALFA = 0.05


# ============================================================ nivel de acaso
def acerto_minimo(n_trials, alfa=ALFA, n_classes=2, alternativa="maior"):
    """Acertos necessarios para bater o acaso EXATO, pelo binomio.

    O acaso NAO e' o numero redondo do acaso teorico (50% com 2 classes): com
    poucos trials a distribuicao binomial e' larga e o critico sobe muito (com
    N=20 e 2 classes sao 15 acertos = 75%). Reportar "58% > acaso" com N=20 e'
    simplesmente errado.

    Devolve dict com acertos_criticos, n_trials, acerto_critico (fracao),
    acerto_critico_pct, p_no_critico, p0, alfa, alternativa e nota.
    """
    n_trials = int(n_trials)
    if n_trials < 1:
        raise ValueError("n_trials deve ser >= 1.")
    p0 = 1.0 / float(n_classes)
    if alternativa not in ("maior", "bicaudal"):
        raise ValueError("alternativa deve ser 'maior' ou 'bicaudal'.")
    lado = alfa if alternativa == "maior" else alfa / 2.0
    critico = None
    for acertos in range(n_trials + 1):
        if stats.binom.sf(acertos - 1, n_trials, p0) <= lado:
            critico = acertos
            break
    if critico is None:                 # impossivel rejeitar com este N/alfa
        return {"acertos_criticos": None, "n_trials": n_trials,
                "acerto_critico": None, "acerto_critico_pct": None,
                "p_no_critico": None, "p0": p0, "alfa": float(alfa),
                "alternativa": alternativa,
                "nota": "com este N e alfa NAO existe resultado significativo "
                        "(N pequeno demais); colete mais trials"}
    return {"acertos_criticos": int(critico), "n_trials": n_trials,
            "acerto_critico": critico / float(n_trials),
            "acerto_critico_pct": 100.0 * critico / float(n_trials),
            "p_no_critico": float(stats.binom.sf(critico - 1, n_trials, p0)),
            "p0": p0, "alfa": float(alfa), "alternativa": alternativa,
            "nota": ""}


def p_valor_acerto(acertos, n_trials, n_classes=2, alternativa="maior"):
    """p-valor do binomio exato para uma acuracia medida."""
    p0 = 1.0 / float(n_classes)
    acertos = int(acertos)
    n_trials = int(n_trials)
    if alternativa == "bicaudal":
        # bicaudal pelo dobro da menor cauda (conservador no discreto).
        cauda = float(stats.binom.cdf(min(acertos, n_trials - acertos),
                                      n_trials, p0))
        return float(min(1.0, 2.0 * cauda))
    return float(stats.binom.sf(acertos - 1, n_trials, p0))


def acaso_aproximado(n_trials, alfa=ALFA, n_classes=2):
    """Fracao critica pela aproximacao normal (apenas para conferencia).

    c ~= 1/K + z_(1-alfa) * sqrt((K-1) / (K^2 * N)). E' referencia: o valor que
    vale e' o EXATO (`acerto_minimo`), sobretudo com N pequeno.
    """
    k = int(n_classes)
    z = float(stats.norm.ppf(1.0 - alfa))
    return float(1.0 / k + z * np.sqrt((k - 1.0) / (k * k * float(n_trials))))


def tabela_acaso(ns, alfa=ALFA, n_classes=2, alternativa="maior"):
    """Tabela (lista de dicts) do critico exato para varios N."""
    return [acerto_minimo(n, alfa, n_classes, alternativa) for n in ns]


def poder_binomial(n_trials, acerto_verdadeiro, alfa=ALFA, n_classes=2):
    """Poder do teste binomial: P(rejeitar | acerto_verdadeiro) exato."""
    info = acerto_minimo(n_trials, alfa, n_classes)
    if info["acertos_criticos"] is None:
        return 0.0
    return float(stats.binom.sf(info["acertos_criticos"] - 1, int(n_trials),
                                float(acerto_verdadeiro)))


def kappa_de_cohen(matriz_confusao):
    """Kappa de Cohen (acordo corrigido pelo acaso) de uma matriz de confusao.

    Motivo: acuracia bruta premia o desbalanceamento (numa tarefa de 3 classes em
    que 60% dos trials sao de uma so', chutar essa classe da' 60%). O kappa
    desconta o acordo esperado por acaso: 0 = acaso, 1 = perfeito.
    """
    matriz = np.asarray(matriz_confusao, np.float64)
    total = matriz.sum()
    if total <= 0:
        return float("nan")
    esperado = (matriz.sum(axis=1) @ matriz.sum(axis=0)) / (total * total)
    observado = np.trace(matriz) / total
    if abs(1.0 - esperado) < 1e-12:
        return float("nan")
    return float((observado - esperado) / (1.0 - esperado))


def acuracia_balanceada(matriz_confusao):
    """Media do recall por classe (robusta a desbalanceamento)."""
    matriz = np.asarray(matriz_confusao, np.float64)
    linhas = matriz.sum(axis=1)
    recalls = [matriz[i, i] / linhas[i] for i in range(len(linhas))
               if linhas[i] > 0]
    return float(np.mean(recalls)) if recalls else float("nan")


# ==================================================== correlacao (PCC) e IC
def pcc(a, b):
    """PCC de Pearson entre dois vetores 1D, ignorando NaN dos dois lados.

    Devolve NaN quando sobra menos de 3 pontos ou quando um dos lados e' constante
    (PCC indefinido). Nunca devolve 0.0 "de consolo": ausencia de medida nao e'
    medida de ausencia.
    """
    va = np.asarray(a, np.float64).reshape(-1)
    vb = np.asarray(b, np.float64).reshape(-1)
    valido = np.isfinite(va) & np.isfinite(vb)
    if valido.sum() < 3:
        return float("nan")
    xa, xb = va[valido], vb[valido]
    if np.ptp(xa) == 0.0 or np.ptp(xb) == 0.0:
        return float("nan")
    return float(np.corrcoef(xa, xb)[0, 1])


def pcc_por_trial(previsao, alvo):
    """PCC por TRIAL e por EIXO: (N, T, K) -> (N, K).

    Mesma definicao do `validate_epoch` de sand_traj_treino.py (correlacao da
    sequencia predita com a alvo, por eixo, trial a trial) -- aqui isolada para
    poder ser testada e para a estatistica ter uma unica fonte.
    """
    pred = np.asarray(previsao, np.float64)
    tgt = np.asarray(alvo, np.float64)
    if pred.shape != tgt.shape:
        raise ValueError(f"previsao {pred.shape} != alvo {tgt.shape}")
    if pred.ndim == 2:
        pred = pred[:, :, None]
        tgt = tgt[:, :, None]
    n_trials, _, n_eixos = pred.shape
    saida = np.full((n_trials, n_eixos), np.nan)
    for n in range(n_trials):
        for k in range(n_eixos):
            saida[n, k] = pcc(pred[n, :, k], tgt[n, :, k])
    return saida


def pcc_alvo_medio(alvo):
    """PCC por trial do BASELINE "alvo medio" (previsao = media dos trials).

    E' o baseline OBRIGATORIO do projeto (ver tools/compara_alvos.py): como as
    trajetorias de uma condicao sao parecidas entre si, um modelo que devolve
    sempre a trajetoria media tira PCC alto SEM ter decodificado nada. Sem este
    numero ao lado, qualquer PCC reportado e' ambiguo.
    """
    tgt = np.asarray(alvo, np.float64)
    if tgt.ndim == 2:
        tgt = tgt[:, :, None]
    media = np.nanmean(tgt, axis=0)
    previsto = np.repeat(media[None, :, :], tgt.shape[0], axis=0)
    return pcc_por_trial(previsto, tgt)


def ic_pcc_fisher(r, n, alfa=ALFA):
    """Intervalo de confianca do PCC pela transformacao z de Fisher.

    Metodo padrao para CORRELACAO: z = atanh(r), erro padrao 1/sqrt(n-3), volta
    por tanh (n = numero de pontos que entrou na correlacao).
    """
    if not np.isfinite(r) or n is None or int(n) < 4:
        return (float("nan"), float("nan"))
    n = int(n)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    erro = 1.0 / np.sqrt(n - 3.0)
    zcrit = float(stats.norm.ppf(1.0 - alfa / 2.0))
    return (float(np.tanh(z - zcrit * erro)), float(np.tanh(z + zcrit * erro)))


def ic_media_bootstrap(dados, n_boot=10000, alfa=ALFA, rng=None,
                       estatistica=np.nanmean):
    """IC percentil bootstrap da media (ou de outra estatistica) de `dados`.

    Usado para reportar "PCC = 0,71 [0,64; 0,78]" sem depender de normalidade.
    `dados` pode ser (N,) ou (N, K): o IC sai por coluna.
    """
    valores = np.asarray(dados, np.float64)
    if valores.ndim == 1:
        valores = valores[:, None]
    n = valores.shape[0]
    if n < 3:
        largura = valores.shape[1]
        return {"media": np.full(largura, np.nan),
                "lo": np.full(largura, np.nan),
                "hi": np.full(largura, np.nan), "n": n, "n_boot": 0}
    gerador = rng or np.random.default_rng(20260923)
    indices = gerador.integers(0, n, size=(int(n_boot), n))
    amostras = estatistica(valores[indices], axis=1)
    return {"media": np.asarray(estatistica(valores, axis=0), np.float64),
            "lo": np.nanpercentile(amostras, 100.0 * alfa / 2.0, axis=0),
            "hi": np.nanpercentile(amostras, 100.0 * (1.0 - alfa / 2.0), axis=0),
            "n": n, "n_boot": int(n_boot)}


def teste_permutacao_pcc(previsao, alvo, n_perm=5000, bloco=None, rng=None):
    """Permutacao do PCC: o modelo usa informacao ESPECIFICA do trial?

    H0: o pareamento previsao<->alvo e' permutavel -- isto e', o que o modelo
    reproduz nao depende de QUAL trial gerou aquele EEG (seria apenas a
    trajetoria media da condicao). A cada permutacao embaralha-se a ORDEM das
    linhas do alvo (mesmos valores, pareamento destruido) e recalcula-se a
    estatistica.

    `bloco` (opcional, N inteiros): permuta DENTRO de cada bloco (sessao/dia).
    Versao conservadora quando os trials de uma sessao compartilham estado
    (eletrodos, fadiga, posicionamento): a H0 de permutabilidade e' muito mais
    defensavel dentro da sessao.

    p = (1 + #{nulo >= observado}) / (1 + n_perm): nunca zero com n_perm finito
    (p = 0 exigiria infinitas permutacoes) e ja' conta o dado observado.

    Devolve dict com observado (K,), observado_medio, p_por_eixo (K,),
    p_agregado, nulo_media (K,), nulo_p95 (K,), n_perm, n_trials e o
    observado_por_trial (N, K) para quem quiser comparar com o baseline.
    """
    pred = np.asarray(previsao, np.float64)
    tgt = np.asarray(alvo, np.float64)
    if pred.ndim == 2:
        pred = pred[:, :, None]
        tgt = tgt[:, :, None]
    obs = pcc_por_trial(pred, tgt)
    observado = np.nanmean(obs, axis=0)
    n_trials = pred.shape[0]
    gerador = rng or np.random.default_rng(20260923)
    if bloco is None:
        grupos = [np.arange(n_trials)]
    else:
        rotulos = np.asarray(bloco)
        grupos = [np.flatnonzero(rotulos == valor)
                  for valor in np.unique(rotulos)]
    nulos = np.full((int(n_perm), obs.shape[1]), np.nan)
    for i in range(int(n_perm)):
        ordem = np.arange(n_trials)
        for grupo in grupos:
            ordem[grupo] = grupo[gerador.permutation(len(grupo))]
        nulos[i] = np.nanmean(pcc_por_trial(pred, tgt[ordem]), axis=0)
    maiores = nulos >= observado[None, :]
    observado_medio = float(np.nanmean(observado))
    nulo_agregado = np.nanmean(nulos, axis=1)
    p_agregado = ((1.0 + float(np.sum(nulo_agregado >= observado_medio)))
                  / (1.0 + int(n_perm)))
    return {"observado": observado, "observado_medio": observado_medio,
            "p_por_eixo": (1.0 + maiores.sum(axis=0)) / (1.0 + int(n_perm)),
            "p_agregado": float(p_agregado),
            "nulo_media": np.nanmean(nulos, axis=0),
            "nulo_p95": np.nanpercentile(nulos, 95.0, axis=0),
            "n_perm": int(n_perm), "n_trials": int(n_trials),
            "observado_por_trial": obs}


def comparacao_pareada(a, b, n_perm=10000, rng=None):
    """Teste pareado generico (permutacao de sinais) + tamanho de efeito.

    Uso tipico: comparar o PCC por trial do MODELO com o do baseline do alvo
    medio, ou duas configuracoes medidas nos MESMOS trials. H0: a diferenca
    pareada e' simetrica em torno de zero (permutar o SINAL de cada diferenca);
    p e' exato para essa H0 e nao exige normalidade.

    Devolve dict com diferenca_media, p_perm, p_wilcoxon, p_t, ic (bootstrap da
    diferenca), d_z, n.
    """
    xa = np.asarray(a, np.float64).reshape(-1)
    xb = np.asarray(b, np.float64).reshape(-1)
    if xa.size != xb.size:
        raise ValueError("comparacao_pareada exige o mesmo numero de pontos")
    valido = np.isfinite(xa) & np.isfinite(xb)
    dif = (xa - xb)[valido]
    n = dif.size
    if n < 5:
        return {"diferenca_media": float("nan"), "p_perm": float("nan"),
                "p_wilcoxon": float("nan"), "p_t": float("nan"),
                "d_z": float("nan"), "n": int(n),
                "ic": (float("nan"), float("nan"))}
    gerador = rng or np.random.default_rng(20260923)
    sinais = gerador.choice((-1.0, 1.0), size=(int(n_perm), n))
    nulo = np.abs((sinais * dif[None, :]).mean(axis=1))
    observado = abs(float(dif.mean()))
    p_perm = (1.0 + float(np.sum(nulo >= observado))) / (1.0 + int(n_perm))
    ic = ic_media_bootstrap(dif, alfa=ALFA, rng=gerador)
    if np.ptp(dif) == 0.0:
        # Diferencas TODAS iguais: a H0 "simetrica em torno de zero" nao e'
        # testavel por postos (scipy devolve NaN com aviso). Vale principalmente
        # para o caso degenerado a == b, que aparece em teste de sanidade.
        p_wilcoxon = float("nan")
    else:
        try:
            p_wilcoxon = float(stats.wilcoxon(dif).pvalue)
        except ValueError:              # todas as diferencas iguais a zero
            p_wilcoxon = float("nan")
    p_t = float(stats.ttest_rel(xa[valido], xb[valido]).pvalue)
    return {"diferenca_media": float(dif.mean()), "p_perm": float(p_perm),
            "p_wilcoxon": p_wilcoxon, "p_t": p_t,
            "d_z": cohens_dz_pareado(xa[valido], xb[valido]), "n": int(n),
            "ic": (float(ic["lo"][0]), float(ic["hi"][0]))}


# ======================================================= tamanho de efeito
def cohens_d(a, b):
    """d de Cohen para amostras INDEPENDENTES (desvio padrao combinado)."""
    xa = np.asarray(a, np.float64)
    xb = np.asarray(b, np.float64)
    xa, xb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
    n_a, n_b = xa.size, xb.size
    if n_a < 2 or n_b < 2:
        return float("nan")
    var = (((n_a - 1) * xa.var(ddof=1) + (n_b - 1) * xb.var(ddof=1))
           / (n_a + n_b - 2))
    if var <= 0:
        return float("nan")
    return float((xa.mean() - xb.mean()) / np.sqrt(var))


def cohens_dz_pareado(a, b):
    """d_z de Cohen para amostras PAREADAS (divide pelo dp DAS DIFERENCAS).

    E' o tamanho de efeito correto quando o teste e' pareado (mesmos trials /
    mesmos sujeitos nas duas condicoes): usar `cohens_d` (independente) aqui
    superestima, porque ignora a correlacao entre as medidas.
    """
    xa = np.asarray(a, np.float64).reshape(-1)
    xb = np.asarray(b, np.float64).reshape(-1)
    valido = np.isfinite(xa) & np.isfinite(xb)
    if valido.sum() < 3:
        return float("nan")
    dif = xa[valido] - xb[valido]
    if dif.std(ddof=1) == 0:
        return float("nan")
    return float(dif.mean() / dif.std(ddof=1))


def hedges_g(a, b):
    """g de Hedges: d de Cohen corrigido para amostras pequenas (n < 20)."""
    d = cohens_d(a, b)
    xa = np.asarray(a, np.float64)
    xb = np.asarray(b, np.float64)
    xa, xb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
    n_a, n_b = xa.size, xb.size
    if not np.isfinite(d) or n_a < 2 or n_b < 2:
        return float("nan")
    gl = n_a + n_b - 2
    return float(d * (1.0 - 3.0 / (4.0 * gl - 1.0)))


# ============================================ comparacoes multiplas (FDR/Holm)
def bh_fdr(p_valores, alfa=ALFA):
    """Benjamini-Hochberg (FDR) sobre uma familia de p-valores.

    Familia = o conjunto de testes que responde a UMA pergunta (ex.: 21 nos do
    overlay; os 3 eixos do PCC; os canais do EEG). O FDR controla a PROPORCAO
    esperada de falsos positivos entre os rejeitados -- padrao em neuroimagem
    quando o numero de testes e' grande e a exploracao e' aceitavel.

    Devolve dict com p_valores (ordem original), p_ajustado, rejeita, n_testes.
    """
    p = np.asarray(p_valores, np.float64).reshape(-1)
    finitos = np.isfinite(p)
    ajustado = np.full(p.shape, np.nan)
    rejeita = np.zeros(p.shape, bool)
    if finitos.sum() == 0:
        return {"p_valores": p, "p_ajustado": ajustado, "rejeita": rejeita,
                "n_rejeitados": 0, "n_testes": 0}
    ajustado[finitos] = stats.false_discovery_control(p[finitos], method="bh")
    rejeita[finitos] = ajustado[finitos] <= alfa
    return {"p_valores": p, "p_ajustado": ajustado, "rejeita": rejeita,
            "n_rejeitados": int(rejeita.sum()), "n_testes": int(finitos.sum())}


def holm(p_valores, alfa=ALFA):
    """Holm-Bonferroni (FWER): mais estrito que o FDR.

    Use quando a familia e' pequena e cada teste ja' e' confirmatorio (ex.: 3
    eixos; 2 configuracoes x 3 metricas). Mantido aqui porque o projeto tem
    familias pequenas nas comparacoes de modelo -- e escolher o metodo DEPOIS de
    ver os p-valores e' justamente a pratica que o doc proibe.
    """
    p = np.asarray(p_valores, np.float64).reshape(-1)
    finitos = np.isfinite(p)
    ajustado = np.full(p.shape, np.nan)
    rejeita = np.zeros(p.shape, bool)
    if finitos.sum() == 0:
        return {"p_valores": p, "p_ajustado": ajustado, "rejeita": rejeita,
                "n_rejeitados": 0, "n_testes": 0}
    indices = np.flatnonzero(finitos)
    ordem = indices[np.argsort(p[indices])]
    m = ordem.size
    acumulado = 0.0
    for posicao, indice in enumerate(ordem):
        valor = min(1.0, float(p[indice]) * (m - posicao))
        acumulado = max(acumulado, valor)
        ajustado[indice] = acumulado
    rejeita[ordem] = ajustado[ordem] <= alfa
    return {"p_valores": p, "p_ajustado": ajustado, "rejeita": rejeita,
            "n_rejeitados": int(rejeita.sum()), "n_testes": int(m)}


# ===================================================================== ITR
def itr_bits(acerto, n_classes=2):
    """Bits por trial (Wolpaw): log2(K) + P log2(P) + (1-P) log2((1-P)/(K-1)).

    Com P = 1/K o ITR e' ZERO (acaso) e com P = 1 e' log2(K). ATENCAO a armadilha
    do ITR (ver doc): ele depende do tempo por decisao, e escolher a melhor
    janela/sujeito/trial DEPOIS de ver os dados infla o numero -- por isso o ITR
    so' entra no relatorio junto com o IC e com o tempo de decisao declarado.
    """
    p = float(acerto)
    k = int(n_classes)
    if k < 2:
        raise ValueError("n_classes deve ser >= 2")
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float(np.log2(k))
    if p <= 1.0 / k:
        return 0.0
    return float(np.log2(k) + p * np.log2(p)
                 + (1.0 - p) * np.log2((1.0 - p) / (k - 1.0)))


def itr_por_minuto(acerto, n_classes=2, tempo_decisao_s=2.0, corrigido=True):
    """ITR em bits/min = itr_bits * 60 / tempo_decisao_s.

    `corrigido=True` desconta o tempo de cada selecao (ITR pratico, o honesto).
    `corrigido=False` devolve o valor por trial apenas convertido -- inflado
    quando o tempo de decisao e' subestimado (Yuan 2013).
    """
    bits = itr_bits(acerto, n_classes)
    if bits <= 0.0:
        return 0.0
    tempo = float(tempo_decisao_s)
    if tempo <= 0.0:
        raise ValueError("tempo_decisao_s deve ser > 0")
    return float(bits * 60.0 / tempo) if corrigido else float(bits * 60.0)


# ==================================== cluster-based permutation (tempo/eixo 1D)
def teste_cluster_permutacao(a, b, limiar=2.0, n_perm=1000, rng=None):
    """Teste de permutacao por CLUSTER em 1D (ex.: tempo), desenho pareado.

    `a`/`b`: (n_unidades, n_pontos) -- as MESMAS unidades (sujeitos ou trials)
    medidas nas duas condicoes. Em cada ponto calcula-se t pareado; pontos
    contiguos com |t| > `limiar` formam clusters; a estatistica do cluster e' a
    SOMA dos t (positivos ou negativos, nunca misturados). A distribuicao nula
    vem de permutar o SINAL das diferencas dentro de cada unidade (troca a/b em
    unidades aleatorias) e guardar o cluster de maior |soma| de cada permutacao.

    LIMITE DE INTERPRETACAO (Sassenhagen & Draschkow 2019; FAQ do FieldTrip): a
    significancia fala da H0 "as duas condicoes vem da mesma distribuicao", NAO
    da extensao nem do inicio do efeito. NUNCA relatar "houve um cluster
    significativo entre 300 e 500 ms" como se isso fosse a localizacao/latencia
    do efeito: a janela que aparece depende do limiar, do SNR e do numero de
    trials. Para onset, defina a janela A PRIORI e reporte o tamanho de efeito.

    Devolve dict com clusters (n, inicio, fim, soma, p), p_menor, n_pontos,
    t_observado (n_pontos,) e nulo_max (n_perm,).
    """
    xa = np.asarray(a, np.float64)
    xb = np.asarray(b, np.float64)
    if xa.shape != xb.shape or xa.ndim != 2:
        raise ValueError("teste_cluster_permutacao exige dois arrays 2D iguais")
    n_unidades, n_pontos = xa.shape
    if n_unidades < 3:
        raise ValueError("cluster-based exige >= 3 unidades (sujeitos/trials)")
    diferencas = xa - xb
    validos = np.isfinite(diferencas)

    def t_por_ponto(valores):
        saida = np.zeros(n_pontos)
        for ponto in range(n_pontos):
            coluna = valores[validos[:, ponto], ponto]
            if coluna.size >= 3 and coluna.std(ddof=1) > 0:
                saida[ponto] = coluna.mean() / (coluna.std(ddof=1)
                                                / np.sqrt(coluna.size))
        return saida

    def clusters_de(t_valores):
        achados, inicio = [], None
        for ponto, valor in enumerate(t_valores):
            acima = abs(valor) >= limiar
            if acima and inicio is None:
                inicio = ponto
            elif not acima and inicio is not None:
                achados.append((inicio, ponto - 1,
                                float(t_valores[inicio:ponto].sum())))
                inicio = None
        if inicio is not None:
            achados.append((inicio, n_pontos - 1,
                            float(t_valores[inicio:].sum())))
        return achados

    t_observado = t_por_ponto(diferencas)
    observados = clusters_de(t_observado)
    gerador = rng or np.random.default_rng(20260923)
    nulo_max = np.zeros(int(n_perm))
    for i in range(int(n_perm)):
        sinais = gerador.choice((-1.0, 1.0), size=(n_unidades, 1))
        achados = clusters_de(t_por_ponto(diferencas * sinais))
        if achados:
            nulo_max[i] = max(abs(soma) for _, _, soma in achados)
    clusters = []
    for inicio, fim, soma in observados:
        clusters.append({
            "inicio": int(inicio), "fim": int(fim), "soma": soma,
            "p": float((1.0 + float(np.sum(nulo_max >= abs(soma))))
                       / (1.0 + int(n_perm))),
            "n_pontos": int(fim - inicio + 1)})
    return {"clusters": clusters,
            "p_menor": float(min((c["p"] for c in clusters),
                                 default=float("nan"))),
            "n_pontos": int(n_pontos), "n_unidades": int(n_unidades),
            "t_observado": t_observado, "nulo_max": nulo_max,
            "limiar": float(limiar), "n_perm": int(n_perm)}


def comparar_grupos(a, b, pareado=False, n_perm=10000, rng=None):
    """Diferenca entre dois grupos por permutacao + tamanho de efeito.

    `pareado=True` usa o teste de sinais (as MESMAS unidades nas duas condicoes);
    `pareado=False` embaralha a atribuicao de grupo (unidades independentes).
    Caso tipico do projeto (log do overlay): desvio com o Kinect VENDO a mao x
    com o Kinect CEGO -- amostras do mesmo experimento, mas de frames diferentes,
    logo o desenho honesto e' o independente, e o efeito sai com d de Cohen.
    """
    xa = np.asarray(a, np.float64).reshape(-1)
    xb = np.asarray(b, np.float64).reshape(-1)
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    n_a, n_b = xa.size, xb.size
    if n_a < 3 or n_b < 3:
        return {"diferenca_media": float("nan"), "p_perm": float("nan"),
                "d": float("nan"), "n_a": int(n_a), "n_b": int(n_b),
                "pareado": bool(pareado)}
    gerador = rng or np.random.default_rng(20260923)
    if pareado:
        if n_a != n_b:
            raise ValueError("pareado exige o mesmo numero de pontos")
        resultado = comparacao_pareada(xa, xb, n_perm=n_perm, rng=gerador)
        return {"diferenca_media": resultado["diferenca_media"],
                "p_perm": resultado["p_perm"], "d": resultado["d_z"],
                "n_a": int(n_a), "n_b": int(n_b), "pareado": True}
    juntos = np.concatenate([xa, xb])
    observado = abs(xa.mean() - xb.mean())
    nulo = np.zeros(int(n_perm))
    for i in range(int(n_perm)):
        ordem = gerador.permutation(juntos.size)
        nulo[i] = abs(juntos[ordem[:n_a]].mean() - juntos[ordem[n_a:]].mean())
    return {"diferenca_media": float(xa.mean() - xb.mean()),
            "p_perm": (1.0 + float(np.sum(nulo >= observado)))
            / (1.0 + int(n_perm)),
            "d": cohens_d(xa, xb), "n_a": int(n_a), "n_b": int(n_b),
            "pareado": False}


def comparar_metodos(tabela, alfa=ALFA, rng=None):
    """Compara METODOS/configuracoes medidos nas MESMAS unidades (sujeitos).

    `tabela` = {nome: valores_por_unidade}. Com >= 3 metodos roda Friedman
    (omnibus nao-parametrico para medidas repetidas) e depois Wilcoxon pareado
    entre todos os pares, corrigido por Holm; com 2 metodos roda so' o pareado.
    Com menos de 2 metodos, erro.

    Por que assim: testar cada par sem correcao e' a fonte numero 1 de "meu
    metodo ganhou" que nao replica (MOABB comparou 30 pipelines em 36 datasets
    com meta-analise). E a media das posicoes pode esconder que cada sujeito
    prefere um metodo diferente -- por isso o resultado devolve tambem quem
    venceu em cada unidade (`vitorias`).
    """
    nomes = list(tabela)
    if len(nomes) < 2:
        raise ValueError("comparar_metodos exige >= 2 metodos")
    dados = {nome: np.asarray(tabela[nome], np.float64).reshape(-1)
             for nome in nomes}
    n_unidades = min(len(valores) for valores in dados.values())
    matriz = np.column_stack([dados[nome][:n_unidades] for nome in nomes])
    valido = np.isfinite(matriz).all(axis=1)
    matriz = matriz[valido]
    gerador = rng or np.random.default_rng(20260923)
    resultado = {"metodos": nomes, "n_unidades": int(matriz.shape[0])}
    if matriz.shape[0] < 3:
        resultado["nota"] = "menos de 3 unidades completas: sem teste"
        return resultado
    vencedores = [nomes[int(np.argmax(linha))] for linha in matriz]
    resultado["vitorias"] = {nome: vencedores.count(nome) for nome in nomes}
    if len(nomes) >= 3:
        estatistica, p_valor = stats.friedmanchisquare(
            *[matriz[:, i] for i in range(len(nomes))])
        resultado["friedman"] = {"estatistica": float(estatistica),
                                 "p": float(p_valor), "gl": len(nomes) - 1}
    pares = []
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            pareado = comparacao_pareada(matriz[:, i], matriz[:, j],
                                         n_perm=2000, rng=gerador)
            pares.append({"a": nomes[i], "b": nomes[j],
                          "p": float(pareado["p_wilcoxon"]),
                          "d_z": float(pareado["d_z"])})
    ajuste = holm([par["p"] for par in pares], alfa=alfa)
    for posicao, par in enumerate(pares):
        par["p_ajustado"] = float(ajuste["p_ajustado"][posicao])
        par["significante"] = bool(ajuste["rejeita"][posicao])
    resultado["pares"] = pares
    resultado["metodo_correcao"] = "holm"
    return resultado

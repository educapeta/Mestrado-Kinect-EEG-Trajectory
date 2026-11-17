"""Teste do FILTRO CAUSAL no treino (item #3 da spec) -- sem hardware.

Motivo: o tempo real filtra o fluxo continuo com o g.Pype Bandpass (CAUSAL),
mas o treino usava `scipy.signal.sosfiltfilt` (ZERO-FASE), que enxerga o futuro.
Isso dava um vies otimista: o modelo era treinado/avaliado com um sinal que o
tempo real nunca produziria.

Casos:
1. `--filtro` existe e o padrao e' 'causal' (o zero-fase fica p/ comparacao);
2. o modo causal e' DIFERENTE do zero-fase e os dois preservam (N, C, T), com o
   ALVO intacto (o filtro so' mexe no EEG);
3. o AQUECIMENTO funciona: com o prefixo de 2 s a janela causal fica muito mais
   proxima do filtro "quente" (fluxo continuo) do que sem prefixo -- e' o que
   evita trocar o vies otimista por um artefato de transiente;
4. causal NAO ve' o futuro: mexer nas amostras DEPOIS da janela nao muda a
   janela filtrada causalmente (bit a bit), mas muda a zero-fase;
5. o checkpoint grava o modo do filtro (a inferencia precisa saber).
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import scipy.signal

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tools"))

import gerar_sessao_sintetica as gen      # noqa: E402
import sand_traj_treino as treino         # noqa: E402

PASTA = tempfile.mkdtemp(prefix="teste_filtro_causal_")
FS = 500.0
F_LO, F_HI = 1.0, 30.0


def args_do_treino(*extra):
    guardado = sys.argv
    sys.argv = ["sand_traj_treino.py", "--data", PASTA, "--epochs", "1",
                "--output-seq-len", "30"] + list(extra)
    try:
        return treino.parse_args()
    finally:
        sys.argv = guardado


def caso_cli():
    padrao = args_do_treino()
    assert padrao.filtro == treino.FILTRO_CAUSAL, padrao.filtro
    zero = args_do_treino("--filtro", "zero-fase")
    assert zero.filtro == treino.FILTRO_ZERO_FASE, zero.filtro
    print(f"1) OK: --filtro com padrao '{padrao.filtro}' (igual ao tempo real) "
          f"e opcao de comparacao '{zero.filtro}'")


def caso_caminho_de_arquivo():
    """Sessoes sinteticas no formato REAL: causal x zero-fase pelo CSV."""
    args_gen = argparse.Namespace(out=PASTA, sessoes=2, trials=6, fs=FS,
                                  canais=32, seed=11, snr=1.0, sem_ruido=False)
    rng = np.random.default_rng(args_gen.seed)
    for numero in (1, 2):
        prefixo, n, trials, duracao = gen.escreve_sessao(args_gen, numero, rng)
        gen.escreve_movimento_e_eventos(args_gen, prefixo, n, trials, duracao,
                                        numero, rng)
    gravacoes = treino.find_recordings(PASTA)
    assert len(gravacoes) == 2, gravacoes
    x_c, y_c, _, _ = treino.load_all_epochs(gravacoes, args_do_treino())
    x_z, y_z, _, _ = treino.load_all_epochs(
        gravacoes, args_do_treino("--filtro", "zero-fase"))
    assert x_c.shape == x_z.shape, (x_c.shape, x_z.shape)
    assert x_c.shape[1:] == (32, 1000), x_c.shape
    assert np.isfinite(x_c).all() and np.isfinite(x_z).all()
    assert np.array_equal(y_c, y_z), "o filtro nao pode mexer no ALVO"
    delta = float(np.abs(x_c - x_z).max())
    assert delta > 1e-6, "causal e zero-fase deram o MESMO resultado"
    print(f"2) OK: X={x_c.shape} finito nos dois modos, alvo identico e "
          f"|dX|max={delta:.4f} uV (os filtros sao de fato diferentes)")
    return x_c, y_c


def caso_aquecimento_e_futuro():
    rng = np.random.default_rng(3)
    n = int(12 * FS)
    tempo = np.arange(n) / FS
    sinal = (rng.standard_normal(n) * 2.0
             + np.sin(2 * np.pi * 10.0 * tempo) * 3.0
             + np.sin(2 * np.pi * 20.0 * tempo) * 2.0)
    sos = scipy.signal.butter(4, [F_LO, F_HI], btype="bandpass", fs=FS,
                              output="sos")
    quente = scipy.signal.sosfilt(sos, sinal)          # fluxo continuo
    inicio = int(8 * FS)                               # longe do transiente
    janela_n = int(2 * FS)
    referencia = quente[inicio:inicio + janela_n]
    aquecimento = int(round(treino.FILTRO_WARMUP_SEC * FS))

    com = scipy.signal.sosfilt(
        sos, sinal[inicio - aquecimento:inicio + janela_n])[aquecimento:]
    sem = scipy.signal.sosfilt(sos, sinal[inicio:inicio + janela_n])
    erro_com = float(np.mean(np.abs(com - referencia)))
    erro_sem = float(np.mean(np.abs(sem - referencia)))
    assert erro_com < erro_sem, (erro_com, erro_sem)
    assert erro_com < 0.25 * erro_sem, (erro_com, erro_sem)
    print(f"3) OK: {treino.FILTRO_WARMUP_SEC:g} s de aquecimento reduzem o "
          f"transiente de {erro_sem:.4f} p/ {erro_com:.6f} uV "
          f"({erro_sem / max(erro_com, 1e-12):.0f}x menor)")

    alterado = sinal.copy()
    alterado[inicio + janela_n:] += 50.0               # muda so' o DEPOIS
    causal_a = scipy.signal.sosfilt(sos, sinal)[inicio:inicio + janela_n]
    causal_b = scipy.signal.sosfilt(sos, alterado)[inicio:inicio + janela_n]
    zero_a = scipy.signal.sosfiltfilt(sos, sinal)[inicio:inicio + janela_n]
    zero_b = scipy.signal.sosfiltfilt(sos, alterado)[inicio:inicio + janela_n]
    assert np.array_equal(causal_a, causal_b), "causal foi afetado pelo futuro"
    assert not np.allclose(zero_a, zero_b), "zero-fase deveria ver o futuro"
    print("4) OK: causal ignora o futuro (igual bit a bit); zero-fase muda "
          f"(|dZ|max={np.abs(zero_a - zero_b).max():.3f} uV)")


def caso_checkpoint(x_c, y_c):
    import sand_trajectory_model as st
    modelo = os.path.join(PASTA, "_teste_filtro_causal.pt")
    args = args_do_treino("--model-out", modelo)
    treino.run_training(args, x_c[:9], y_c[:9], x_c[9:], y_c[9:],
                        np.arange(x_c.shape[1]), {"teste": True})
    bci = st.SandTrajectoryBCI(modelo)
    assert bci.cfg.get("filtro") == treino.FILTRO_CAUSAL, bci.cfg.get("filtro")
    print(f"5) OK: checkpoint registra filtro='{bci.cfg.get('filtro')}' "
          "(a inferencia sabe com o que o modelo foi treinado)")


def main():
    try:
        caso_cli()
        x_c, y_c = caso_caminho_de_arquivo()
        caso_aquecimento_e_futuro()
        caso_checkpoint(x_c, y_c)
        print("FILTRO_CAUSAL_OK: 5/5 casos (CLI, causal x zero-fase pelo CSV, "
              "aquecimento contra o transiente, causalidade estrita e "
              "checkpoint)")
    finally:
        shutil.rmtree(PASTA, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Teste de REGRESSAO do carregamento de arquivos no treino (caminho CSV+JSON).

Motivo: em 18/09/2026, ao gerar sessoes sinteticas para o piloto sem g.Nautilus
(tools/gerar_sessao_sintetica.py), o treino quebrou com

    RuntimeError: Calculated padded input size per channel: (1). Kernel size: (16)

e imprimiu "Modelo: 1000 canais x 1000 amostras" em vez de 32 x 1000. Causa: um
`.T` a mais em Recording.build_epochs fazia a janela sair (T, C) em vez de
(C, T); como load_all_epochs usava x.shape[1] como numero de canais, o modelo
ganhava 1000 "canais" e o conv estourava. O selftest nao pegava porque ele nao
passa por CSV/JSON.

Casos:
1. gerar 2 sessoes sinteticas e ler com find_recordings + load_all_epochs;
2. X sai (N, C, T) = (48, 32, 1000) e Y sai (N, seq, 3);
3. build_channel_map com o 2o eixo (numero de canais) devolve 32 canais;
4. preprocess_epochs (CAR + z-score por janela) preserva a forma;
5. run_training roda 1 epoca sem erro, sem NaN e salva checkpoint legivel;
6. alvo CONCORRENTE (padrao) e PREDITIVO (--target-start-sec) tem a mesma
   forma, mas alvos diferentes (o preditivo olha o futuro).
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tools"))

import gerar_sessao_sintetica as gen      # noqa: E402
import sand_traj_treino as treino         # noqa: E402
import sand_trajectory_model as st        # noqa: E402

PASTA = tempfile.mkdtemp(prefix="teste_loader_")


def args_do_gerador(trials=6):
    return argparse.Namespace(out=PASTA, sessoes=2, trials=trials, fs=500.0,
                              canais=32, seed=11, snr=1.0, sem_ruido=False)


def args_do_treino(*extra):
    guardado = sys.argv
    sys.argv = ["sand_traj_treino.py", "--data", PASTA, "--epochs", "1",
                "--output-seq-len", "30"] + list(extra)
    try:
        return treino.parse_args()
    finally:
        sys.argv = guardado


def main():
    # 1) sessoes sinteticas no formato REAL (CSV + movimento + JSON)
    args_gen = args_do_gerador()
    rng = np.random.default_rng(args_gen.seed)
    for numero in (1, 2):
        prefixo, n, trials, duracao = gen.escreve_sessao(args_gen, numero, rng)
        gen.escreve_movimento_e_eventos(args_gen, prefixo, n, trials, duracao,
                                        numero, rng)
    arquivos = os.listdir(PASTA)
    assert len([a for a in arquivos if a.endswith("_eventos.json")]) == 2
    assert len([a for a in arquivos if a.endswith("_movimento.csv")]) == 2

    # 2) leitura pelo caminho de ARQUIVO (o que estava quebrado)
    args = args_do_treino()
    gravacoes = treino.find_recordings(args.data)
    assert len(gravacoes) == 2, gravacoes
    x, y, sessoes, origens = treino.load_all_epochs(gravacoes, args)
    assert x.shape[0] == 12, x.shape            # 2 sessoes x 6 trials
    canais, amostras = int(x.shape[1]), int(x.shape[2])
    assert canais == 32, (f"X deveria ter 32 canais no 2o eixo, tem {canais} "
                          "(eixo trocado?)")
    assert amostras == 1000, amostras           # 2 s a 500 Hz
    assert y.shape == (12, 30, 3), y.shape
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert set(np.unique(sessoes)) == {0, 1}
    assert origens.shape == (2, 3)
    print(f"1-2) OK: X={x.shape} (N, canais, amostras), Y={y.shape}")

    # 3) montagem: o 2o eixo de X e' o numero de canais
    channel_map = treino.build_channel_map(args, canais)
    assert channel_map.size == 32, channel_map.size
    assert list(channel_map) == list(range(32))
    x_sel = x[:, channel_map, :]
    assert x_sel.shape == x.shape
    print(f"3) OK: montagem com {channel_map.size} canais (padrao = todos)")

    # 4) pre-processamento
    x_pp = treino.preprocess_epochs(x_sel, args)
    assert x_pp.shape == x.shape, x_pp.shape
    assert np.abs(x_pp[0].mean(axis=1)).max() < 1e-5
    print("4) OK: CAR + z-score por janela preservam (C, T) e media ~0")
    return x_sel, x_pp, y, channel_map


def casos_finais(x_sel, x_pp, y, channel_map):
    # 5) treino de 1 epoca + checkpoint + inferencia pelo wrapper do tempo real
    modelo_pt = os.path.join(PASTA, "_teste_loader.pt")
    args = args_do_treino("--model-out", modelo_pt)
    resultado = treino.run_training(args, x_sel[:9], y[:9],
                                    x_sel[9:], y[9:], channel_map,
                                    {"teste": True})
    assert os.path.exists(modelo_pt), "checkpoint nao foi salvo"
    for chave in ("best_val_loss", "pcc_x", "pcc_y", "pcc_z"):
        assert np.isfinite(resultado[chave]), (chave, resultado[chave])
    bci = st.SandTrajectoryBCI(modelo_pt)
    assert bci.n_timesteps == 1000, bci.n_timesteps          # 2 s a 500 Hz
    assert bci.channel_map.size == 32, bci.channel_map.size
    assert bci.cfg["window_sec"] == 2.0, bci.cfg
    assert len(bci.cfg["channel_map"]) == 32, bci.cfg["channel_map"]
    trajeto, movimentacao = bci.predict_window(x_sel[0].T)   # wrapper: (T, C)
    trajeto = np.asarray(trajeto)
    assert trajeto.shape == (30, 3), trajeto.shape
    assert np.isfinite(trajeto).all()
    assert np.isfinite(movimentacao) and movimentacao >= 0.0, movimentacao
    print(f"5) OK: checkpoint salvo + wrapper do tempo real devolve "
          f"{trajeto.shape} a partir de (T, C)={x_sel[0].T.shape} "
          f"(movimento previsto ~{movimentacao:.4f} m)")

    # 6) alvo concorrente x preditivo (mesma forma, valores diferentes)
    args_pred = args_do_treino("--event-code", "795", "--window-start-sec",
                               "-0.5", "--target-start-sec", "1.5",
                               "--target-end-sec", "2.5")
    gravacoes = treino.find_recordings(args_pred.data)
    x_p, y_p, _, _ = treino.load_all_epochs(gravacoes[:1], args_pred)
    assert x_p.shape[1:] == (32, 1000), x_p.shape
    assert y_p.shape[1:] == (30, 3), y_p.shape
    delta = np.abs(y_p - y[:y_p.shape[0]]).max()
    assert delta > 1e-6, "alvo preditivo identico ao concorrente"
    print(f"6) OK: alvo preditivo {y_p.shape} difere do concorrente "
          f"(|dY|max={delta:.4f} m)")

    print("TREINO_LOADER_OK: 6/6 casos (formas, montagem, pre-processamento, "
          "treino 1 epoca, checkpoint + inferencia, alvo concorrente x "
          "preditivo)")


if __name__ == "__main__":
    try:
        casos_finais(*main())
    finally:
        shutil.rmtree(PASTA, ignore_errors=True)
    sys.exit(0)
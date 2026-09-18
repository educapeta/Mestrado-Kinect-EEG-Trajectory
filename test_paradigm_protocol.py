"""Teste do PROTOCOLO do paradigma (sem hardware) — item #14 da revisao.

Cobre o que o smoke manual nao garantia:
1. Lista de trials balanceada (250 = 6 condicoes) e cortada em blocos de 50.
2. Duracao exata do trial (repouso 2 + cue/ME 4 + video 2 + MI 4 = 12 s).
3. Codigos de marcador: unicos entre si e TODOS com legenda em CODE_NAMES.
4. Contrato de colunas: EEG cru (34 colunas) + movimento em arquivo (58).
5. Vigia do link (#15): sem amostras novas avisa e marca 899/898.
6. Tabela de impedancias (#23): marca canal ruim, conta OK, trata "-10".
7. Questionario do participante (#2): ENTER mantem o padrao e o JSON sai.
8. SharedState: diagnosticos novos e traducao da fonte da posicao (KT_src).
"""
import builtins
import json
import os
import queue
import random
import sys
import time

import numpy as np

import eeg_motor_paradigm as p

# ---------------------------------------------------------------------------
# 1) Lista de trials: balanceada na SESSAO e embaralhada
# ---------------------------------------------------------------------------
rng = random.Random(42)
trials = p._ParadigmController.build_trial_list(50, 5, rng)
assert len(trials) == 250, len(trials)
contagem = {}
for trial in trials:
    contagem[trial["condicao"]] = contagem.get(trial["condicao"], 0) + 1
assert len(contagem) == 6, contagem
assert set(contagem.values()) == {41, 42}, contagem
assert all(t["objeto"] in p.OBJECTS and t["mao"] in p.HANDS for t in trials)
ciclica = [f"{obj}_{mao}" for obj in p.OBJECTS for mao in p.HANDS]
assert [t["condicao"] for t in trials[:6]] != ciclica, "ordem nao embaralhada"
# blocos de 50: 5 blocos exatos
assert len(trials) % 50 == 0

# ---------------------------------------------------------------------------
# 2) Duracao do trial = 12 s e protocolo sem ITI (repouso do proximo trial)
# ---------------------------------------------------------------------------
total_trial = (p.HOME_SEC + p.CUE_ME_SEC + p.PAUSE_VIDEO_SEC + p.MI_SEC)
assert abs(total_trial - 12.0) < 1e-9, total_trial
assert p.ITI_MIN_SEC == 0.0 and p.ITI_MAX_SEC == 0.0
assert abs(p.HOME_SEC - 2.0) < 1e-9 and abs(p.CUE_ME_SEC - 4.0) < 1e-9
assert abs(p.PAUSE_VIDEO_SEC - 2.0) < 1e-9 and abs(p.MI_SEC - 4.0) < 1e-9

# ---------------------------------------------------------------------------
# 3) Codigos: unicos, com legenda, e mao/condicao sem colisao entre si
# ---------------------------------------------------------------------------
codigos = {
    "sessao": p.CODE_SESSION, "fim_sessao": p.CODE_SESSION_END,
    "olhos_abertos": p.CODE_EYES_OPEN, "olhos_fechados": p.CODE_EYES_CLOSED,
    "baseline_repouso": p.CODE_BASELINE_REST, "trial": p.CODE_TRIAL,
    "origem_ini": p.CODE_ORIGIN_START, "origem_fim": p.CODE_ORIGIN_END,
    "me_ini": p.CODE_ME_START, "me_fim": p.CODE_ME_END,
    "mi_ini": p.CODE_MI_START, "mi_fim": p.CODE_MI_END,
    "bloco_ini": p.CODE_BLOCK_START, "bloco_fim": p.CODE_BLOCK_END,
    "pausa_ini": p.CODE_PAUSE_START, "pausa_fim": p.CODE_PAUSE_END,
    "video": p.CODE_SLOWMO_START,
    "link_perdido": p.CODE_LINK_LOST, "link_ok": p.CODE_LINK_OK,
    "movimento_onset": p.CODE_MOVE_ONSET,
}
assert len(set(codigos.values())) == len(codigos), "codigo de marcador repetido"
for nome, valor in codigos.items():
    assert valor in p.CODE_NAMES, f"sem legenda em CODE_NAMES: {nome}"
for mao, valor in p.CODE_HAND.items():
    assert valor in p.CODE_NAMES, mao
for condicao, valor in p.CODE_CONDITION.items():
    assert valor in p.CODE_NAMES, condicao
assert set(p.CODE_HAND.values()).isdisjoint(p.CODE_CONDITION.values())
todos = (list(codigos.values()) + list(p.CODE_HAND.values())
         + list(p.CODE_CONDITION.values()))
assert len(set(todos)) == len(todos), "colisao entre codigos de mao/condicao"

# ---------------------------------------------------------------------------
# 4) Contrato de colunas: padrao = EEG cru (34) + movimento separado (58)
# ---------------------------------------------------------------------------
assert p.MOTION_IN_EEG_CSV is False, "o padrao deve ser EEG CRU + movimento fora"
assert p.N_MOTION_COLS == 59, p.N_MOTION_COLS
mov_cols = list(p.MOTION_COLUMNS)
assert mov_cols[-3:] == ["KT_hand", "KT_src", "KT_onset"], mov_cols[-3:]
assert len(mov_cols) == len(set(mov_cols)), "colunas de movimento duplicadas"
n_eeg = 32
eeg_header = (["Time"] + [f"EEG_Ch{i + 1:02d}" for i in range(n_eeg)]
              + ["Marker"])
assert len(eeg_header) == 34, len(eeg_header)
assert eeg_header[-1] == "Marker" and "KT_x_m" not in eeg_header
mov_header = ["t_mono_s", "t_epoch_s"] + mov_cols
assert len(mov_header) == 61, len(mov_header)
assert mov_header[2] == "KT_x_m" and mov_header[-1] == "KT_onset"

# ---------------------------------------------------------------------------
# 5) Vigia do link (#15)
# ---------------------------------------------------------------------------
class _MuxerFalso:
    def __init__(self):
        self.samples_written = 0


fila = queue.Queue()
estado = p.SharedState()
muxer_falso = _MuxerFalso()
vigia = p.LinkWatchdog(muxer_falso, estado, fila, timeout=0.2, period=0.05)
vigia.start()
time.sleep(0.6)
assert estado.link_ok == 0, "o vigia deveria ter marcado o link como perdido"
eventos = []
while not fila.empty():
    eventos.append(fila.get_nowait())
assert any(e["code"] == p.CODE_LINK_LOST for e in eventos), eventos
for evento in eventos:
    assert {"code", "nome", "monotonic_s"} <= set(evento)
    assert evento["nome"], "evento do vigia sem legenda"
assert vigia.perdas == 1, vigia.perdas
# amostras voltam E continuam chegando (se pararem de novo, o vigia acusa de
# novo -- comportamento correto): mantem o contador avancando por ~0,6 s
for valor in range(100, 140):
    muxer_falso.samples_written = valor
    time.sleep(0.015)
vigia.stop()
while not fila.empty():
    eventos.append(fila.get_nowait())
assert any(e["code"] == p.CODE_LINK_OK for e in eventos), eventos
assert estado.link_ok == 1 and vigia.perdas == 1, (estado.link_ok, vigia.perdas)

# ---------------------------------------------------------------------------
# 6) Tabela de impedancias (#23): marca ruim, conta OK, trata "-10" (nao medido)
# ---------------------------------------------------------------------------
valores = [3.0, 12.0, -10.0, 80.0] + [5.0] * 28
texto = p._impedance_table(valores, limite_kohm=50.0)
assert "Ch01" in texto and "Ch32" in texto, texto
linhas = texto.splitlines()
assert "30/32 canais OK" in linhas[-1], linhas[-1]
# as linhas de canal vem antes da legenda: exatamente um '*' (Ch04, 80 kOhm)
assert sum(linha.count("*") for linha in linhas[:-1]) == 1, texto
assert "Ch04: 80.0*" in texto, texto
assert "Ch03:  ---" in texto, texto          # -10 = nao medido
assert "impedancia ainda nao medida" in p._impedance_table(None)


# ---------------------------------------------------------------------------
# 7) Questionario do participante (#2): ENTER mantem o padrao; JSON e' gravado
# ---------------------------------------------------------------------------
class _ArgsFalso:
    participante = "P07"
    sessao = "S2"
    mao_dominante = "direita"


entradas = iter(["", "", "24", "", "", "", "", "", "", "cansado"])
input_original = builtins.input
builtins.input = lambda _prompt="": next(entradas)
try:
    respostas = p._ask_participant(_ArgsFalso())
finally:
    builtins.input = input_original
assert respostas["participante"] == "P07"      # padrao vindo do CLI
assert respostas["sessao"] == "S2"
assert respostas["mao_dominante"] == "direita"
assert respostas["idade"] == "24"
assert respostas["observacoes"] == "cansado"
assert "respondido_em" in respostas

p._write_participant_file("_teste_protocolo.csv",
                          {"protocolo": "teste", "participante": "P07"},
                          respostas)
caminho_participante = "_teste_protocolo_participante.json"
assert os.path.exists(caminho_participante)
with open(caminho_participante, encoding="utf-8") as handle:
    dados = json.load(handle)
assert dados["participante"]["idade"] == "24"
assert dados["sessao_meta"]["protocolo"] == "teste"
os.remove(caminho_participante)

# ---------------------------------------------------------------------------
# 8) SharedState: diagnosticos (#15/#23) e traducao da fonte (KT_src)
# ---------------------------------------------------------------------------
estado2 = p.SharedState()
assert estado2.link_ok == 1 and estado2.impedance_values is None
estado2.set_link_status(0, "sem amostras")
assert estado2.link_ok == 0 and estado2.link_text == "sem amostras"
estado2.set_impedance([1.5, 2.5], "tabela")
valores_guardados, texto_guardado = estado2.impedance_snapshot()
assert valores_guardados == [1.5, 2.5] and texto_guardado == "tabela"
movimento, _fase, _cue, _msg = estado2.snapshot()
assert movimento["hand"] == 0 and movimento["src_code"] == 0
assert movimento["onset"] == 0
# a coluna KT_src distingue MEDIDA real de ESTIMATIVA
assert p.kt_src_code("triangulado_laptop") == 1
assert p.kt_src_code("kinect_depth") == 2
assert p.kt_src_code("esqueletoSDK") == 3
assert p.kt_src_code("modelo3D") == 4
assert p.kt_src_code("nominal!") == 7
assert p.kt_src_code("") == 0

# ---------------------------------------------------------------------------
# 9) Integracao com o TREINO: o carregador le o CSV de movimento separado (#1)
#    e alinha ao EEG pelas ancoras de relogio do JSON (#8).
# ---------------------------------------------------------------------------
import shutil
import tempfile

import sand_traj_treino as st

pasta_teste = tempfile.mkdtemp(prefix="_teste_sessao_", dir=".")
try:
    caminho_eeg = os.path.join(pasta_teste, "gravacao_MEMI_teste.csv")
    caminho_mov = os.path.join(pasta_teste, "gravacao_MEMI_teste_movimento.csv")
    caminho_json = os.path.join(pasta_teste, "gravacao_MEMI_teste_eventos.json")

    fs_teste, n_amostras = 500.0, 2000
    with open(caminho_eeg, "w", encoding="utf-8") as handle:
        handle.write("Time," + ",".join(f"EEG_Ch{i:02d}" for i in range(1, 5))
                     + ",Marker\n")
        for indice in range(n_amostras):
            colunas = [f"{0.5 * np.sin(2 * np.pi * 10 * indice / fs_teste):.6g}"
                       for _ in range(4)]
            marca = p.CODE_ME_START if indice == 500 else 0
            handle.write(f"{indice / fs_teste:.4f}," + ",".join(colunas)
                         + f",{marca}\n")

    # Movimento a 30 Hz com tempos ABSOLUTOS (o mesmo relogio do JSON).
    t0_abs = 1000.0
    with open(caminho_mov, "w", encoding="utf-8") as handle:
        handle.write("t_mono_s,t_epoch_s,KT_x_m,KT_y_m,KT_z_m,KT_src\n")
        for indice in range(120):
            instante = t0_abs + indice / 30.0
            # a linha 30 e' uma ESTIMATIVA (KT_src=4) e deve ser invalidada
            fonte = 4 if indice == 30 else 1
            handle.write(f"{instante:.6f},{instante + 1.7e9:.6f},"
                         f"{indice * 0.01:.6f},0,0,{fonte}\n")

    with open(caminho_json, "w", encoding="utf-8") as handle:
        json.dump({"eventos": [
            {"code": p.CODE_SESSION, "sample_index": 0,
             "monotonic_s": t0_abs, "nome": "inicio da sessao"},
            {"code": p.CODE_ME_START, "sample_index": 500,
             "monotonic_s": t0_abs + 500 / fs_teste, "nome": "inicio fase ME"},
        ], "origem_xyz_m": [1.0, 2.0, 3.0]}, handle, ensure_ascii=False)

    # O arquivo de movimento NAO pode ser confundido com uma sessao.
    achados = st.find_recordings(pasta_teste)
    assert len(achados) == 1 and achados[0][0] == caminho_eeg, achados

    gravacao = st.Recording(caminho_eeg, caminho_json, fs=fs_teste,
                            window_n=1000, output_seq_len=30,
                            event_code=p.CODE_ME_START, ktt_valid_min=1.0,
                            f_lo=1.0, f_hi=30.0)
    gravacao._parse_header()
    assert gravacao.motion is not None, "deveria carregar o movimento separado"
    assert gravacao.list_event_samples() == [500]
    a, b = gravacao._sample_to_time()
    assert abs(a - 1.0 / fs_teste) < 1e-9, (a, b)
    assert abs(b - t0_abs) < 1e-6, (a, b)

    eeg, kt, valid = gravacao.load_rows(500, 100)
    assert eeg.shape == (100, 4) and kt.shape == (100, 3)
    # amostra 500 do EEG cai em t0_abs + 1.0 s = linha 30 do movimento (0.30 m)
    assert abs(kt[0, 0] - 0.30) < 1e-6, kt[0]
    # ...mas essa linha e' estimativa (KT_src=4) -> invalida para treino
    assert valid[0] == 0.0, valid[:3]
    # amostra 560 -> t0_abs + 1.12 s -> linha 34 (medida real, 0.34 m)
    assert abs(kt[60, 0] - 0.34) < 1e-6, kt[60]
    assert valid[60] == 1.0, valid[55:65]

    # 9b) ALVO PREDITIVO: com --target-start-sec o alvo deixa de ser a propria
    #     janela e passa a ser a trajetoria FUTURA (o que controle de protese
    #     exige). No CSV sintetico KT_x cresce 0.01 por linha de movimento
    #     (30 Hz): o alvo deslocado em 1 s deve comecar ~0.30 m mais adiante.
    preditivo = st.Recording(caminho_eeg, caminho_json, fs=fs_teste,
                             window_n=1000, output_seq_len=30,
                             event_code=p.CODE_ME_START, ktt_valid_min=1.0,
                             f_lo=1.0, f_hi=30.0,
                             start_offset=int(-1.0 * fs_teste),
                             target_offset=int(1.0 * fs_teste),
                             target_n=int(1.0 * fs_teste))
    preditivo._parse_header()
    _eeg_p, _kt_p, valid_p = preditivo.load_rows(550, 300)
    assert valid_p.mean() > 0.5, valid_p.mean()
    assert preditivo.target_offset == 500 and preditivo.target_n == 500
    # a amostra 550 do EEG cai em t0_abs+1.1 s -> linha 33 do movimento
    # (KT_x = 0.33 m); o alvo comeca na linha 63 do movimento (KT_x = 0.63 m)
    assert abs(_kt_p[0, 0] - 0.33) < 1e-6, _kt_p[0]
    _, kt_alvo, _v = preditivo.load_rows(550 + 500, 30)
    assert abs(kt_alvo[0, 0] - 0.63) < 1e-6, kt_alvo[0]
finally:
    shutil.rmtree(pasta_teste, ignore_errors=True)

print("PARADIGM_OK: 9/9 blocos (trials balanceados, trial de 12 s, codigos "
      "unicos, EEG de 34 colunas + movimento de 45 a 30 Hz (com KT_onset), "
      "vigia 899/898, painel de impedancias, questionario, KT_src e integracao "
      "com o treino)")
sys.exit(0)
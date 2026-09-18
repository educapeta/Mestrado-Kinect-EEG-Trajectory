"""Gera SESSOES SINTETICAS no formato exato do Programa 1 (teste sem hardware).

Motivo: validar o pipeline completo (marcacao do onset, janela ancorada no
planejamento motor, alvo concorrente x preditivo, treino do SAND, metadados)
sem o g.Nautilus e sem o Kinect.

O que ele simula, por trial (12 s como no protocolo real):
    [home 2 s] -> [cue + ME 4 s] -> [pausa + video 2 s] -> [MI 4 s]

  - MOVIMENTO: trajetoria 3D do punho (minimo-jerk) com alvo proprio: alcance
    (~0,15 m), pegada/inclinacao, levantamento e retorno, com ONSET aleatorio
    0,35-0,9 s apos a cue (como um participante real);
  - EEG (32 canais, uV): mu (10 Hz) e beta (20 Hz) com ERD (dessincronizacao)
    time-locked ao onset + MRCP (deflexao lenta 0,5-3 Hz negativa) comecando
    ~1 s ANTES do onset (mais forte em FC1/C3/Cz/C4) + ruido de banda 0,5-40 Hz.
    Assim os dados sinteticos CONTEM a informacao pre-movimento que a janela
    ancorada no onset deve capturar.

Arquivos gerados (identicos ao formato real):
    <stamp>.csv                 EEG cru: Time, EEG_Ch01..32, Marker
    <stamp>_movimento.csv       movimento a 30 Hz (MOTION_COLUMNS do paradigma)
    <stamp>_eventos.json        marcadores com a amostra exata + meta
    <stamp>_participante.json   questionario ficticio

Uso:
    .venv\\Scripts\\python.exe tools\\gerar_sessao_sintetica.py
    .venv\\Scripts\\python.exe tools\\gerar_sessao_sintetica.py --sessoes 3 --trials 30
    .venv\\Scripts\\python.exe tools\\gerar_sessao_sintetica.py --out gravacoes_sinteticas

Depois o treino roda igual ao real (ex.):
    .venv\\Scripts\\python.exe sand_traj_treino.py --data gravacoes_sinteticas \\
        --event-code 795 --window-start-sec -0.5 --window-sec 2.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# A biblioteca do paradigma vive na raiz do projeto (um nivel acima de tools/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eeg_motor_paradigm as emp  # noqa: E402

FS = 500.0
FS_MOV = 30.0
HOME_SEC, ME_SEC, PAUSE_SEC, MI_SEC = 2.0, 4.0, 2.0, 4.0
TRIAL_SEC = HOME_SEC + ME_SEC + PAUSE_SEC + MI_SEC
T_MONO_START = 1000.0          # relogio monotonico ficticio das sessoes


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out", default="gravacoes_sinteticas")
    parser.add_argument("--sessoes", type=int, default=2)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--fs", type=float, default=FS)
    parser.add_argument("--canais", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--snr", type=float, default=1.0,
                        help="Escala global do sinal de movimento (1 = tipico)")
    parser.add_argument("--sem-ruido", action="store_true",
                        help="EEG sem ruido (teste ideal, PCC ~1)")
    parser.add_argument("--sem-direcao", action="store_true",
                        help="EEG sem a informacao da direcao do alvo (so' o "
                             "instante do movimento): serve para demonstrar "
                             "que nesse caso o baseline da trajetoria media "
                             "NAO pode ser batido por nenhum modelo")
    return parser.parse_args()


def interpolacao_minimo_jerk(fracao):
    """Perfil suave 0->1 (minimo-jerk) para o alcance do braco."""
    fracao = np.clip(fracao, 0.0, 1.0)
    return 10 * fracao ** 3 - 15 * fracao ** 4 + 6 * fracao ** 5


def trajetoria_do_punho(t, t_onset, rng, alvo):
    """Posicao 3D do punho (m) no instante t (s), dada o onset simulado.

    Fases: alcance (0,45 s) -> pega/levanta (0,45 s) -> segura (0,5 s) ->
    retorno ao home (0,6 s). `alvo` = deslocamento do objeto (m).
    """
    pulso = np.zeros(3)
    if t < t_onset:
        return pulso
    dt = t - t_onset
    if dt < 0.45:                                    # alcance ate o objeto
        pulso = alvo * interpolacao_minimo_jerk(dt / 0.45)
    elif dt < 0.90:                                  # fecha a mao e levanta
        u = interpolacao_minimo_jerk((dt - 0.45) / 0.45)
        pulso = alvo + np.array([0.0, -0.07 * u, -0.02 * u])
    elif dt < 1.40:                                  # segura
        pulso = alvo + np.array([0.0, -0.07, -0.02])
    elif dt < 2.00:                                  # retorna ao home
        u = interpolacao_minimo_jerk((dt - 1.40) / 0.60)
        pulso = (alvo + np.array([0.0, -0.07, -0.02])) * (1.0 - u)
    return pulso + rng.normal(0, 0.0008, 3)          # ruido de medicao (~1 mm)


def ganho_direcional(canal, direcao):
    """Ganho do ERD que CODIFICA o alvo do movimento (canal 1-based).

    No dado sintetico a topografia e' ficticia mas explicita: os canais 9-16
    modulam com o alvo em X e os canais 17-24 com o alvo em Z. E' o analogo
    grosseiro do ERD contralateral (que no EEG real codifica a direcao do
    movimento). Com `--sem-direcao` os ganhos ficam todos em 1,0 e o EEG passa a
    conter APENAS o instante do movimento -- nesse caso nenhum modelo pode
    superar o baseline da trajetoria media (ver tools/compara_alvos.py).
    """
    if direcao is None:
        return 1.0
    ganho = 1.0
    for eixo, primeiro, ultimo in (("x", 9, 16), ("z", 17, 24)):
        if primeiro <= canal <= ultimo:
            deslocamento = float(direcao[0] if eixo == "x" else direcao[2])
            ganho = 1.0 + 0.7 * float(np.clip(deslocamento / 0.12, -1.5, 1.5))
    return ganho


def eeg_sintetico(t, t_onset, t_cue, canais, fs, rng, snr=1.0,
                  com_ruido=True, direcao=None):
    """EEG (canais, T) com ERD de mu/beta e MRCP ancorados no onset."""
    n = t.size
    eeg = np.zeros((canais, n))
    for canal in range(canais):
        # ganho por canal: mais forte sobre FC1/C3/Cz/C4 (motor)
        motor = 1.6 if canal in (8, 12, 13, 14) else 0.7
        motor *= ganho_direcional(canal + 1, direcao)
        fase = rng.uniform(0, 2 * np.pi)
        # --- ERD de mu (10 Hz) e beta (20 Hz): amplitude cai a partir de
        # onset-0,5 s e volta ao normal ~1,5 s depois (dessincronizacao)
        def erd(dt, largura=0.5):
            return np.where(dt < -largura, 1.0,
                            np.where(dt > 1.5, 1.0,
                                     np.clip(0.45 + 0.55 * np.abs(dt) / 1.5,
                                             0.45, 1.0)))
        dt = t - t_onset
        mu = motor * 8.0 * erd(dt) * snr * np.sin(2 * np.pi * 10 * t + fase)
        beta = motor * 4.0 * erd(dt) * snr * np.sin(
            2 * np.pi * 20 * t + fase * 1.7)
        # --- MRCP: deflexao lenta negativa (0,5-3 Hz) que comeca ~1 s antes
        # do onset e termina ~0,5 s depois
        mrpc = -motor * 9.0 * snr * np.exp(-0.5 * ((t - t_onset) / 0.45) ** 2) \
            * (t > t_onset - 1.2) * (t < t_onset + 0.6)
        # --- ritmo de repouso (alpha 10 Hz) fora do movimento + deriva lenta
        alpha = 3.0 * motor * np.sin(2 * np.pi * 9.5 * t + fase * 0.3)
        deriva = 4.0 * snr * np.sin(2 * np.pi * 0.3 * t + fase * 0.7)
        eeg[canal] = mu + beta + mrpc + alpha + deriva
    if com_ruido:
        eeg += rng.normal(0, 4.0, (canais, n))       # ruido branco ~4 uV
    return eeg


def eventos_do_trial(t0, objeto, mao, trial):
    """Marcadores do trial com os MESMOS codigos do paradigma real."""
    t_cue = t0 + HOME_SEC
    t_onset = t_cue + trial["atraso"]
    t_pause = t_cue + ME_SEC
    t_mi = t_pause + PAUSE_SEC
    t_fim = t_mi + MI_SEC
    lista = [
        (emp.CODE_TRIAL, t0, {"trial": trial["numero"], "bloco": trial["bloco"]}),
        (emp.CODE_HAND[mao], t_cue, {"mao": mao}),
        (emp.CODE_CONDITION[(objeto, mao)], t_cue + 0.1,
         {"objeto": objeto, "mao": mao, "condicao": f"{objeto}_{mao}"}),
        (emp.CODE_ME_START, t_cue, {"fase": "ME"}),
        (emp.CODE_MOVE_ONSET, t_onset, {"fase": "onset"}),
        (emp.CODE_ME_END, t_pause, {"fase": "fim ME"}),
        (emp.CODE_SLOWMO_START, t_pause, {"fase": "video 0.5x"}),
        (emp.CODE_MI_START, t_mi, {"fase": "MI"}),
        (emp.CODE_MI_END, t_fim, {"fase": "fim MI"}),
    ]
    return sorted(lista, key=lambda item: item[1])


def alvo_do_objeto(objeto, rng):
    """Posicao do objeto na mesa (m, relativa ao home).

    A lateral varia por objeto (esquerda/centro/direita), como na mesa real:
    isso torna os alvos BIMODAIS em X, enfraquece o baseline da trajetoria media
    e faz o EEG direcional (ganho_direcional) ser a unica fonte de informacao
    sobre o lado do alvo.
    """
    lateral = {"garrafa": (-0.17, -0.09), "caneta": (-0.03, 0.03),
               "bola": (0.09, 0.17)}[objeto]
    return np.array([rng.uniform(*lateral), rng.uniform(-0.06, 0.06),
                     rng.uniform(-0.16, -0.08)])


def monta_trials(args, rng):
    """Lista balanceada de trials com onset/trajetoria sorteados."""
    condicoes = [(o, m) for o in emp.OBJECTS for m in emp.HANDS]
    base, sobra = divmod(args.trials, len(condicoes))
    sorteio = condicoes * base + list(rng.permutation(condicoes))[:sobra]
    rng.shuffle(sorteio)
    trials = []
    for indice, (objeto, mao) in enumerate(sorteio):
        trials.append({
            "numero": indice + 1,
            "bloco": indice // max(1, (args.trials + 4) // 5) + 1,
            "objeto": objeto, "mao": mao,
            "atraso": float(rng.uniform(0.35, 0.90)),   # onset apos a cue
            "alvo": alvo_do_objeto(objeto, rng),
            "onset_pendente": True,
        })
    return trials


def escreve_sessao(args, numero_sessao, rng):
    fs, canais = args.fs, args.canais
    trials = monta_trials(args, rng)
    n_trials = len(trials)
    duracao = 2.0 + n_trials * TRIAL_SEC + 2.0
    n = int(duracao * fs)
    t = np.arange(n) / fs
    eeg = np.zeros((canais, n), np.float32)
    marcadores = np.zeros(n, np.int32)
    for trial in trials:
        t0 = 2.0 + (trial["numero"] - 1) * TRIAL_SEC
        t_cue = t0 + HOME_SEC
        ini = max(0, int((t0 - 1.5) * fs))
        fim = min(n, int((t_cue + ME_SEC + 0.5) * fs))
        eeg[:, ini:fim] += eeg_sintetico(
            t[ini:fim], t_cue + trial["atraso"], t_cue, canais, fs, rng,
            args.snr, not args.sem_ruido,
            direcao=(None if getattr(args, "sem_direcao", False)
                     else trial["alvo"])).astype(np.float32)
        for codigo, instante, _extra in eventos_do_trial(
                t0, trial["objeto"], trial["mao"], trial):
            marcadores[min(n - 1, int(round(instante * fs)))] = codigo

    marca = f"{time.strftime('%Y%m%d_%H%M%S')}_S{numero_sessao}"
    prefixo = os.path.join(args.out, f"gravacao_MEMI_{marca}")
    cabecalho = ("Time,"
                 + ",".join(f"EEG_Ch{i:02d}" for i in range(1, canais + 1))
                 + ",Marker")
    with open(prefixo + ".csv", "w", encoding="utf-8", newline="") as handle:
        handle.write(cabecalho + "\n")
        for indice in range(n):
            linha = [f"{indice / fs:.4f}"]
            linha += [f"{valor:.6g}" for valor in eeg[:, indice]]
            linha.append(str(int(marcadores[indice])))
            handle.write(",".join(linha) + "\n")
    return prefixo, n, trials, duracao

def escreve_movimento_e_eventos(args, prefixo, n, trials, duracao,
                                numero_sessao, rng):
    """Movimento a 30 Hz (MOTION_COLUMNS) + JSON de eventos + participante."""
    fs = args.fs
    base_motion = dict(emp.SharedState().motion)
    with open(prefixo + "_movimento.csv", "w", encoding="utf-8",
              newline="") as handle:
        handle.write("t_mono_s,t_epoch_s,"
                     + ",".join(emp.MOTION_COLUMNS) + "\n")
        for instante in np.arange(0.0, duracao, 1.0 / FS_MOV):
            indice_trial = int((instante - 2.0) // TRIAL_SEC) \
                if instante >= 2.0 else -1
            trial = next((tr for tr in trials
                          if tr["numero"] - 1 == indice_trial), None)
            movimento = dict(base_motion)
            movimento["onset"] = 0
            if trial is None:
                posicao = np.zeros(3)
            else:
                t_cue = 2.0 + (trial["numero"] - 1) * TRIAL_SEC + HOME_SEC
                t_onset = t_cue + trial["atraso"]
                posicao = trajetoria_do_punho(instante, t_onset, rng,
                                              trial["alvo"])
                if trial["onset_pendente"] and instante >= t_onset:
                    movimento["onset"] = 1        # 1 linha = amostra do onset
                    trial["onset_pendente"] = False
            movimento.update({
                "x": float(posicao[0]), "y": float(posicao[1]),
                "z": float(posicao[2]), "valid": 1, "src": "triangulado",
                "src_code": 1,
                "hand": emp.ARM_SIDE_CODE.get(trial["mao"] if trial else "", 0),
                "roll": float(6 * np.sin(2 * np.pi * 0.2 * instante)),
                "pitch": float(4 * np.cos(2 * np.pi * 0.15 * instante)),
                "yaw": float(10 * np.sin(2 * np.pi * 0.1 * instante)),
                "zero_lock": 1,
            })
            linha = [f"{T_MONO_START + instante:.6f}",
                     f"{T_MONO_START + instante:.6f}"]
            linha += [f"{valor:.6g}" for valor in
                      emp.build_motion_row(movimento, None, None)]
            handle.write(",".join(linha) + "\n")

    eventos = []
    for trial in trials:
        t0 = 2.0 + (trial["numero"] - 1) * TRIAL_SEC
        for codigo, instante, extra in eventos_do_trial(
                t0, trial["objeto"], trial["mao"], trial):
            evento = {
                "code": int(codigo),
                "nome": emp.CODE_NAMES.get(int(codigo), "?"),
                "hora": time.strftime("%H:%M:%S"),
                "monotonic_s": round(T_MONO_START + instante, 6),
                "sample_index": int(min(n - 1, round(instante * fs))),
                "origem": "sintetico",
            }
            evento.update(extra)
            eventos.append(evento)
    eventos.sort(key=lambda item: item["sample_index"])
    payload = {
        "fs": fs,
        "canais_eeg": args.canais,
        "movimento_no_csv_de_eeg": False,
        "colunas_movimento": emp.MOTION_COLUMNS,
        "hold_samples": int(0.1 * fs),
        "legenda_codigos": emp.CODE_NAMES,
        "origem_xyz_m": [0.0, 0.0, 0.0],
        "origem_rpy_deg": [0.0, 0.0, 0.0],
        "modelo_braco": None,
        "numero_de_eventos": len(eventos),
        "meta": {
            "protocolo_versao": "SINTETICO-1.0",
            "participante": f"SYN_{numero_sessao}",
            "sessao": f"S{numero_sessao}",
            "sintetico": True,
            "gerador": "tools/gerar_sessao_sintetica.py",
            "descricao": ("EEG com ERD de mu/beta e MRCP ancorados no onset; "
                          "movimento minimo-jerk com onset em [0.35, 0.90] s "
                          "apos a cue"),
            "unidades": {"EEG_ChNN": "uV", "KT_*": "m (relativo a origem)"},
            "trials": len(trials),
        },
        "eventos": eventos,
    }
    with open(prefixo + "_eventos.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with open(prefixo + "_participante.json", "w", encoding="utf-8") as handle:
        json.dump({"participante": {"participante": f"SYN_{numero_sessao}",
                                    "sintetico": "sim", "idade": "30"}},
                  handle, ensure_ascii=False, indent=2)
    return len(eventos)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print(f"Gerando {args.sessoes} sessao(oes) em '{args.out}' "
          f"({args.trials} trials de {TRIAL_SEC:g} s, {args.canais} canais, "
          f"{args.fs:g} Hz)")
    for numero in range(1, args.sessoes + 1):
        prefixo, n, trials, duracao = escreve_sessao(args, numero, rng)
        n_eventos = escreve_movimento_e_eventos(args, prefixo, n, trials,
                                               duracao, numero, rng)
        onsets = [tr["atraso"] for tr in trials]
        print(f"  S{numero}: {os.path.basename(prefixo)}.csv "
              f"({n} amostras = {duracao:.0f} s) | {n_eventos} eventos | "
              f"onset medio +{np.mean(onsets):.2f} s "
              f"(min +{min(onsets):.2f}, max +{max(onsets):.2f})")
    print("\nTreino sugerido (janela ancorada no planejamento motor):")
    print(f"  .venv\\Scripts\\python.exe sand_traj_treino.py --data {args.out} "
          f"--event-code {emp.CODE_MOVE_ONSET} --window-start-sec -0.5 "
          f"--window-sec 2.0")
    print("Compare com o alvo concorrente ancorado na cue (--event-code "
          f"{emp.CODE_ME_START}) e com o alvo preditivo "
          "(--target-start-sec 1.5 --target-end-sec 2.5).")


if __name__ == "__main__":
    main()
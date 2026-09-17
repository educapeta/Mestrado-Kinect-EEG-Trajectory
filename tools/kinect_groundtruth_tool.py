"""Ferramenta autonoma de calibracao/diagnostico do ground truth 3D da mao.

Este programa era o `main()` de `kinect_imu_groundtruth.py`; foi movido para
`tools/` (itens #9/#10 da revisao) para deixar a biblioteca como biblioteca --
antes, importar o modulo trazia junto um programa de 700+ linhas.

Uso (da raiz do projeto, que e' onde ficam as calibracoes):

    python tools/kinect_groundtruth_tool.py

Teclas: C origem | B bias do acelerometro | T diagnostico do tabuleiro (48
cantos) | K trim fixo do overlay | H calibracao do esqueleto da mao | L log de
desvio | R reset | ESC sair. Precisa do Kinect v2 (SDK 2.0) e, opcionalmente,
da webcam auxiliar e do ESP32/UDP.

Nao ha copia de codigo: o corpo abaixo e' o mesmo de antes e usa os nomes do
modulo importado (constantes, classes e helpers), copiados para o namespace
deste arquivo logo apos o import.
"""
from __future__ import annotations

import os
import sys

# A biblioteca vive na raiz do projeto (um nivel acima de tools/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kinect_imu_groundtruth as _gt  # noqa: E402

# Copia TODO o namespace do modulo (inclusive nomes com "_"), para o corpo
# abaixo enxergar exatamente os mesmos nomes de antes do movimento.
globals().update({nome: valor for nome, valor in vars(_gt).items()
                  if not nome.startswith("__")})


def main():
    receiver = ImuReceiver(UDP_PORT)
    receiver.start()
    tracker = KinectHandTracker()
    alignment = CameraAlignment()
    # Use the same auxiliary resolution the stereo calibration was made at.
    requested_size = alignment.calibrated_auxiliary_size or (1280, 720)
    auxiliary = cv2.VideoCapture(AUX_CAMERA_INDEX)
    auxiliary.set(cv2.CAP_PROP_FRAME_WIDTH, requested_size[0])
    auxiliary.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_size[1])
    obtained_size = (
        int(auxiliary.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(auxiliary.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print(
        f"Camera auxiliar: solicitado {requested_size[0]}x{requested_size[1]}, "
        f"obtido {obtained_size[0]}x{obtained_size[1]} | "
        f"calibrado: {alignment.calibrated_auxiliary_size}"
    )
    if alignment.stereo_rotation is not None:
        if alignment.stereo_ready:
            state = "aceita"
        elif alignment.stereo_usable:
            state = "aproximada (RMS alto)"
        else:
            state = f"rejeitada (RMS > {STEREO_OVERLAY_MAX_RMS_PX:.0f} px)"
        print(
            f"Calibracao stereo: RMS={alignment.stereo_rms:.2f} px ({state}) | "
            f"tam. Kinect={alignment.calibrated_kinect_size} tam. aux={alignment.calibrated_auxiliary_size}"
        )
    else:
        print("Calibracao stereo: arquivo ausente ou invalido; sobreposicao desligada")
    fusion = PositionFusion()
    # Camera calibration is performed only by stereo_calibration.py.
    camera_calibrating = False
    bias_calibration_until = 0.0
    calibration_until = time.monotonic() + CALIBRATION_SECONDS
    imu_calibration_active = True
    # Calibração do esqueleto (mão): captura nó-a-nó auxiliar <-> Kinect.
    hand_calib_active = False
    hand_calib_samples_aux = []
    hand_calib_samples_kin = []
    hand_calib_samples_depths = []
    hand_calib_count = 0
    last_hand_sample_time = 0.0
    # Log de diagnostico do overlay (tecla L): 1 linha/frame em
    # overlay_deviation_log.csv. Colunas: t, z_usado, z_bruto, cx, cy,
    # borda(0 centro..1 borda), velocidade_px_s, + por no (dev,dx,dy).
    overlay_logging = False
    overlay_log_file = None
    # Medicao do trim fixo (tecla K): coleta ~3 s com a mao parada.
    trim_measure_until = 0.0
    trim_measure_samples = []
    overlay_prev_palm = None
    overlay_prev_time = 0.0
    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["pc_time_s", "imu_timestamp_us", "kinect_x_m", "kinect_y_m", "kinect_z_m", "fused_x_m", "fused_y_m", "fused_z_m", "roll_deg", "pitch_deg", "yaw_deg", "elbow_angle_deg"])
    try:
        while True:
            now = time.monotonic()
            color, depth = tracker.frames()
            auxiliary_ok, auxiliary_frame = auxiliary.read()
            if auxiliary_ok:
                auxiliary_frame = transform_auxiliary_frame(auxiliary_frame)
            if auxiliary_ok:
                alignment.set_auxiliary_size(auxiliary_frame)
            auxiliary_for_calibration = None if not auxiliary_ok else auxiliary_frame.copy()
            color_for_calibration = None if color is None else color.copy()
            display, camera_position, landmarks = tracker.detect_hand(color, depth)
            auxiliary_hands = tracker.detect_auxiliary_hands(auxiliary_frame if auxiliary_ok else None)
            # A mao auxiliar saiu do frame? Reseta o EMA de profundidade
            # estimada (modelo 3D). Sem isso, o EMA rejeitaria o primeiro
            # estimate pos-retorno como "salto >30 cm" e a mao vermelha
            # ficaria CONGELADA na profundidade antiga (desalinhada).
            if not auxiliary_hands and alignment._hand_depth_ema_initialized:
                alignment.reset_hand_depth_ema()
            overlay_deviation = None
            depth_text = ""
            # Captura de amostras para a calibração do esqueleto (mão)
            if hand_calib_active and landmarks is not None and len(auxiliary_hands) > 0:
                if (
                    hand_calib_count < HAND_CALIBRATION_SAMPLES_REQUIRED
                    and now - last_hand_sample_time > 2.0
                ):
                    # Só aceita amostra se o Kinect depth é confiável
                    # (>=14 nós com depth válido). Caso contrário, pula
                    # e tenta de novo no próximo frame (sem contar).
                    sample_depths = tracker.landmark_depths(landmarks).astype(np.float64)
                    valid_depth = np.isfinite(sample_depths) & (sample_depths > 0.1)
                    if valid_depth.sum() < 14:
                        print(
                            "Amostra IGNORADA: poucos nós com depth válido "
                            f"({valid_depth.sum()}/21). Mão mais centrada/"
                            "afastada do Kinect."
                        )
                        last_hand_sample_time = now
                    else:
                        hand_calib_samples_aux.append(
                            np.array(
                                [(lm.x, lm.y) for lm in auxiliary_hands[0]],
                                dtype=np.float64,
                            )
                        )
                        hand_calib_samples_kin.append(
                            np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float64)
                        )
                        hand_calib_samples_depths.append(sample_depths)
                        hand_calib_count = len(hand_calib_samples_aux)
                        last_hand_sample_time = now
                        print(
                            f"Amostra de mao: {hand_calib_count}/"
                            f"{HAND_CALIBRATION_SAMPLES_REQUIRED} (varie posicao, "
                            "profundidade e inclinacao da mao)"
                        )
                    if hand_calib_count >= HAND_CALIBRATION_SAMPLES_REQUIRED:
                        all_aux = np.concatenate(hand_calib_samples_aux)
                        span = np.ptp(all_aux, axis=0)
                        if np.hypot(*span) < HAND_CALIBRATION_MIN_SPAN:
                            print(
                                "Calibração de mano: pouca varredura (a mano quase "
                                "sempre na mesma zona). Refaça a captura (tecla P) "
                                "moviendo a mano por TODO o frame."
                            )
                            hand_calib_active = False
                        else:
                            ok = alignment.solve_hand_landmark_calibration(
                                hand_calib_samples_aux,
                                hand_calib_samples_kin,
                                samples_depths=hand_calib_samples_depths,
                            )
                            if ok:
                                print(
                                    "Calibração de esqueleto guardada em "
                                    f"{HAND_CALIBRATION_FILE.name}; aplique-se "
                                    "automaticamente no overlay."
                                )
                            else:
                                print("Falha ao resolver a calibração de esqueleto.")
                            hand_calib_active = False
            if display is not None and (alignment.stereo_usable or alignment.hand_landmark_ready):
                # REGRA DE DESENHO: a transformacao aux->Kinect e 100%
                # calibracao (stereo 3D + profundidade estimada por
                # calibracao). Nada aqui "segue a mao verde" do Kinect:
                # refinar o overlay com landmarks do Kinect acoplava a
                # projecao ao tracking e fazia a mao vermelha perder o
                # alinhamento exatamente quando o Kinect perde a mao.
                kinect_depth_m = None
                depth_note = "Kinect"
                # --- Prioridade 1: triangulacao estereo (espaco vetorial 3D) ---
                # Com as DUAS cameras vendo a mao, o esqueleto 3D (21,3) em
                # metros e MEDIDO pela interseccao de raios (calibracao
                # stereo). A profundidade deixa de ser estimada por escala
                # aparente e o overlay fica imune a inclinacao da mao.
                triangulated_3d = None
                if (
                    alignment.stereo_usable
                    and landmarks is not None
                    and len(landmarks) == 21
                    and len(auxiliary_hands) > 0
                ):
                    aux_px = [
                        (lm.x * alignment.auxiliary_width, lm.y * alignment.auxiliary_height)
                        for lm in auxiliary_hands[0]
                    ]
                    kin_px = [
                        (lm.x * tracker.color_width, lm.y * tracker.color_height)
                        for lm in landmarks
                    ]
                    tri3d, tri_err = alignment.triangulate_hand_points(aux_px, kin_px)
                    if tri3d is not None and np.isfinite(tri3d).any():
                        triangulated_3d = tri3d
                        # Guarda o esqueleto 3D medido (espaco vetorial) com o
                        # instante da medicao para fusao/CSV/analises futuras.
                        alignment.last_triangulated_3d = tri3d
                        alignment.last_triangulated_3d_time = now
                    else:
                        alignment.last_triangulated_3d = None
                        alignment.last_triangulated_3d_time = None
                else:
                    # Sem as DUAS cameras vendo a mao nao existe medicao 3D:
                    # nao deixa esqueleto velho sobreviver aqui (consumidores
                    # checam last_triangulated_3d_time para frescor).
                    alignment.last_triangulated_3d = None
                    alignment.last_triangulated_3d_time = None
                if landmarks is not None:
                    kinect_depth_m = tracker.landmark_depths(landmarks)
                    if not np.isfinite(np.asarray(kinect_depth_m, np.float32)).any():
                        kinect_depth_m = None
                if kinect_depth_m is None and auxiliary_hands:
                    # Norma de desenho: o overlay NAO depende do tracking do
                    # Kinect. Se o Kinect perdeu a mao, a profundidade vem de
                    # fontes independentes do tracking verde, por prioridade:
                    # (a) esqueleto do SDK (medida REAL de depth — o Z vem do
                    #     body/depth index, nao segue tracking nenhum);
                    # (b) modelo 3D calibrado da vista auxiliar (escala
                    #     aparente, calibracao fixa do .npz);
                    # (c) amostragem da cena onde a palma projeta.
                    # A roxa/vermelha fica onde a calibracao diz.
                    sdk_anchor = tracker.body_hand_anchor()
                    if sdk_anchor is not None:
                        kinect_depth_m = [sdk_anchor[2]]
                        depth_note = "esqueletoSDK"
                        # Atualiza a memoria de profundidade (mesma EMA do
                        # detect_hand) para alimentar os fallbacks seguintes.
                        if tracker.last_hand_depth_m is None:
                            tracker.last_hand_depth_m = sdk_anchor[2]
                        else:
                            tracker.last_hand_depth_m = (
                                0.8 * tracker.last_hand_depth_m
                                + 0.2 * sdk_anchor[2]
                            )
                    if kinect_depth_m is None:
                        hand_aux = auxiliary_hands[0]
                        aux_normalized = np.array(
                            [(lm.x, lm.y) for lm in hand_aux], dtype=np.float64
                        )
                        model_depth = alignment.get_smoothed_hand_depth(aux_normalized)
                        if model_depth is not None:
                            kinect_depth_m = [model_depth]
                            depth_note = "modelo3D"
                        else:
                            # Sem calibracao 3D da mao: amostra a cena onde a
                            # palma auxiliar projeta (ainda usa a transformacao
                            # calibrada, nao o tracking verde).
                            guess = tracker.last_hand_depth_m or NOMINAL_HAND_DEPTH_M
                            kinect_depth_m = [
                                tracker.depth_at_projected(
                                    alignment, hand_aux[9], guess
                                )
                            ]
                            depth_note = "amostrada"
                if kinect_depth_m is None:
                    if tracker.last_hand_depth_m is not None:
                        kinect_depth_m = [tracker.last_hand_depth_m]
                        depth_note = "ultima valida"
                    else:
                        kinect_depth_m = [NOMINAL_HAND_DEPTH_M]
                        depth_note = "nominal"

                # --- Profundidad unica robusta, EMA adaptativa y con bootstrap --
                # El overlay proyecta la mano con UN solo Z (cuerpo rigido).
                # Tres problemas del EMA anterior, visibles en el log de desvio:
                # (1) arrancaba en NOMINAL_HAND_DEPTH_M y tardaba >1 s en
                #     converger (transitorio de ~200 px);
                # (2) alpha fija 0.15 no seguia la mano al acercarse rapido a
                #     la auxiliar (z 1.2->0.8 m) y la proyeccion vuela fuera;
                # (3) no distinguia una medida real del Kinect de un chute.
                # Ahora: bootstrap desde la primera medida real, alpha adaptativa
                # a la variacion frame-a-frame, rechazo de picos >0.50 m y clip
                # fisico al rango del sensor.
                depths_for_projection = kinect_depth_m
                if kinect_depth_m is not None:
                    depths_arr = np.asarray(kinect_depth_m, np.float32).reshape(-1)
                    valid_depths = depths_arr[
                        np.isfinite(depths_arr)
                        & (depths_arr > 0.1)
                        & (depths_arr <= (DEPTH_MAX_MM + 100) / 1000.0)
                    ]
                    if valid_depths.size > 0:
                        single_depth = float(np.median(valid_depths))
                    else:
                        single_depth = tracker.last_hand_depth_m or NOMINAL_HAND_DEPTH_M
                    single_depth = float(np.clip(single_depth, 0.4, 3.0))
                    # confidence: 2 = medida real do Kinect (MediaPipe+depth ou
                    # esqueleto do SDK; pode bootstrap),
                    # 1 = estimacao da vista auxiliar (modelo 3D calibrado ou
                    # amostragem da cena), 0 = chute nominal.
                    confidence = 0
                    if depth_note in ("Kinect", "esqueletoSDK"):
                        confidence = 2
                    elif depth_note in ("modelo3D", "amostrada"):
                        confidence = 1
                    ema = getattr(alignment, "_kinect_depth_ema", None)
                    ema_trust = getattr(alignment, "_kinect_depth_ema_trust", 0)
                    if ema is None or not np.isfinite(ema):
                        ema, ema_trust = single_depth, confidence
                    elif confidence >= 2 and ema_trust < 2:
                        # Primera medida real del Kinect: bootstrap inmediato
                        # para eliminar el transitorio del arranque.
                        ema, ema_trust = single_depth, 2
                    else:
                        delta = float(abs(single_depth - ema))
                        if delta > 0.50:
                            # Salto grande en UN frame: imposible para una mano
                            # real (max ~0.05 m/frame a 30 fps). Es basura del
                            # sensor (mide el fondo, oclusion, borde del FOV).
                            # NO se debe seguir el pico; pero si la medida es
                            # real (el bootstrap inicial fue un fondo), una
                            # deriva minima (0.05) permite recuperarse sin que
                            # el pico tome el control.
                            alpha = 0.05 if confidence >= 2 else 0.0
                        else:
                            # Adaptativa: sigue rapido cuando Z se mueve (mano
                            # acercandose), suave cuando esta quieta.
                            alpha = float(np.clip(0.12 + 2.5 * delta, 0.12, 0.60))
                            if confidence < 2:
                                alpha = min(alpha, 0.20)
                        ema = (1.0 - alpha) * ema + alpha * single_depth
                        ema_trust = max(ema_trust, confidence)
                    alignment._kinect_depth_ema = float(ema)
                    alignment._kinect_depth_ema_trust = int(ema_trust)
                    depths_for_projection = [float(ema)]
                if triangulated_3d is not None:
                    # A triangulacao (medida real em 3D) tem prioridade sobre
                    # qualquer profundidade estimada/EMA para o desenho.
                    palm_z = float(triangulated_3d[9, 2])
                    if not np.isfinite(palm_z):
                        palm_z = float(np.nanmedian(triangulated_3d[:, 2]))
                    depths_for_projection = [palm_z]
                    depth_note = "triangulado"

                overlay_deviation = tracker.draw_auxiliary_hands_on_color(
                    display, auxiliary_hands, alignment, depths_for_projection,
                    triangulated_3d,
                )
                overlay_red = overlay_green = None
                if overlay_deviation is not None and len(overlay_deviation) == 5:
                    dev, dx, dy, overlay_red, overlay_green = overlay_deviation
                    # Sem verde (Kinect cego) o desvio e indefinido, mas a
                    # projecao vermelha segue sendo desenhada e logada.
                    overlay_deviation = (dev, dx, dy) if dev is not None else None
                # --- Medicao do trim fixo (tecla K, 3 s, mao PARADA) ------------
                # Coleta (verde - vermelho) por frame e, ao fim, grava a
                # traducao constante em overlay_trim.json. O offset medido e
                # somado ao trim atual (o vermelho desenhado ja inclui o trim).
                if trim_measure_until > 0.0:
                    if now < trim_measure_until:
                        if overlay_red is not None and overlay_green is not None:
                            diffs = np.asarray(overlay_green, np.float64) - np.asarray(
                                overlay_red, np.float64
                            )
                            valid = np.isfinite(diffs).all(axis=1)
                            if valid.any():
                                trim_measure_samples.append(
                                    diffs[valid].mean(axis=0) + alignment.overlay_trim
                                )
                    else:
                        trim_measure_until = 0.0
                        if len(trim_measure_samples) >= 20:
                            samples = np.asarray(trim_measure_samples, np.float64)
                            med = np.median(samples, axis=0)
                            spread = np.linalg.norm(samples - med[None, :], axis=1)
                            keep = spread <= max(15.0, float(np.median(spread)) * 3.0)
                            new_trim = samples[keep].mean(axis=0) if keep.any() else med
                            alignment.overlay_trim = np.asarray(new_trim, np.float64)
                            try:
                                with open(OVERLAY_TRIM_PATH, "w", encoding="utf-8") as fh:
                                    json.dump(
                                        {
                                            "dx": float(new_trim[0]),
                                            "dy": float(new_trim[1]),
                                            "samples": int(len(samples)),
                                        },
                                        fh,
                                    )
                                print(
                                    f"Trim do overlay salvo: dx {new_trim[0]:+.1f} px, "
                                    f"dy {new_trim[1]:+.1f} px ({len(samples)} frames)"
                                )
                            except OSError as exc:
                                print(
                                    f"Trim medido (falha ao salvar: {exc}): "
                                    f"dx {new_trim[0]:+.1f}, dy {new_trim[1]:+.1f}"
                                )
                        else:
                            print(
                                "Trim cancelado: poucos frames validos "
                                f"({len(trim_measure_samples)}). Mao precisa estar "
                                "visivel nas DUAS cameras e parada."
                            )
                        trim_measure_samples = []
                mark = "" if depth_note in ("Kinect", "esqueletoSDK") else "!"
                depth_text = (
                    f"Z {np.nanmedian(np.asarray(kinect_depth_m, np.float32)):.2f} m "
                    f"({depth_note}{mark})"
                )
                # --- Log de diagnostico do overlay (tecla L liga/desliga) --------
                # Grava 1 linha/frame em overlay_deviation_log.csv com o desvio
                # vermelho-verde por no + contexto (profundidade, posicao na
                # imagem, velocidade da mao). Serve para classificar o erro:
                # estatico (bias) x movimento (sincronia) x bordas (distorcao).
                # Frames SEM mao verde sao gravados com as colunas de desvio
                # vazias (so a trajetoria vermelha + contexto).
                if overlay_logging and overlay_red is not None:
                    try:
                        red = np.asarray(overlay_red, np.float64).reshape(-1, 2)
                        grn = (
                            np.asarray(overlay_green, np.float64).reshape(-1, 2)
                            if overlay_green is not None
                            else None
                        )
                        n_nodes = 21 if grn is None else min(21, red.shape[0], grn.shape[0])
                        palm_ref = red if grn is None else grn
                        palm = palm_ref[9] if n_nodes > 9 else palm_ref[0]
                        cx = float(palm[0]) if np.isfinite(palm).all() else float("nan")
                        cy = float(palm[1]) if np.isfinite(palm).all() else float("nan")
                        edge = float("nan")
                        if np.isfinite([cx, cy]).all():
                            edge = float(
                                max(
                                    cx / tracker.color_width,
                                    (tracker.color_width - cx) / tracker.color_width,
                                    cy / tracker.color_height,
                                    (tracker.color_height - cy) / tracker.color_height,
                                )
                            )
                        z_used = float(np.nanmedian(np.asarray(depths_for_projection, np.float32)))
                        z_raw = float(np.nanmedian(np.asarray(kinect_depth_m, np.float32)))
                        palm_now = np.array([cx, cy], np.float64)
                        speed = float("nan")
                        if np.isfinite(palm_now).all() and overlay_prev_palm is not None:
                            dt = now - overlay_prev_time if overlay_prev_time else float("nan")
                            if dt and dt > 1e-3:
                                speed = float(np.linalg.norm(palm_now - overlay_prev_palm) / dt)
                        overlay_prev_palm = palm_now if np.isfinite(palm_now).all() else None
                        overlay_prev_time = now
                        row = [f"{now:.3f}", f"{z_used:.3f}", f"{z_raw:.3f}",
                               f"{cx:.1f}", f"{cy:.1f}", f"{edge:.3f}",
                               f"{speed:.1f}" if np.isfinite(speed) else "",
                               depth_note]
                        for n in range(n_nodes):
                            if (
                                grn is not None
                                and np.isfinite(red[n]).all()
                                and np.isfinite(grn[n]).all()
                            ):
                                d = grn[n] - red[n]
                                row.append(f"{np.hypot(*d):.1f}")
                                row.append(f"{d[0]:+.1f}")
                                row.append(f"{d[1]:+.1f}")
                            else:
                                row += ["", "", ""]
                        overlay_log_file.write(",".join(row) + "\n")
                    except (ValueError, IndexError, TypeError):
                        pass
            if auxiliary_ok:
                auxiliary_display = (
                    cv2.flip(auxiliary_frame, 1)
                    if AUXILIARY_DISPLAY_MIRROR
                    else auxiliary_frame.copy()
                )
                tracker.draw_auxiliary_hands(auxiliary_display, auxiliary_hands)
                cv2.putText(
                    auxiliary_display,
                    f"Camera auxiliar | maos: {len(auxiliary_hands)}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                if alignment.stereo_ready:
                    alignment_text = f"stereo carregado (RMS {alignment.stereo_rms:.2f} px)"
                elif alignment.stereo_usable:
                    alignment_text = (
                        f"stereo aproximado (RMS {alignment.stereo_rms:.2f} px) "
                        "- recalibre para < 2 px"
                    )
                elif alignment.stereo_rms is not None:
                    alignment_text = f"stereo rejeitado (RMS {alignment.stereo_rms:.2f} px)"
                else:
                    alignment_text = "stereo nao carregado"
                cv2.putText(
                    auxiliary_display,
                    alignment_text,
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255) if alignment.stereo_usable else (0, 0, 255),
                    2,
                )
                aux_small = cv2.resize(auxiliary_display, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)
                cv2.imshow("Camera auxiliar - tracking", aux_small)
            sample = receiver.get_latest()
            palm_direction = fusion.palm_direction(sample)
            if camera_calibrating:
                if color_for_calibration is not None and auxiliary_ok:
                    calibration_kinect = cv2.cvtColor(color_for_calibration, cv2.COLOR_BGRA2BGR)
                    kinect_found = alignment.draw_board_detection(calibration_kinect)
                    auxiliary_found = alignment.draw_board_detection(auxiliary_frame)
                    auxiliary_calibration_display = (
                        cv2.flip(auxiliary_frame, 1)
                        if AUXILIARY_DISPLAY_MIRROR
                        else auxiliary_frame.copy()
                    )
                    cv2.putText(
                        calibration_kinect,
                        f"Tabuleiro: {'OK' if kinect_found else 'nao encontrado'} | ESPACO captura",
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0) if kinect_found else (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        auxiliary_calibration_display,
                        f"Tabuleiro: {'OK' if auxiliary_found else 'nao encontrado'} | ESPACO captura",
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0) if auxiliary_found else (0, 0, 255),
                        2,
                    )
                    cal_kin_small = cv2.resize(calibration_kinect, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)
                    cal_aux_small = cv2.resize(auxiliary_calibration_display, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)
                    cv2.imshow("Calibracao - Kinect", cal_kin_small)
                    cv2.imshow("Calibracao - camera auxiliar", cal_aux_small)
            if imu_calibration_active:
                fusion.add_bias_sample(sample)
                palm_normal = None if landmarks is None else tracker.palm_normal_camera(landmarks)
                if now >= calibration_until and sample is not None and camera_position is not None and palm_normal is not None:
                    fusion.calibrate_palm_direction(sample, palm_normal)
                    fusion.origin = camera_position.copy()
                    fusion.position[:] = 0.0
                    fusion.velocity[:] = 0.0
                    imu_calibration_active = False
            elif now < bias_calibration_until:
                fusion.add_bias_sample(sample)
            fused = fusion.position.copy() if imu_calibration_active else fusion.update(sample, camera_position)
            if sample is not None:
                imu_text = f"IMU R/P/Y {sample.roll_deg:.1f}/{sample.pitch_deg:.1f}/{sample.yaw_deg:.1f}"
            else:
                imu_text = "IMU aguardando UDP"
            if camera_position is None:
                camera_text = f"Kinect sem mao ({tracker.last_hand_reason})"
            else:
                camera_text = f"Kinect XYZ {camera_position[0]:.3f} {camera_position[1]:.3f} {camera_position[2]:.3f} m"
            if camera_calibrating:
                calibration_text = f"CALIBRACAO: tabuleiro visivel nas duas cameras, ESPACO ({len(alignment.auxiliary_points)}/{CALIBRATION_SAMPLES_REQUIRED})"
            elif imu_calibration_active:
                remaining = max(0.0, calibration_until - now)
                calibration_text = f"CALIBRANDO IMU: mao visivel e parada ({remaining:.1f} s)"
            else:
                calibration_text = "C origem | B bias | R reset | H cal mao | P limpa | T tabuleiro | ESC sair"
            if display is not None and palm_direction is not None and landmarks is not None:
                tracker.draw_palm_vector(display, landmarks, palm_direction)
            depth_status = "depth: sem frame"
            stats = tracker.depth_stats()
            if stats is not None:
                count, dmin, dmax, frac = stats
                health = "OK" if frac > 0.30 else ("FRACO" if frac > 0.02 else "FALHA")
                depth_status = f"depth {health}: {frac:.0%} valido ({dmin:.0f}-{dmax:.0f} mm)"
            if alignment.stereo_usable:
                if alignment.hand_3d_ready:
                    overlay_text = (
                        f"sobreposicao mano 3D (z-est) | "
                        f"maos aux: {len(auxiliary_hands)}"
                    )
                elif alignment.hand_landmark_ready:
                    overlay_text = (
                        f"sobreposicao mano 2D (RMS {alignment.hand_landmark_rms:.4f} norm) | "
                        f"maos aux: {len(auxiliary_hands)}"
                    )
                else:
                    overlay_text = (
                        f"sobreposicao stereo (RMS {alignment.stereo_rms:.2f}) | "
                        f"{depth_text} | maos aux: {len(auxiliary_hands)}"
                    )
                if overlay_deviation is not None:
                    dev, dx, dy = overlay_deviation
                    overlay_text += (
                        f" | desvio vermelho-verde {dev:.0f} px "
                        f"(dx {dx:+.0f}, dy {dy:+.0f})"
                    )
            elif alignment.hand_3d_ready:
                overlay_text = (
                    f"sobreposicao mano 3D | maos aux: {len(auxiliary_hands)}"
                )
            elif alignment.hand_landmark_ready:
                overlay_text = (
                    f"sobreposicao mano (RMS {alignment.hand_landmark_rms:.4f} norm) | "
                    f"maos aux: {len(auxiliary_hands)}"
                )
            elif alignment.stereo_rms is not None:
                overlay_text = (
                    f"sobreposicao OFF: RMS {alignment.stereo_rms:.2f} px > "
                    f"{STEREO_OVERLAY_MAX_RMS_PX:.0f} (recalibre)"
                )
            else:
                overlay_text = "sobreposicao OFF: sem calibracao stereo valida"
            lines = [
                imu_text,
                camera_text,
                f"Fusao XYZ {fused[0]:.3f} {fused[1]:.3f} {fused[2]:.3f} m",
                calibration_text,
                overlay_text,
                depth_status,
            ]
            if display is not None:
                for index, text in enumerate(lines):
                    cv2.putText(display, text, (20, 35 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                display_small = cv2.resize(display, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)
                cv2.imshow("Kinect RGB + esqueleto + IMU", display_small)
                depth_view_frame = tracker.depth_view()
                if depth_view_frame is not None:
                    depth_small = cv2.resize(depth_view_frame, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)
                    cv2.imshow("Kinect depth", depth_small)
            writer.writerow([now, sample.timestamp_us if sample else "", *(camera_position if camera_position is not None else ["", "", ""]), *fused, sample.roll_deg if sample else "", sample.pitch_deg if sample else "", sample.yaw_deg if sample else "", ""])
            csv_file.flush()
            key = cv2.waitKey(1) & 0xFF
            if camera_calibrating and key == ord(" "):
                if color_for_calibration is not None and auxiliary_ok:
                    calibration_kinect = cv2.cvtColor(color_for_calibration, cv2.COLOR_BGRA2BGR)
                    if alignment.add_sample(auxiliary_for_calibration, calibration_kinect):
                        print(f"Amostra de calibracao aceita: {len(alignment.auxiliary_points)}/{CALIBRATION_SAMPLES_REQUIRED}")
                        if len(alignment.auxiliary_points) >= CALIBRATION_SAMPLES_REQUIRED:
                            alignment.solve()
                            camera_calibrating = False
                            cv2.destroyWindow("Calibracao - Kinect")
                            cv2.destroyWindow("Calibracao - camera auxiliar")
                            print(f"Calibracao salva em {CALIBRATION_FILE}")
                continue
            if camera_calibrating:
                continue
            if key == ord("c") and camera_position is not None:
                fusion.origin = camera_position.copy(); fusion.position[:] = 0.0; fusion.velocity[:] = 0.0
            elif key == ord("b"):
                fusion.bias_samples.clear()
                bias_calibration_until = time.monotonic() + 2.0
                print("Calibrando bias do acelerometro por 2 segundos. Mantenha a mao parada.")
            elif key == ord("r"):
                fusion.reset()
            elif key == ord("h"):
                hand_calib_active = not hand_calib_active
                if hand_calib_active:
                    print(
                        "Calibracao de esqueleto (man): capturando. Segure a man "
                        "visibile nas DUAS cameras e VARIE posicao/profundidade/"
                        "inclinacao; H para pausar, P para descartar."
                    )
                else:
                    print("Captura de esqueleto pausada.")
            elif key == ord("k"):
                trim_measure_until = now + 3.0
                trim_measure_samples = []
                print(
                    "Medindo trim do overlay (3 s): segure a mao PARADA e "
                    "visivel nas DUAS cameras..."
                )
            elif key == ord("l"):
                overlay_logging = not overlay_logging
                if overlay_logging:
                    overlay_log_file = open(OVERLAY_LOG_PATH, "w", newline="", encoding="utf-8")
                    header = ["t_s", "z_usado_m", "z_bruto_m", "cx_px", "cy_px",
                              "borda_0c_1b", "vel_px_s", "note"]
                    for n in range(21):
                        header += [f"n{n}_dev_px", f"n{n}_dx", f"n{n}_dy"]
                    overlay_log_file.write(",".join(header) + "\n")
                    overlay_prev_palm = None
                    overlay_prev_time = now
                    print(f"Log overlay LIGADO -> {OVERLAY_LOG_PATH.name} (1 linha/frame)")
                else:
                    if overlay_log_file is not None:
                        overlay_log_file.close()
                        overlay_log_file = None
                    print(f"Log overlay DESLIGADO (arquivo em {OVERLAY_LOG_PATH.name})")
            elif key == ord("p"):
                hand_calib_active = False
                hand_calib_samples_aux = []
                hand_calib_samples_kin = []
                hand_calib_samples_depths = []
                hand_calib_count = 0
                print("Captura de esqueleto resetada.")
            elif key == ord("t"):
                # Saude do fluxo de profundidade (o ground truth depende dele)
                stats = tracker.depth_stats()
                if stats is None:
                    print("Diagnostico depth: NENHUM frame de profundidade recebido")
                else:
                    count, dmin, dmax, frac = stats
                    print(
                        f"Diagnostico depth: {count}/{tracker.last_depth.size} px "
                        f"validos ({frac:.0%}) | min {dmin:.0f} max {dmax:.0f} mm | "
                        f"frames recebidos ate agora: {tracker.depth_frame_index}"
                    )
                # Diagnostico: erro ponta-a-ponta da sobreposicao no tabuleiro
                # (superficie rigida com correspondencia exata; separa erro de
                # projecao/calibracao de efeitos especificos da mao).
                if color is None or not auxiliary_ok:
                    print("Diagnostico: sem frame simultaneo das duas cameras")
                elif not alignment.stereo_usable:
                    print("Diagnostico: calibracao stereo nao carregada")
                else:
                    clean_color = (
                        color_for_calibration
                        if color_for_calibration is not None
                        else cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
                    )
                    if clean_color.ndim == 3 and clean_color.shape[2] == 4:
                        clean_color = cv2.cvtColor(clean_color, cv2.COLOR_BGRA2BGR)
                    board_kinect = alignment.find_board(clean_color)
                    board_aux = alignment.find_board(auxiliary_frame)
                    if board_kinect is None or board_aux is None:
                        print("Diagnostico: tabuleiro nao visto nas duas cameras")
                    else:
                        print("Diagnostico: processando tabuleiro...")
                        depths = np.full(len(board_kinect), np.nan, np.float32)
                        for index, (corner_x, corner_y) in enumerate(
                            board_kinect.reshape(-1, 2)
                        ):
                            depth_pixel = tracker._color_to_depth_pixel(
                                int(corner_x), int(corner_y)
                            )
                            if depth_pixel is None:
                                continue
                            depth_mm = tracker._median_depth(
                                *depth_pixel, radius=DEPTH_PATCH_RADIUS * 3
                            )
                            if depth_mm is not None:
                                depths[index] = depth_mm / 1000.0
                        valid = int(np.isfinite(depths).sum())
                        total = len(depths)
                        print(
                            f"Diagnostico: {valid}/{total} cantos com profundidade "
                            "valida do Kinect"
                        )
                        if depths is not None:
                            valid = np.isfinite(depths)
                            if valid.sum() < 5:
                                print(
                                    "Diagnostico: tabuleiro fora do alcance do "
                                    "depth do Kinect; afaste o tabuleiro para "
                                    ">= 0.8 m e mantenha-o dentro do FOV do "
                                    "depth (evite as bordas extremas do video)"
                                )
                            else:
                                depths = np.where(
                                    valid, depths, float(np.nanmedian(depths))
                                )
                                projected = np.asarray(
                                    alignment.auxiliary_to_kinect_stereo(
                                        board_aux, depths
                                    ),
                                    np.float64,
                                ).reshape(-1, 2)
                                differences = projected - board_kinect.reshape(-1, 2)
                                errors = np.linalg.norm(differences, axis=1)
                                finite_err = errors[np.isfinite(errors)]
                                dx = float(np.nanmean(differences[:, 0]))
                                dy = float(np.nanmean(differences[:, 1]))
                                centro = np.mean(board_kinect.reshape(-1, 2), axis=0)
                                radius = np.linalg.norm(
                                    board_kinect.reshape(-1, 2) - centro, axis=1
                                )
                                slope = None
                                if finite_err.size >= 2:
                                    r_median = max(1e-9, float(np.median(radius)))
                                    slope = float(
                                        np.polyfit(
                                            radius / r_median,
                                            finite_err,
                                            1,
                                        )[0]
                                    )
                                print(
                                    f"Diagnostico tabuleiro: {np.nanmean(errors):.1f} px media | "
                                    f"mediana {float(np.nanmedian(errors)):.0f} | max {np.nanmax(errors):.0f} | "
                                    f"centro {errors[24]:.0f} | bordas "
                                    f"{np.nanmean(np.concatenate((errors[:8], errors[-8:]))):.0f} | "
                                    f"dx {dx:.0f} dy {dy:.0f} | slope {slope if slope is not None else '--'}"
                                )
            elif key == 27:
                break
    finally:
        if overlay_log_file is not None:
            overlay_log_file.close()
        receiver.stop(); tracker.close(); auxiliary.release(); csv_file.close(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

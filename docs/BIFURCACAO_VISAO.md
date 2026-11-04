# Bifurcação controlada: normal da palma por depth vs multi-view

Objetivo: comparar, nas mesmas amostras, dois backends de visão para a normal
da palma. A decisão final é por dados, não por preferência.

## Antes de rodar
- Para multi-view é necessário ter as câmeras auxiliares calibradas
  (`stereo_calibration_aux{N}.npz`) e o `CameraAlignment` marcando
  `stereo_usable=True`.
- O KinectDepth usa os 5 pontos do dorso medidos pelo depth do Kinect.
- O MultiView usa os mesmos landmarks triangulados com a auxiliar.

## Sintético (sem hardware)
```powershell
.\venv\Scripts\python.exe tools\vision_ab.py --synthetic --duracao 5 --log-csv ab_sintetico.csv
```

## Hardware
```powershell
.\venv\Scripts\python.exe tools\vision_ab.py --aux-index 0 --ponte-serial COM4 --duracao 15 --sem-janela --log-csv ab_hardware.csv
```
Se a bridge já estiver aberta em outro terminal, omita `--ponte-serial`.

## O que comparar
- `angle_deg` entre as normais;
- `edge_on` e `sigma_deg` de cada backend;
- estabilidade temporal (jitter) e taxa de frames válidos;
- latência.

## Cuidado
O A/B não substitui validação independente. As duas fontes ainda são ópticas;
o jig A4/nível continua sendo a verdade externa.


## Status do ciclo (05/11/2026)

- Calibração estéreo aux0/aux1: aceita (RMS 1,12 px e 0,84 px).
- Overlay do groundtruth tool: **correto**, confirmado na bancada. Isso indica
  que a calibração/extrínseca está boa; a discrepância do A/B está no caminho
  `AuxStereoSource`, não no `stereo_calibration.npz`.
- A/B com mesmos 5 pontos do dorso: KinectDepth estável (confiança alta, RMS
  sub-mm), MultiView ainda com posição deslocada ~300 mm em Y e normal
  36-82 graus diferente.
- Decisão do ciclo: **KinectDepth é a fonte primária**; MultiView fica
  experimental/estacionado. Não ativar `FUSED_*` no EEG.
- Pendências para a próxima bancada:
  1. comparar `AuxStereoSource` com o overlay do groundtruth (pixels, frame,
     lateralidade, espelhamento, resolução) e achar a convenção errada;
  2. só depois repetir A/B e o jig A4/nível;
  3. então decidir V2 e integração no EEG.

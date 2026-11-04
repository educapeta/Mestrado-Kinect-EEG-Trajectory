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

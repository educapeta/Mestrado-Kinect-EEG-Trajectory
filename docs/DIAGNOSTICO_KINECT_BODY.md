# Diagnóstico do body tracking / JointOrientation (05/11/2026)

## Resultado

- `pykinect2` abriu o runtime do Kinect v2 com `Color | Depth | Body` após
  `kinect_compat`.
- Resolução: color 1920x1080; depth 512x424; `max_body_count = 6`.
- Em 8 s:
  - **color frame OK**;
  - **depth frame OK**;
  - **321 body frames recebidos**;
  - **0 corpos rastreados** (`body.is_tracked == False` em todos).
- Portanto o problema **não é o sensor color/depth nem o pykinect2**: o body
  tracking está rodando, mas não reconhece corpo na cena atual.

## Evidência visual

`docs/kinect_body_view.png` mostra o Kinect alto olhando a mesa de cima: só
aparecem cabeça, tronco superior e braços; **quadril/pernas estão fora do
quadro**. O body tracker do Kinect v2 precisa de mais corpo (tronco/quadril)
para rastrear e publicar `JointOrientation`. Nessa montagem o
`JointOrientation` **não é utilizável**.

## Consequência para o plano

- A V1 **não deve usar a orientação das juntas do Kinect como fonte independente**.
  Ela continua válida para posição/depth/landmarks do MediaPipe.
- O protótipo V1 fica: Kinect (posição/depth) + MediaPipe (landmarks/normal) +
  IMU (Mahony) + fusão em quaternion; orientação do Kinect só entra se houver
  reposicionamento que mostre o corpo inteiro.
- Para testar no futuro: afastar o Kinect/reposicionar a câmera ou o usuário
  para enquadrar tronco + quadril, e repetir
  `tools/diag_kinect_joint_orientation.py`.
- Não é defeito do sensor: color/depth funcionam 100%.

## Arquivos
- `tools/diag_kinect_joint_orientation.py` (diagnóstico color+depth+body;
  salva `_kinect_color.png` e `_kinect_depth.png`).
- `docs/kinect_body_view.png` (frame capturado).

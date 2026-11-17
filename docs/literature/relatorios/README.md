# Relatórios da Fase 1 (brutos)

Gerados por 2 subagentes NVIDIA Nemotron Ultra a partir dos PDFs em
`literatura/` (Mahony, Markley, Tian, Acharya; MediaPipe, Kusunose, Xefteris,
IJMPE, BIBE, Virtual Kinect).

## Importante: citações/páginas NÃO verificadas

Os relatórios são úteis como **síntese conceitual**, mas os agentes
**alucinaram páginas, equações e até autores**. Exemplos:
- MediaPipe Hands (PDF de 5 páginas) é citado com páginas 117–125;
- IJMPE (PDF de 4 páginas) é citado com páginas 144–175;
- Xefteris (PDF de 13 páginas) é citado com página 407;
- Markley (2002) aparece com autores extras que não estão no PDF;
- o título do Mahony foi reescrito de forma diferente do PDF.

**Regra:** antes de usar um número/equação/citação no texto da dissertação,
conferir no PDF original e corrigir a referência. A Fase 1.5 do projeto é
justamente essa auditoria.

## Auditoria

- `FASE1_5_AUDITORIA.md` -- conferência dos metadados e das afirmações centrais contra os PDFs. Use este documento, não as páginas dos relatórios brutos.

## Arquivos
- `FASE1_A1_fusao_calibracao.md` — atitude/fusão/calibração.
- `FASE1_A2_visao_mediapipe.md` — MediaPipe/yaw/fusão visão+IMU.

## O que foi aproveitado (conceito)
- Mahony 6 eixos: roll/pitch confiáveis, yaw deriva; precisa de referência
  externa de yaw.
- Markley: média de quaternions com pesos e cuidado com sinal/outliers.
- Tian: arquitetura UKF `[posição, velocidade, quaternion]` com correção por
  Kinect; adaptar, não copiar números.
- Acharya: IMU bom em tempo/relativo, Kinect bom em posição absoluta.
- MediaPipe: 21 landmarks 2.5D, `z_rel` não métrico; usar depth do Kinect.
- Kusunose: features geométricas normalizadas e ML para pronação/supinação;
  não é orientação de palma.
- Xefteris/BIBE/Virtual Kinect: fusão e sincronização; Virtual Kinect é corpo,
  não mão.

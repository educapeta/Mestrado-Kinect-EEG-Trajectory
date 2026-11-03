# Literatura do projeto -- Fase 0 (05/11/2026)

Pasta local para os PDFs e metadados das referências do plano de fusão
Kinect v2 + MediaPipe + estéreo + IMU (MPU6050/Mahony).

> Os PDFs são para uso local/estudo; não versionar no git (ver .gitignore).
> Para os paywalled, baixar pelo acesso institucional (CAPES/Periódicos) e
> salvar nesta pasta com o nome indicado.

## Baixados e verificados nesta máquina

| Arquivo | Referência | DOI/URL | Observação |
|---|---|---|---|
| `Markley_2002_Quaternion_Averaging.pdf` | Markley, F. L. (2002). Quaternion Averaging. NASA. | https://ntrs.nasa.gov/citations/20070017872 | PDF aberto; base para média de rotações |
| `Xefteris_2025_Multimodal_IMU_RGB_pose.pdf` | Xefteris et al. (2025). Multimodal fusion of inertial sensors and single RGB camera data for 3D human pose estimation... | https://zenodo.org/records/17348922 | PDF aberto (Zenodo); conferir ano/venue exato no PDF |
| `MediaPipeHands_2020_Zhang.pdf` | Zhang et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. | https://arxiv.org/abs/2006.10214 | PDF aberto; referência técnica do MediaPipe |

## Confirmados, mas PDF paywalled/ainda não baixado

| Referência | DOI/URL | Status |
|---|---|---|
| Jaruchokphuriwaj, T.; Varagul, J. (2024). Accurate Hand Pose Estimation Using Depth Camera and IMU. IJMPE 12(7), pp. 20-23. | http://ijmpe.iraj.in/paperdetails/21035 | Artigo confirmado. **Usa Intel RealSense D435 + Witmotion JY61P**, não Kinect v2. PDF não exposto como link direto na página; baixar pelo site/botão PDF no navegador. |
| Tian et al. (2015). Upper limb motion tracking with the integration of IMU and Kinect. Neurocomputing. | https://doi.org/10.1016/j.neucom.2015.01.071 | Confirmado (ACM/Elsevier), paywalled. |
| Acharya et al. (2022). A Comparative Analysis of Two Approaches for Estimation of Upper Limb Orientation Using Inertial and Kinect Sensors. | https://researcher.manipal.edu/en/publications/a-comparative-analysis-of-two-approaches-for-estimation-of-upper-/ | Confirmado o título; paywalled. |
| Real-time Joint Angle Estimation using Mediapipe Framework and Inertial Sensors. IEEE BIBE 2022. | https://www.computer.org/csdl/proceedings-article/bibe/2022/848700a128/1J6hJGkP4nS | Confirmado; paywalled. |
| The Virtual Kinect. ACM. | https://doi.org/10.1145/3488162.3488215 | Confirmado; PDF ACM retornou 403 para download automático. |
| Mahony, R.; Hamel, T.; Pflimlin, J.-M. (2008). Nonlinear Complementary Filters on the Special Orthogonal Group. IEEE TAC. | https://doi.org/10.1109/TAC.2008.923738 | Confirmado; HAL exige desafio anti-bot; baixar pelo navegador/institucional. |
| Polytechnique Montréal -- calibration framework for hand pose estimation. | (tese exata ainda não confirmada) | Candidata encontrada: https://publications.polymtl.ca/67793/ (multi-camera calibration). Confirmar se é a tese citada no plano. |

## Não confirmadas -- NÃO usar como referência ainda

| Referência do plano | Status |
|---|---|
| Kusunose et al. (2025), MediaPipe + IMU para pronação/supinação | Não encontrada em buscas rápidas; precisa DOI/PDF. Pode estar com título diferente ou ser inexistente. |
| "Virtual Kinect (ACM), mapeamento Kinect <-> MediaPipe" | Existe um paper "The Virtual Kinect" no ACM (DOI acima); confirmar se é o mesmo citado. |
| Tese Polytechnique Montréal "Calibration framework for hand pose estimation" | Confirmar autor/título/URL exata. |

## Observações críticas para a Fase 1

1. O artigo principal (Jaruchokphuriwaj & Varagul, 2024) usa **RealSense D435 + Witmotion JY61P**, não Kinect v2. Os números 2,33 mm / 0,057° / 0,45° / 0,34° são do setup deles; não adotar como meta do nosso protótipo.
2. Tian et al. (2015) é UKF com Kinect; útil para arquitetura, mas o hardware/época é outro.
3. Antes de qualquer agente "ler" um paper, o PDF precisa estar nesta pasta (ou o DOI/abstract verificado). Sem isso, o modelo pode alucinar.
4. Prioridade de leitura para o nosso caso:
   1. Markley (média de rotações) -- já baixado;
   2. Xefteris (fusão IMU+RGB) -- já baixado;
   3. MediaPipe Hands (landmarks) -- já baixado;
   4. Mahony (filtro) -- DOI confirmado;
   5. artigo principal IJMPE -- confirmar detalhes do método;
   6. Tian (UKF Kinect+IMU) e BIBE -- se conseguirmos acesso.

## Como pedir acesso
- Portal de Periódicos CAPES (DOI/Elsevier/ACM/IEEE);
- biblioteca da universidade;
- ResearchGate/arXiv/preprints dos autores (respeitando licença);
- salvar com o nome da tabela acima.

# Revisao: AHRS (firmware novo) x groundtruth Kinect

Data: 29/10/2026
Escopo: comparar a convencao/qualidade do vetor do AHRS novo
(`AHRS/firmware/esp32_teste_ahrs` + `AHRS/tools/`) com o pipeline do Kinect
(`kinect_imu_groundtruth.py`, `imu.py`, `tools/kinect_groundtruth_tool.py`).
Metodo: 4 subagentes independentes com NVIDIA Nemotron 3 Ultra (API NVIDIA),
cada um com um escopo (convencoes, tool Kinect, referencia AHRS, dados), todos
read-only. As afirmacoes abaixo foram **verificadas localmente** quando possivel;
o que e' hipotese esta' marcado.

## Fatos verificados

1. **A convencao de rotacao e' compativel.** Peguei os quaternions do log AHRS
   novo (`AHRS/logs/ahrs_20261029_152802.csv`), converti q->matriz, extrai
   roll/pitch/yaw ZYX como no `AttitudeMahony.to_euler_zyx_deg` e comparei com
   `imu.rotation_matrix(roll,pitch,yaw)`. Diferenca angular:
   **max 0,000002 grau, mediana 0,000000** (265 amostras). Ou seja, nao ha'
   divergencia de ordem w,x,y,z / ZYX / transposicao entre firmware e `imu.py`.

2. **`UP_AXIS = 2` NAO e' bug de camera space.** O `imu.py` aplica
   `R @ accel` e subtrai 1 g no eixo 2 **do mundo do IMU** (mundo Z para cima,
   que e' o referencial do `AttitudeMahony`). Teste: com `R=I` e accel=(0,0,1),
   `R @ accel = (0,0,1)` e `residual_world[2] -= 1g` zera corretamente.
   O comentario de `imu.py:36` e' enganoso ("camera space"), mas a matematica
   esta' no mundo do IMU. Nao mudar `UP_AXIS` sem um teste que prove o contrario.

3. **As setas comparadas sao grandezas fisicas diferentes.** O AHRS novo mostra
   um vetor **fixo no corpo, tareado no primeiro quadro para -Z do mundo/chao**
   (`v_corpo = R0^T . (0,0,-1)`; depois `v = R . v_corpo`). O groundtruth mostra
   `v = T . R . f` (**normal da palma no referencial da camera**). Sem aplicar
   `T`/`f` e comparar no mesmo referencial, os dois vetores nao devem coincidir
   ponto a ponto. Comparar "a olho" AHRS x groundtruth e' invalido.

4. **O lado esquerdo nao se moveu no log mais recente.** No CSV
   `ahrs_20261029_152802.csv`: direita tem saltos de aceleracao de ate
   **2,081 g** (foi ao teto: 179,1 graus de -Z, voltou a 1,8 graus); esquerda
   tem variacao total de aceleracao de apenas **~0,1 g** e angulo maximo 4,8
   graus. Ou o modulo esquerdo (0x69) nao foi o que foi movido, ou a leitura/
   fixacao/fiacao dele esta' com problema. Isso e' **hardware/leitura**, nao
   convencao de software.

## Diferencas entre os dois pipelines

| Aspecto | AHRS novo (teste) | Groundtruth Kinect |
|---|---|---|
| Saida de atitude | quaternion corpo->mundo no ESP32 | roll/pitch/yaw do firmware da luva -> `rotation_matrix` |
| Representacao | quaternion (sem gimbal lock) | Euler ZYX (singular em pitch = +-90) |
| Vetor exibido | vetor fixo no corpo, tareado a -Z mundo | `v = T R f` da palma no frame da camera |
| Verdade | tara do primeiro quadro | normal do Kinect (plano dorso) + T/f calibrados |
| Yaw | nao importa para o vetor tareado | yaw do boot e' arbitrario; precisa `reancorar_yaw` |
| Calibracao | nenhuma (tara) | O/J/X/V; T/f por lado |

## Achados priorizados (revisao)

### P0 -- lado esquerdo (hardware/leitura)
Antes de culpar o software, repetir o teste movendo **somente** o modulo 0x69
e ver no painel OpenCV se `accel L` muda. Se nao mudar, conferir fiacao/
alimentacao/AD0 desse modulo; trocar os modulos para ver se o problema segue o
sensor ou o endereco. O log acima mostra que o lado esquerdo nao mudou.

### P1 -- `AncoradorYawBoot` grava T reancorado no estado (possivel bug de save)
`imu.py` `AncoradorYawBoot.amostra` chama `fusion.set_palm_alignment(t_novo,...)`
(linhas ~466-468), que atualiza `fusion.imu_to_camera`; `save_palm_alignment`
grava `fusion.palm_alignment_state()` = **T atual**. O docstring diz que o T
corrigido vale so' para o boot e o JSON salvo fica intocado, mas se o usuario
salvar (O/J/X) depois da ancoragem, o T com o delta daquele boot pode ir para o
arquivo. Candidato forte para "funcionava e degradou depois do reboot/save".
**Verificar e corrigir** mantendo T original + delta de yaw separado.

### P1 -- bootstrap de 1 amostra (`calibrate_palm_direction`)
Sem alinhamento salvo, `imu_to_camera=I` e `f` e' ancorado numa unica pose;
isso so' vale naquela pose. Se o tool estiver usando bootstrap (sem arquivo
valido), o vetor vermelho erra ao girar a mao. O tool deveria **avisar/bloquear**
a seta vermelha nesse estado, ou exigir O/X. Nao e' um erro de conta, e' falta
de validacao de estado.

### P1 -- caminho Euler/gimbal lock
O firmware da luva envia roll/pitch/yaw; em pitch +-90 (sensor apontando para o
teto, exatamente o teste do usuario) yaw/roll ficam acoplados. O AHRS novo usa
quaternion e passa o teste do teto; o groundtruth reconstroi Euler. Mesmo que a
matriz resultante esteja correta em condicoes ideais, ruido perto da
singularidade degrada. Considerar enviar **quaternion** no UDP (o firmware ja
calcula q) e usar q direto no `imu.py`/tool.

### P2 -- sinal da normal do Kinect / lado
`palm_normal_from_plane`/`palm_normal_from_points` tem sinal anatomico por lado;
`palm_normal_camera` pode forcar nz<0, o que so' vale na pose "palma para a
lente". Nas poses de diagnostico (mesa/rezando) a comparacao pode usar a normal
com sinal inadequado. Padronizar: sempre normal anatomica, e usar
`palm_plano_estavel` para QA/validacao; nunca comparar grandezas com sinais
diferentes.

### P2 -- T/f antigos
`imu_palm_alignment_*.json` ainda sao de 22/10 (residuos 3,65/3,71). Se o
firmware/calibracao mudou desde entao (itens 124+), T/f podem estar stale.
Recalibrar depois de corrigir os itens acima.

## Plano de acao

1. **Isolar lado esquerdo** (P0): teste de aceleracao ao mover apenas 0x69;
   se falhar, e' hardware/fiacao (nao mexer no software do groundtruth).
2. **Corrigir o bug de save/reancoragem** (P1): manter no objeto o T original e
   o delta de yaw do boot; `save_palm_alignment` grava o T original; adicionar
   teste sintetico (reancorar + salvar + carregar).
3. **Bloquear bootstrap silencioso** (P1): se nao houver alinhamento salvo,
   mostrar aviso claro e nao desenhar seta vermelha como se fosse valida.
4. **Quaternion no UDP** (P1): adicionar qw,qx,qy,qz ao pacote do firmware da
   luva e usar quaternion no `imu.py`/tool; manter Euler como fallback.
5. **Normal do Kinect** (P2): padronizar sinal anatomico e usar plano estabilizado
   no QA; criar teste com normais simuladas direita/esquerda.
6. **Recalibrar** T/f (J/X) depois das correcoes e rodar V/diagnostico.
7. **Teste controlado**: com o mesmo IMU, registrar q e normal do Kinect na
   mesma sessao; comparar `T R f` com a normal no mesmo referencial (nao a olho).

## Observacao sobre os revisores

Os agentes acertaram em apontar os temas (save/reancoragem, bootstrap, Euler,
normal), mas dois "criticos" foram **falsos positivos**: `UP_AXIS=2` e
"rotation_matrix duplica a rotacao". A matematica q->matriz->`rotation_matrix`
foi verificada numericamente (erro ~2e-6 grau). Nao aplicar essas duas
"correcoes" sem teste.

## Arquivos relevantes
- `imu.py`: `rotation_matrix` (131-141), `solve_palm_alignment*` (~202-297),
  `reancorar_yaw` (300-378), `AncoradorYawBoot` (381-487),
  `calibrate_palm_direction` (554-577), `set_palm_alignment` (601-629),
  `palm_alignment_state` (631-640).
- `tools/kinect_groundtruth_tool.py`: `save_palm_alignment` (389-423),
  carregamento/ancoragem (~2295-2312), overlay (~3480+).
- `kinect_imu_groundtruth.py`: `palm_normal_from_points` (~4904),
  `palm_normal_from_plane` (~1341-1454).

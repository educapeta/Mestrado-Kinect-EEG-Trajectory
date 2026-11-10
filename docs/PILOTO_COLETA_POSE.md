# Piloto de coleta — pose da mão para EEG

> Objetivo do piloto: descobrir se existe sinal suficiente para decodificar
> pose 6D (posição + orientação) com EEG e labels de visão multi-câmera.
> **Não** é para provar controle de mão robótica ainda. É para validar o
> dataset, a sincronização e o pipeline de labels.

## 0. Escopo mínimo

| Item | Piloto |
| --- | --- |
| Sujeitos | 1 (você mesmo) |
| Sessões | 3, em dias diferentes |
| Mão | direita primeiro; esquerda depois se houver tempo |
| Câmeras | 2 calibradas (as auxiliares atuais) |
| Kinect | opcional; usar só RGB se ajudar no MediaPipe |
| Depth | **não usar** no piloto (labels por triangulação) |
| IMU | gravar bruto, mas **não usar** no modelo do piloto |
| EEG | montagem atual, 32 canais, 500 Hz |
| Duração | 15–20 min por sessão |

O piloto tem três metas:

1. os streams ficam sincronizados?
2. os labels de pose são confiáveis (reprojeção baixa, quaternion válido)?
3. o SAND-Pose aprende algo acima do acaso?

## 1. Preparação física

- Câmeras fixas, sem tocar durante a sessão.
- Sujeito sentado, tronco encostado, ombro na mesma posição nas 3 sessões.
- Marcar no chão a posição da cadeira e a altura da mesa.
- Luvas/IMU bem presas, mas o piloto **não depende** de T/f.
- Iluminação constante; evitar janela/sol direto nas mãos.
- Fundo com contraste para as mãos (evitar parede da mesma cor da pele).

## 2. Definição do frame e pose de repouso

Frame `base`:

- origem no esterno (ou ponto médio dos ombros);
- `x` direita, `y` cima, `z` frente;
- registrar a definição em `meta.yaml`.

Pose de repouso do piloto:

- sentado, braços apoiados na mesa;
- antebraços apoiados, mãos relaxadas, palmas para baixo;
- ombros relaxados, sem contrair;
- gravar **10 s no início e 10 s no fim** da sessão.

Não use “braços pendurados” no piloto: é mais difícil repetir a posição e
gera mais EMG/artefato. Mesa é mais estável. Se depois o alvo for outra pose,
refaça o piloto com ela — não misture as duas no mesmo dataset.

## 3. Calibração das câmeras (antes da sessão)

1. Calibrar o par estéreo (ou multi-view) com o tabuleiro.
2. Salvar o `.npz` e registrar o nome em `meta.yaml`.
3. Conferir reprojeção: mediana < 5 px; se pior, recalibrar.
4. Conferir que as duas câmeras veem a mão na região de trabalho.

Sem calibração boa, os labels de pose ficam errados e o modelo aprende ruído.

## 4. Roteiro da sessão

### Bloco A — repouso (20 s)

- 10 s no início: mãos na mesa, parado.
- 10 s no fim: repetir a mesma pose.

### Bloco B — poses estáticas (6 poses × 5 s)

Manter a mão parada em cada pose, com a palma/dorso visível para as câmeras:

1. palma para baixo (mesa);
2. palma para cima;
3. palma para a câmera;
4. palma para a esquerda;
5. palma para a direita;
6. punho girado 90° (roll), mantendo o antebraço apoiado.

Fazer as duas mãos, uma por vez. Entre poses, voltar ao repouso por 2 s.

### Bloco C — movimentos dinâmicos (10 trials)

Cada trial de 6 s:

1. repouso (1 s);
2. alcançar um ponto à frente (1 s);
3. girar a mão (roll) devagar (1,5 s);
4. manter (1 s);
5. voltar ao repouso (1,5 s).

Variações: 3 alturas, 3 distâncias laterais, 2 rotações. Não precisa ser
aleatório; o importante é cobrir o espaço de posição e orientação.

### Bloco D — repouso final (10 s)

## 5. O que gravar em cada trial

- `trial_id`;
- instrução;
- `t_start_us` / `t_end_us`;
- mão;
- evento de início/fim (LED/teclado/serial);
- observações (ex.: “mão saiu do quadro”, “câmera tremida”).

## 6. Checklist imediatamente após a sessão

Rodar/verificar:

- [ ] EEG: 32 canais, sem canal morto, `nan_frac < 1%`;
- [ ] câmeras: mão visível em ≥80% dos frames nas duas vistas;
- [ ] MediaPipe: confiança mediana > 0,9;
- [ ] triangulação: `n_views >= 2` em ≥80% dos frames;
- [ ] `pose_labels`: `pct_with_quat >= 0,80`;
- [ ] reprojeção mediana ≤ 5 px;
- [ ] IMU: ~100 Hz nos dois lados, `valid > 0,99` (só gravar);
- [ ] sincronização: offset máximo ≤ 10 ms;
- [ ] `manifest.csv` atualizado;
- [ ] backup da sessão antes de fechar.

Se algum item falhar, **não descarte**: anote em `quality_report.json` e
repita a condição na próxima sessão.

## 7. Erros comuns que invalidam o piloto

| Erro | Consequência |
| --- | --- |
| mexer a câmera no meio | labels 3D errados em parte da sessão |
| mão fora de uma das vistas | sem triangulação; buracos no alvo |
| luz mudando | MediaPipe instável |
| pose de repouso diferente entre sessões | frame inconsistente; modelo aprende offset |
| eventos sem timestamp | janela de EEG desalinhada |
| usar `palm_normal` sem saber a montagem | quaternion errado |
| misturar mão esquerda/direita no mesmo label | frame anatômico trocado |
| treinar e validar na mesma sessão | resultado inflado |

## 8. Critério para continuar

Depois de 3 sessões, rode o baseline do SAND-Pose.

- Se a **posição** já sai razoável e a rotação é ruim: o problema é o label de
  orientação ou a representação; investigar quaternion/6D.
- Se **posição e rotação** são ruins: primeiro checar sincronização/qualidade
  dos labels; depois aumentar dados; por último mexer na arquitetura.
- Se só o **IMU** melhora: guarde o IMU para a etapa seguinte; não trave o
  piloto por causa de T/f.

## 9. O que o piloto **não** responde

- se EEG controla uma mão robótica real;
- se o modelo generaliza para novos sujeitos (precisa LOSO);
- se o IMU-only substitui a visão;
- se a pose de repouso na mesa é a melhor para uso clínico.

Essas perguntas vêm depois; o piloto só valida o pipeline e o sinal.

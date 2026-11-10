# Emenda ao Projeto de Pesquisa — CEP/UFSC

## Protocolo de decodificação de pose da mão a partir de EEG: execução motora real como referência e imagética motora subsequente

> **Status deste documento:** ESBOÇO para discussão com o orientador antes da submissão.
> Campos entre `[PREENCHER]` dependem de dados do projeto original (número do parecer,
> CAAE, nome do orientador) e devem ser completados pelo pesquisador responsável.
>
> **Versão:** rascunho v0.1 — 2026-11-10
> **Documento de referência (aprovado):** projeto "Interface Cérebro-Máquina baseada em
> EEG para controle de prótese de mão" — Parecer CEP/UFSC nº `[PREENCHER]`, CAAE
> `[PREENCHER]`, aprovado em `[PREENCHER]`.
> **Pesquisador responsável:** `[PREENCHER]` · **Orientador:** `[PREENCHER]`
> **Instituição:** Universidade Federal de Santa Catarina — Instituto de Engenharia
> Biomédica (IEB/UFSC) e PsiCowLab.

---

## ⚠️ Aviso — vedação de coleta antes da aprovação

**Nenhum participante pode ser recrutado, convidado, agendado ou submetido a qualquer
procedimento deste protocolo novo antes da aprovação desta emenda pelo CEP/UFSC.**

A aprovação anterior cobre **apenas** o protocolo de imagética motora (MI) sem execução
motora real, sem câmeras e sem registro de pose. Ela **não** autoriza:

- execução motora real repetitiva;
- captura de vídeo dos membros superiores dos participantes;
- uso de sensores inerciais (IMU) no punho/mão;
- visitas de duração ampliada;
- novos desfechos de pose, orientação e abertura da mão.

Caso a emenda seja aprovada com exigências (parecer consubstanciado com pendências), a
coleta só se inicia após o cumprimento integral das pendências e a emissão do parecer
final de aprovação.

> **Observação sobre dados já existentes:** se houver qualquer dado de piloto já
> registrado com o próprio pesquisador (autoexperimentação) ou com terceiros antes desta
> aprovação, isso deve ser declarado ao CEP na carta de encaminhamento e **não** pode ser
> usado como dado do estudo sem orientação expressa do comitê.

---

## 1. Sumário executivo da emenda

| Item | Projeto aprovado (v1) | Emenda proposta (v2) |
| --- | --- | --- |
| Paradigma | apenas imagética motora (MI) | **execução motora real (ME) + MI após a execução**, na mesma trial |
| Nº de participantes | 5 | **5 (máximo 6)** |
| Nº de sessões | 6 | **2** |
| Nº de trials | 48 por sessão (288/participante) | **125 por sessão (250/participante)** |
| Duração do trial | 9 s | **12 s** |
| Alvo de decodificação | classe (esquerda/direita/repouso) | **pose contínua da mão** (posição + orientação + abertura), com classificação como tarefa secundária |
| Sensores | EEG 32 canais (g.Nautilus) + EMG de superfície | EEG 32 canais + **2 ou mais câmeras calibradas (multi-view)** + **IMU bruto (opcional)** + EMG de superfície |
| Ground truth | rótulo de classe da pista visual | **pose medida por visão computacional multi-view**, triangulada e calibrada |
| Duração da visita | ~1 h | **~1 h 25 min a 1 h 45 min** (participante), com pausas e limites definidos na seção 3.9 |
| Desfechos principais | acurácia, sensibilidade, especificidade, kappa, ITR | **erro de pose final/hold** (posição, orientação, abertura) + desfechos secundários de classificação |
| Riscos novos | — | fadiga neuromuscular por movimento repetitivo, desconforto osteomuscular, captura de imagem/vídeo (privacidade), sensor inercial em contato com a pele |
| TCLE | versão 1 | **versão 2**, com seções novas de execução motora, vídeo, IMU, guarda e descarte |
| Guarda de dados | EEG e respostas comportamentais | + **vídeo bruto**, landmarks, pose derivada, IMU bruto — com regras próprias de acesso, retenção e descarte |

**O que a emenda não altera:** local de coleta, faixa etária (18–65 anos), critérios gerais
de inclusão/exclusão, caráter não invasivo do estudo, equipamento de EEG e sistema de
aquisição, natureza exploratória e tamanho amostral reduzido.

---

## 2. Justificativa da emenda

### 2.1. Limitação científica do protocolo aprovado

O projeto aprovado investiga a classificação de imagética motora (mão direita, mão
esquerda e repouso) a partir de EEG. Esse desenho responde a uma pergunta discreta
("qual classe?") e é adequado para uma prova de conceito de ICM. Ele não permite,
contudo, responder à pergunta que sustenta a aplicação pretendida — **qual pose a mão
deve assumir** para preensão de um objeto.

Duas consequências práticas motivam a emenda:

1. **Sistema de controle com três classes não gera movimento funcional.** Para acionar
   uma prótese de mão em tarefas de alcance e preensão é necessário um alvo contínuo
   (posição, orientação e abertura), e não apenas um rótulo de "abrir/fechar".
2. **A MI não fornece rótulo de pose.** A imagética não é observável por vídeo nem por
   sensor inercial; portanto, sem uma referência externa, não há como supervisionar um
   modelo de regressão de pose a partir de MI.

### 2.2. Estratégia científica adotada: execução como referência, MI como alvo de uso

A literatura recente em decodificação de trajetória e cinemática a partir de EEG mostra
que é viável reconstruir movimento contínuo quando existe uma referência cinemática
durante o treinamento — por exemplo, transformers de trajetória (MTRT, Wang et al., 2023,
`wang2023mtrt`), predição de trajetória em MI a partir da execução (E2T, Kwon et al.,
2026, `kwon2026e2t`), atenção multi-escala temporal (M3T-attention, Zhu et al., 2026,
`m3t2026`) e decodificação de cinemática da mão a partir de dinâmica espectral de baixa
frequência (SAND, Fu et al., 2026, `fu2026sand`). Revisões da área sistematizam essas
estratégias e suas limitações (`reviewtraj2023`).

O protocolo proposto adota a lógica:

```text
treino:  EEG da EXECUÇÃO motora + pose medida (ground truth por câmeras)
teste:   EEG da IMAGÉTICA (MI)  →  pose prevista, sem pose medida disponível
```

Isto é: a execução fornece os rótulos que a MI não tem; o uso final continua sendo a MI.
A hipótese de trabalho é que essa representação **transfira parcialmente** para a MI. A
emenda existe para permitir medir essa transferência — e não para afirmá-la como
resultado.

Fundamentos adicionais que justificam parear execução e MI na mesma trial:

- a MI cinestésica, e não a visual, modula a excitabilidade corticomotora (`Stinear2005`),
  o que sustenta a instrução de imagética cinestésica já usada no projeto;
- a imagética de movimento voluntário ativa representações motoras somatotópicas
  correspondentes à parte do corpo (`imageryvoluntary2003`), o que fundamenta o pareamento
  entre lado instruído e padrão cortical esperado;
- o paradigma de preensão e elevação com variação de objeto tem precedente metodológico
  consolidado em bases públicas de EEG (WAY-EEG-GAL, Luciw et al., 2014, `luciw2014way`);
- o uso de estímulo visual esquemático (e não vídeo de mão humana) reduz confundidores
  atencionais e visuais durante a preparação da MI.

### 2.3. Justificativa metodológica da emenda (e não de um novo projeto)

A alteração **não** substitui o objeto do estudo: mantém EEG não invasivo, participantes
saudáveis, caráter exploratório e a mesma infraestrutura. O que muda é o paradigma
experimental, a instrumentação de referência (câmeras) e, consequentemente, os riscos e
as obrigações de proteção de dados. Por isso a via adequada é **emenda**, e não novo
projeto: preserva-se o histórico de avaliação ética e a continuidade do vínculo
institucional.

---

## 3. Descrição do protocolo novo

### 3.1. Delineamento

Estudo piloto, experimental, longitudinal, intra-sujeito, com **2 sessões por
participante** e amostra de conveniência de **5 participantes (máximo 6)**, saudáveis,
maiores de 18 anos. Não há grupo controle e não há intervenção terapêutica; não se trata
de ensaio clínico e não há randomização entre condições de tratamento — a randomização
existe apenas para a **ordem dos estímulos dentro da sessão**.

Local: sala 102 do segundo andar do Instituto de Engenharia Biomédica (IEB/UFSC),
Campus Universitário — Trindade, Florianópolis/SC. A mesma sala é usada nas duas
sessões, com marcação no piso da posição da cadeira e da mesa para reprodutibilidade.

### 3.2. Participantes

Mantêm-se os critérios do projeto aprovado, com **um acréscimo de exclusão** motivado
pela execução motora real.

**Inclusão:** adultos de 18 a 65 anos; destros ou ambidestros; sem experiência prévia com
ICM; capazes de ler, compreender e assinar o TCLE; **capazes de executar alcance,
preensão e elevação de objetos leves com ambas as mãos, sem dor**.

**Exclusão (mantidas):** estágios severos de doenças e infecções; doenças neurológicas não
estabilizadas (epilepsia, AVC, traumatismo craniano); lesões no couro cabeludo; déficit
visual ou auditivo severo; alergia a gel condutor ou a produtos de limpeza derivados de
álcool; pessoas com marca-passo ou implantes metálicos (restrição do equipamento de EEG).

**Exclusão (nova):** dor ou lesão osteomuscular em atividade no membro superior
(ombro, cotovelo, punho ou mão); tendinopatia ou síndrome do túnel do carpo em crise;
neuropatia periférica diagnosticada; limitação de amplitude de movimento que impeça a
tarefa; gestação de risco ou qualquer condição em que a permanência sentada por ~1 h 30
min seja contraindicada por avaliação médica; uso de medicamento que altere o controle
motor fino, a critério do pesquisador.

**Recrutamento:** convite presencial e por e-mail entre alunos e colaboradores do IEB/UFSC
e do PsiCowLab. Não serão usados cartazes, redes sociais nem bancos de dados de pacientes.

**Orientações prévias (mantidas e ampliadas):** dormir ao menos 8 h na noite anterior;
abster-se de estimulantes (cafeína, nicotina, energéticos) por 4 h; não realizar exercício
físico de força para membros superiores nas 24 h anteriores; comparecer com roupa que
permita expor antebraços e mãos; remover anéis, relógios e pulseiras; lavar o cabelo com
shampoo neutro no dia anterior; não usar creme ou óleo nas mãos no dia da sessão (afeta a
fixação do IMU e a detecção por visão).

### 3.3. Estrutura das sessões

| Item | Sessão 1 | Sessão 2 | Total por participante |
| --- | --- | --- | --- |
| Trials | 125 | 125 | **250** |
| Blocos | 5 blocos de 25 trials | 5 blocos de 25 trials | 10 blocos |
| Duração dos trials | 125 × 12 s = 25 min | 125 × 12 s = 25 min | 50 min |
| Pausas entre blocos | 4 pausas de 3 min | 4 pausas de 3 min | 24 min |
| Mãos | ambas | ambas | — |
| Intervalo entre sessões | — | ≥ 48 h, idealmente 7 dias, mesma faixa horária | — |

Justificativa da redução de 6 para 2 sessões e do aumento de 48 para 125 trials por
sessão: (i) o trial novo (12 s) é mais longo que o original (9 s); (ii) o protocolo exige
duas montagens de câmeras calibradas por sessão, o que torna sessões numerosas
dispendiosas e menos padronizadas; (iii) o total de trials por participante permanece na
ordem de 250, valor compatível com o planejamento estatístico da seção 8.

### 3.4. Estrutura do trial (12 s)

| Fase | Duração | O que ocorre | O que é registrado |
| --- | --- | --- | --- |
| `home` (repouso) | 2 s | mão apoiada na posição de repouso; tela neutra | EEG, vídeo, IMU; `rest_start` |
| `cue` + execução motora (ME) | 4 s | 1 s de pista visual (seta/lado + **animação esquemática** da trajetória-alvo) seguida de ~3 s de execução guiada: alcance → preensão → elevação → manutenção curta → retorno | EEG, vídeo multi-view, IMU, EMG; `cue_onset`, `movement_onset`, `hold_onset`, `return_onset` |
| Intervalo + animação | 2 s | tela neutra com reapresentação **estática** da animação esquemática (priming); retorno à pose de repouso | EEG; `animation_start`, `animation_end` |
| Imagética motora (MI) | 4 s | mesma instrução, **sem execução**; imagética cinestésica do trajeto até a pose-alvo | EEG, vídeo, EMG (verificação de ausência de movimento); `mi_onset`, `mi_end` |
| **Total** | **12 s** | — | — |

Notas de implementação:

- a divisão exata entre `cue` e execução dentro do bloco de 4 s segue o script
  `eeg_motor_paradigm.py`; qualquer ajuste deve ser documentado no `meta.yaml` da sessão;
- o `pausa + vídeo 2 s` do script original é o intervalo de priming; o vídeo de mão humana
  é **substituído por animação esquemática** (fundo neutro, esfera/caixa de alvo, cursor
  percorrendo o caminho, sem mão, sem textura, sem detalhes anatômicos);
- a fase de MI é verificada por EMG de superfície nos extensores do antebraço e por
  inspeção de vídeo; trials com contração visível são anotados e excluídos da análise de
  MI (mas não da análise de ME);
- os quatro segundos de MI geram uma única pose-alvo (pose final/hold da ME precedente) —
  não se assume que a MI replique a velocidade ou a duração da execução.

### 3.5. Tarefa, objetos e randomização

- 3 objetos distintos (massa entre ~100 g e ~400 g; tamanhos e geometrias diferentes),
  posicionados em 3 alvos fixos sobre a mesa, sempre na mesma posição ao longo do estudo;
- mão **direita** e **esquerda**, alternadas em blocos ou pseudo-aleatorizadas, conforme
  versão final do script;
- sequência por trial: alcançar → preensão → elevar → manter → retornar ao repouso;
- randomização de objeto, lado e ordem dentro de cada bloco, com semente registrada;
- **não** se manipulam peso e fricção como no WAY-EEG-GAL: o interesse é pose, não força;
- restrição de velocidade: o participante é instruído a executar em ritmo confortável e
  constante, sem pressa, e o ritmo é delimitado pela animação — mas sem exigir
  sincronização milimétrica, para não introduzir esforço atencional excessivo.

### 3.6. Instrumentação

| Sistema | Especificação | Observações |
| --- | --- | --- |
| EEG | g.Nautilus Multi-Purpose 32 canais (g.tec), certificado IEC 60601-1, bateria interna | mesmo equipamento e montagem do projeto aprovado; 500 Hz; gel condutor hipoalergênico |
| EMG | eletrodos de superfície nos extensores do antebraço, bilateral | mantido do projeto aprovado; verifica ausência de contração na MI |
| Câmeras | **2 ou mais** câmeras RGB calibradas (intrínsecos e extrínsecos), fixas, com vista do antebraço/mão | calibração verificada no início de cada sessão (erro de reprojeção mediano ≤ 5 px); Kinect opcional, sem uso de profundidade como rótulo principal |
| IMU | 1–2 unidades inerciais (acelerômetro + giroscópio, quaternion), ~100 Hz, fixadas no punho/dorso da mão | **opcional**; gravação **bruta**; sem uso online; sem calibração T/f com câmera |
| Sincronização | mesmo PC para EEG, câmeras e paradigma; marcador de evento por hardware/LED; `sync.csv` com desvio máximo aceitável de 10 ms | — |
| Estímulo | monitor único, fundo neutro, animação esquemática | sem vídeo de mão humana |

### 3.7. Enquadramento das câmeras e privacidade de imagem (medida técnica)

- as câmeras são posicionadas e focalizadas **exclusivamente** na região de trabalho:
  antebraços, mãos e objetos sobre a mesa;
- o enquadramento **exclui o rosto e o tronco** do participante; ajuste verificado com o
  participante presente, antes de iniciar a gravação;
- não há gravação de áudio;
- o vídeo bruto é processado localmente, sem envio para serviços em nuvem ou APIs
  externas, e pode ser **descartado após a extração dos rótulos de pose** a pedido do
  participante (mantendo-se apenas landmarks e pose derivada — ver seção 7).

### 3.8. Desfechos

**Desfecho primário**

1. Erro de decodificação da **pose final/hold** da mão: erro de posição (mm), erro de
   orientação (graus, ângulo geodésico) e erro de abertura (mm ou % da amplitude).

**Desfechos secundários**

2. Transferência execução → MI: desempenho do modelo treinado em ME e avaliado em MI
   (zero-shot) e após ajuste fino com poucas trials de MI.
3. Classificação mão direita / esquerda / repouso nas janelas de MI (acurácia, kappa,
   ITR), como tarefa secundária e comparável à literatura.
4. Desempenho intra e inter-sujeito (LOSO), com e sem calibração curta.

**Desfechos exploratórios**

5. **Velocidade linear 3D** (e, se disponível, angular) como alvo auxiliar de consistência
   temporal/suavidade — não como alvo principal.
6. Correlação entre métricas de EEG (ERD/ERS em μ e β) e erro de pose.
7. Contribuição do IMU bruto como entrada auxiliar (análise apenas offline).
8. Qualidade dos rótulos de visão (erro de reprojeção, cobertura de frames com 2+ vistas).

**Não desfechos (explicitamente fora de escopo):** controle online de prótese em tempo
real, generalização para pacientes, decodificação independente de dedos, avaliação
clínica ou terapêutica.

### 3.9. Duração da visita, pausas e monitoramento

Estimativa por sessão:

```text
Preparação antes da chegada do participante
  montagem e calibração das câmeras + verificação de sync         15–25 min

Com o participante presente
  acolhimento, releitura do TCLE e verificação das orientações     10 min
  montagem da touca, gel e verificação de impedância               20–30 min
  baseline de repouso (olhos abertos/fechados, conforme script)     5 min
  treino/prática (sem registro, ~10 trials)                         3–5 min
  125 trials × 12 s                                                25 min
  4 pausas de 3 min                                                12 min
  desmontagem + limpeza do couro cabeludo                          10–15 min
  ----------------------------------------------------------------------
  permanência do participante                                  ~85–100 min
  tempo total de laboratório                                  ~100–130 min
```

O tempo de MI efetivo é de ~8,3 min por sessão (125 × 4 s) e ~16,7 min por participante;
o restante é execução, repouso e pausa. O protocolo **aumenta a duração da visita** em
relação ao projeto aprovado (~1 h), o que constitui risco novo de fadiga (seção 5).

Regras de proteção contra fadiga, fixadas antes do início:

- pausa obrigatória de 3 min a cada bloco de 25 trials (~5 min);
- pausa **extra imediata** a pedido do participante, sem necessidade de justificativa;
- interrupção da sessão se houver dor, câimbra, formigamento, tontura, náusea, fadiga
  excessiva ou queda perceptível de desempenho;
- registro de fadiga autorreferida (escala 0–10) no início, no meio e no fim da sessão;
- hidratação e acesso a banheiro entre blocos;
- reassentamento e reposicionamento do participante após pausas, com reconferência da
  posição da cadeira e da mão de repouso.

---

## 4. Mudanças em relação ao projeto aprovado

| # | Dimensão | v1 (aprovado) | v2 (emenda) | Impacto ético |
| --- | --- | --- | --- | --- |
| 1 | Tarefa do participante | imaginar movimento, sem executar | **executar** alcance/preensão/elevação e, em seguida, imaginar | novo risco físico (fadiga, esforço repetitivo); exige nova seção de riscos e TCLE |
| 2 | Nº de sessões / trials | 6 sessões × 48 trials | **2 sessões × 125 trials** | menos visitas, visitas mais longas: muda o perfil de fadiga e de exposição |
| 3 | Duração do trial | 9 s | **12 s** | sessão mais longa; recalcular pausas |
| 4 | Duração da visita | ~1 h | **~1 h 25 a 1 h 45** | exige pausas, monitoramento e critérios de interrupção |
| 5 | Vídeo | não havia | **2+ câmeras RGB calibradas, multi-view** | dado pessoal novo, sensível do ponto de vista de privacidade; exige TCLE específico, enquadramento restrito e plano de descarte |
| 6 | Ground truth | rótulo de classe | **pose medida por triangulação multi-view** | aumenta a complexidade técnica e o volume de dados derivados |
| 7 | Sensores adicionais | EEG + EMG | **+ IMU bruto no punho/mão** (opcional) | contato com a pele, fixação, mínimo risco de irritação |
| 8 | Estímulo visual | seta/círculo | seta/círculo + **animação esquemática** de trajetória | sem risco novo; reduz confundidor visual |
| 9 | Desfechos | classificação | **pose final/hold** (primário) + classificação (secundário) | TCLE e devolutiva precisam descrever os novos desfechos |
| 10 | Critérios de exclusão | neurológicos, dermatológicos, implantes | **+ osteomusculares do membro superior e limitação de movimento** | proteção direta contra risco físico novo |
| 11 | Guarda de dados | EEG e respostas, anonimizados, 5 anos | **+ vídeo bruto, landmarks, pose, IMU**, com regras próprias de acesso e descarte | exige atualização da seção de LGPD e do TCLE |
| 12 | Orçamento | R$ 700 | **acréscimo estimado de R$ 600 a R$ 1.200** (mídia de armazenamento, gel adicional, lanches/transporte em 2 visitas, impressão de TCLE) — `[PREENCHER]` com valores reais | sem risco ético; exige atualização orçamentária |
| 13 | Cronograma | aprovação jul/2026, coleta ago/2026 | **reprogramado** (seção 9) | as datas originais já foram ultrapassadas; a coleta passa a depender desta aprovação |

**O que permanece inalterado:** local, equipamento de EEG, faixa etária, forma de
recrutamento (convite direto, sem cartazes), ausência de benefício direto garantido,
devolutiva de QEEG, caráter não invasivo, financiamento institucional.

---

## 5. Riscos e benefícios

### 5.1. Classificação de risco

O projeto permanece classificado como de **risco mínimo** nos termos da Resolução CNS
nº 466/2012 e, no que couber, da Resolução CNS nº 510/2016, considerando que: (i) não há
intervenção terapêutica; (ii) não há administração de fármacos, contraste ou radiação; (iii) os procedimentos são não
invasivos e reversíveis; (iv) os desconfortos esperados são leves, transitórios e
autolimitados. A execução motora real, porém, **acrescenta riscos físicos de natureza
musculoesquelética** que não existiam no protocolo aprovado e que passam a ser
descritos explicitamente.

### 5.2. Riscos novos e medidas de mitigação

| Risco novo | Probabilidade | Gravidade | Mitigação |
| --- | --- | --- | --- |
| **Fadiga muscular** de antebraço, mão e ombro por movimento repetitivo (125 trials/sessão) | Alta | Leve | blocos de 25 trials, pausas de 3 min, pausa extra a pedido, escala de fadiga 0–10, objetos leves (≤ ~400 g), interrupção se dor |
| **Desconforto ou dor osteomuscular** transitória (punho, cotovelo, ombro) | Moderada | Leve a moderada | exclusão de quem tem dor/lesão ativa; execução em ritmo confortável; alongamento na pausa; encerramento da sessão em caso de dor |
| **Fadiga mental/atencional** por sessão longa com dupla tarefa (executar e imaginar) | Alta | Leve | pausas, ambiente silencioso, instruções curtas, monitoramento contínuo, encerramento a pedido |
| **Desconforto dermatológico** pela fixação do IMU (fita/velcro) | Baixa | Leve | fixação sobre pele íntegra, sem adesivo sobre pelos, retirada imediata em caso de irritação |
| **Irritação por gel condutor** | Baixa | Leve | mantida a conduta do v1: interrupção, remoção com água morna, encaminhamento se necessário |
| **Perda de privacidade por captura de imagem** | Baixa (mitigada) | Moderada (se ocorrer) | enquadramento sem rosto/tronco, verificação presencial, gravação apenas da região de trabalho, sem áudio, processamento local, opção de descarte do vídeo bruto |
| **Constrangimento por erro de execução ou dificuldade com a tarefa** | Moderada | Leve | instrução explícita de que não há desempenho "certo"; treino prévio; ausência de avaliação individual publicada |
| **Tontura ou desconforto postural** pela permanência sentada e pela movimentação do braço | Baixa | Leve | pausas para levantar, cadeira com apoio, monitoramento |
| **Cansaço visual** pela animação repetida | Baixa | Leve | luminosidade ajustável, fundo neutro, pausas |
| **Falha de calibração/roteiro** que exija repetir a sessão | Moderada | Nenhuma (risco ao dado, não ao participante) | checklist pré-sessão (seção 11), conferência de reprojeção e sincronização antes de iniciar |

### 5.3. Riscos mantidos do projeto aprovado

Nervosismo/constrangimento em situação de avaliação; tédio; incômodo com a touca e o gel;
reação alérgica local ao gel (rara); riscos elétricos inexistentes (equipamento
certificado IEC 60601-1, alimentação isolada por bateria interna).

### 5.4. Medidas gerais de proteção

- todos os procedimentos conduzidos por pesquisador treinado, com no mínimo um segundo
  membro da equipe presente;
- sala silenciosa, iluminação e temperatura controladas, cadeira confortável;
- monitoramento contínuo de sinais de cansaço, dor ou desconforto;
- participante pode solicitar pausa ou encerrar a participação **a qualquer momento**, sem
  justificativa e sem prejuízo;
- em caso de evento adverso: interrupção imediata, registro em formulário próprio,
  comunicação ao CEP conforme fluxo institucional, e encaminhamento para avaliação médica
  sem custo ao participante, quando indicado;
- nenhum procedimento é realizado sem TCLE assinado na versão vigente e aprovada.

### 5.5. Benefícios

**Individuais:** não há benefício direto garantido. O participante recebe devolutiva com
seu eletroencefalograma quantitativo (QEEG) analisado por profissional e, se desejar,
resumo dos resultados agregados do estudo.

**Coletivos:** o protocolo gera o primeiro conjunto de dados brasileiro, em nosso
conhecimento, com EEG de 32 canais pareado a pose 6D da mão medida por visão multi-view
durante execução motora e MI — base necessária para o desenvolvimento de decodificadores
de pose que possam, no futuro, comandar próteses de mão de forma mais natural que os
esquemas discretos de abrir/fechar.

---

## 6. TCLE — o que precisa ser acrescentado

O TCLE deve ser reemitido como **versão 2**, com nova data e novo campo de rubrica por
página. Recomenda-se manter a estrutura do v1 e inserir os blocos abaixo.

### 6.1. Novos parágrafos obrigatórios

**(a) Execução motora real.** Texto sugerido:

> Nesta etapa do estudo, diferente da etapa anterior, você **vai realizar movimentos
> reais** com as mãos e os braços: alcançar objetos leves sobre a mesa, segurá-los,
> elevá-los por alguns instantes e devolvê-los à posição inicial. Em seguida, você vai
> **imaginar** o mesmo movimento, sem executá-lo. Você fará isso cerca de 125 vezes por
> sessão, em 5 blocos, com pausas de 3 minutos entre eles, em 2 sessões. A tarefa é
> simples e não exige esforço intenso, mas pode cansar o braço e a mão.

**(b) Fadiga e desconforto físico.** Texto sugerido:

> Os movimentos são leves (objetos de até cerca de 400 g), mas são repetidos muitas
> vezes. Por isso, é possível que você sinta cansaço muscular, desconforto ou dor leve no
> antebraço, na mão, no cotovelo ou no ombro, que costuma passar com repouso. Se sentir
> dor, câimbra, formigamento ou cansaço excessivo, avise imediatamente: a sessão será
> pausada ou encerrada, e você poderá retomar em outro dia, se quiser. Você pode pedir
> pausa a qualquer momento, sem precisar justificar.

**(c) Vídeo e imagem.** Texto sugerido:

> Para medir a posição da sua mão durante os movimentos, usaremos **duas ou mais câmeras**
> posicionadas à frente da mesa. As câmeras são ajustadas para mostrar **apenas seus
> antebraços, mãos e os objetos**, e **não** o seu rosto, a sua roupa ou outras partes do
> corpo; não há gravação de áudio. As imagens são processadas apenas para calcular a
> posição e a orientação da sua mão. Você pode verificar o enquadramento antes do início.
> Se você preferir, o vídeo bruto pode ser **apagado após o cálculo dessas medidas**,
> mantendo-se somente os números (posição, orientação, abertura) — basta marcar a opção
> correspondente no final deste termo.

**(d) Sensores no punho/mão (IMU).** Texto sugerido:

> Poderemos fixar um pequeno sensor de movimento (acelerômetro/giroscópio) no seu punho
> ou no dorso da mão, com uma faixa ou fita hipoalergênica. Ele não emite nenhum estímulo
> e serve apenas para registrar o movimento. Pode causar leve pressão ou incômodo na
> pele; se houver irritação, o sensor é retirado imediatamente. Este registro é
> **opcional** e você pode recusá-lo sem qualquer prejuízo à sua participação.

**(e) Duração.** Texto sugerido:

> Cada visita tem duração estimada de **1 hora e 25 minutos a 1 hora e 45 minutos**,
> incluindo preparação, tarefa e pausas. São **2 visitas**, com pelo menos 48 horas de
> intervalo entre elas.

**(f) Uso e guarda dos dados (resumo em linguagem simples).** Texto sugerido:

> Seus dados de EEG, os vídeos das mãos, as medidas de movimento e as respostas às tarefas
> serão identificados por um **código** (por exemplo, S01), e não pelo seu nome. Ficam em
> computador institucional com acesso restrito e senha, por **5 anos** após o fim do
> estudo, e depois são **excluídos**. Nenhuma imagem sua será publicada. Os resultados
> serão divulgados apenas de forma agregada.

**(g) Novos desfechos, em linguagem simples.** Texto sugerido:

> O objetivo desta etapa é verificar se o sinal cerebral permite estimar **qual posição,
> orientação e abertura da mão** você pretende usar, e não apenas "abrir" ou "fechar".
> Não há resposta certa nem errada na tarefa, e o seu desempenho individual não será
> divulgado.

### 6.2. Itens de checagem (checkbox) a acrescentar no TCLE

- [ ] Autorizo a gravação de vídeo da região das minhas mãos e antebraços.
- [ ] Autorizo o uso do sensor de movimento no meu punho/mão *(opcional)*.
- [ ] Solicito que o **vídeo bruto seja apagado** após a extração das medidas de pose,
      mantendo-se apenas os dados numéricos *(opcional)*.
- [ ] Autorizo o uso dos meus dados **anonimizados** em publicações e apresentações
      científicas *(não implica publicação de imagens)*.
- [ ] Recebi e li a versão 2 do TCLE, com data de `____/____/______`, e tive minhas
      dúvidas esclarecidas.

### 6.3. O que **não** pode constar no TCLE

- promessa de benefício clínico, de controle de prótese ou de melhora funcional;
- autorização genérica e ilimitada de uso de imagem ("para qualquer finalidade");
- cessão de direitos sobre imagem ou dados;
- renúncia a direitos de ressarcimento por danos decorrentes da pesquisa.

---

## 7. Dados, privacidade e LGPD

### 7.1. Categorias de dados e classificação

| Dado | Natureza LGPD | Local de armazenamento | Prazo |
| --- | --- | --- | --- |
| EEG bruto + eventos | dado pessoal (saúde em sentido amplo) | computador institucional, pasta criptografada | 5 anos |
| Vídeo bruto das mãos/antebraços | **dado pessoal** (imagem) | HD externo institucional dedicado, acesso restrito | até 5 anos, ou descarte após extração, se solicitado |
| Landmarks 2D + pose 3D triangulada | dado pseudonimizado derivado | computador institucional | 5 anos |
| IMU bruto | dado pessoal (movimento) | computador institucional | 5 anos |
| QEEG / devolutiva individual | dado pessoal (saúde) | entregue ao participante; cópia com o pesquisador | 5 anos |
| Termos assinados e lista de contato | dado pessoal + identificável | arquivo físico da instituição, armário com chave | 5 anos, depois destruição |

### 7.2. Base legal e consentimento

Tratamento com base no **consentimento livre, esclarecido e específico** do participante
(LGPD, art. 7º, I, e art. 11, I, para dados de saúde), formalizado no TCLE v2, com
finalidade determinada (decodificação de pose por EEG), sem autorização para finalidades
secundárias não descritas. O consentimento é revogável a qualquer momento, sem prejuízo.

### 7.3. Minimização e segurança

- **minimização:** o vídeo cobre apenas a região de trabalho (antebraço, mão, objetos);
  rosto e tronco ficam fora do enquadramento, verificado com o participante antes de
  gravar; não há áudio; o Kinect, quando usado, fornece apenas imagem RGB;
- **finalidade:** o vídeo não é usado para identificar pessoas, para análise de expressão,
  para biometria ou para qualquer finalidade além da estimação de pose;
- **processamento local:** todo o processamento de visão ocorre offline, no laboratório,
  em máquina institucional, sem envio de imagens a serviços de nuvem ou APIs externas;
- **pseudonimização:** identificação por código (`S01`, `S02`, ...); a tabela de
  correspondência código ↔ nome fica em arquivo separado, protegido e com acesso apenas
  do pesquisador responsável;
- **segurança:** disco/partição criptografada; acesso por usuário nominal; backup
  redundante sem uso de nuvem pública; registro de acesso; proibição de cópia em
  dispositivos pessoais e em unidades sincronizadas automaticamente (OneDrive/Dropbox/
  Google Drive) — a exceção são documentos administrativos sem dados de participantes
  (protocolo, TCLE em branco, cronograma);
- **versionamento público:** EEG, vídeo, landmarks e pose **não** entram em repositório
  Git público; apenas código e documentação de esquema.

### 7.4. Retenção, descarte e revogação

- retenção de 5 anos após o término do estudo, conforme recomendação do CEP;
- ao final do prazo: exclusão definitiva dos arquivos e sobrescrita/degradação física da
  mídia, com registro de data e responsável em ata interna;
- se o participante revogar o consentimento: os dados identificáveis e o vídeo são
  excluídos; dados já agregados em publicação não podem ser individualizados nem retirados
  retrospectivamente, e esse limite é informado no TCLE;
- descarte do vídeo bruto após extração dos rótulos, quando solicitado pelo participante,
  mantendo-se apenas landmarks e pose.

### 7.5. Compartilhamento e publicação

- compartilhamento apenas de **dados derivados e anonimizados** (pose, landmarks sem
  identificadores, EEG pseudonimizado) com finalidade de pesquisa e mediante descrição no
  TCLE;
- publicação de resultados somente agregados; **nenhuma figura com vídeo ou fotografia de
  participante**;
- submissão de conjuntos de dados a repositórios públicos somente se houver consentimento
  específico — não é o caso desta emenda;
- em caso de solicitação por autoridade competente, a resposta seguirá o trâmite
  institucional e a LGPD.

### 7.6. Direitos do participante

Confirmar acesso, correção, anonimização, bloqueio ou eliminação, e revogação do
consentimento, exercidos por contato direto com o pesquisador responsável e com o
Encarregado de Dados (DPO) da UFSC, cujo contato deve constar no TCLE.

---

## 8. Análise estatística

Plano de análise **fixado antes do início da coleta** e registrado neste documento; desvios
serão documentados e justificados no relatório final.

### 8.1. Métricas por desfecho

| Desfecho | Métricas |
| --- | --- |
| Pose final/hold — posição | RMSE e MAE em mm (ou cm), erro por eixo, R² |
| Pose final/hold — orientação | erro angular geodésico médio (graus), mediana e IQR |
| Pose final/hold — abertura | RMSE em mm e % da amplitude de abertura medida |
| Velocidade auxiliar (exploratório) | correlação de Pearson/Spearman por eixo, erro DTW |
| Classificação (secundário) | acurácia, precisão, recall, F1, kappa de Cohen, ITR |
| Qualidade do dado | % frames com 2+ vistas, erro de reprojeção, exclusões |

### 8.2. Unidade de análise e prevenção de pseudorreplicação

A unidade de análise é o **trial** (não o frame). Frames consecutivos dentro da mesma
trial são altamente correlacionados e não constituem observações independentes. Por isso:

- agregação por trial quando o desfecho é pose final (uma predição por trial, por exemplo
  via pooling por atenção ou janela tardia);
- quando o modelo for treinado em frames, os **splits são por sujeito e por sessão**,
  nunca por frame aleatório (evita vazamento temporal e inflação de desempenho);
- intervalos de confiança calculados por **bootstrap agrupado por sujeito** (10.000
  reamostragens), e não por frame.

### 8.3. Modelagem

1. **Baseline obrigatório:** regressão linear/ridge sobre características espectrais
   simples (potência em μ e β por canal) — referência mínima para qualquer modelo
   profundo; e preditor "constante" (média do alvo) como piso de erro.
2. **Controle negativo:** rótulos embaralhados entre trials (permutação) e teste de
   permutação com 1.000 repetições para verificar se o desempenho é superior ao acaso.
3. **Modelo principal:** arquitetura compacta com encoder compartilhado e cabeças de pose
   e de classe (inspirada em SAND/MTRT, mas com número de parâmetros compatível com
   ~250 trials por participante).
4. **Validação:** leave-one-subject-out (LOSO) para estimar generalização inter-sujeito;
   validação cruzada por sessão dentro do sujeito para estimar desempenho intra-sujeito;
   avaliação zero-shot (treino em ME, teste em MI) e após ajuste fino com poucas trials.

### 8.4. Inferência estatística

- comparação entre condições (ME vs MI; zero-shot vs fine-tune; mão direita vs esquerda)
  com **teste de Wilcoxon para amostras pareadas** (não paramétrico, adequado a n = 5–6)
  ou **modelos mistos lineares** com intercepto aleatório por participante, quando o
  modelo convergir;
- tamanhos de efeito reportados obrigatoriamente: **d de Cohen pareado (d_z)**, delta de
  Cliff, e IC 95% por bootstrap;
- **correção para múltiplas comparações** (Holm–Bonferroni ou FDR de Benjamini–Hochberg)
  sobre as três métricas do desfecho primário (posição, orientação, abertura) e sobre o
  conjunto de comparações secundárias;
- nível de significância α = 0,05 (bilateral) após correção;
- toda inferência será apresentada com estimativa pontual **e** intervalo de confiança;
  nenhuma conclusão será baseada apenas em valor-p.

### 8.5. Poder, limitações e interpretação

- com n = 5–6, o estudo **não tem poder para testes confirmatórios**; o objetivo é
  estimar tamanhos de efeito com precisão suficiente para planejar um estudo maior;
- não será realizado cálculo de poder a priori apresentado como garantia; a justificativa
  amostral é de viabilidade e de precedente na literatura (estudos piloto de BCI com 4–6
  participantes);
- resultados negativos (ausência de transferência ME → MI) são informativos e serão
  reportados como tal, sem reformulação post hoc das hipóteses;
- não haverá análise interina com regra de parada por eficácia; a parada por fadiga ou
  evento adverso não é estatística e não afeta a interpretação.

### 8.6. Critérios de exclusão de dados (definidos a priori)

- trials com erro de sincronização > 10 ms;
- trials sem pose válida (menos de 2 vistas em > 20% dos frames, ou `confidence` baixa);
- trials com artefato de EEG grosseiro (saturação, perda de canal, `nan_frac` > 1%);
- trials de MI com contração muscular visível no EMG;
- sessões com calibração de câmera inválida (reprojeção mediana > 5 px após
  recalibragem) — nesse caso a sessão é repetida, não "corrigida";
- todas as exclusões serão quantificadas e reportadas, com análise de sensibilidade
  incluindo os casos excluídos.

### 8.7. Reprodutibilidade

- sementes aleatórias registradas em arquivo de configuração versionado;
- versões de código, ambiente e parâmetros de aquisição registradas por sessão
  (`meta.yaml`);
- scripts de análise versionados em repositório Git privado; dados não versionados;
- relatório de qualidade por sessão (`quality_report.json`) com os critérios de 8.6.

---

## 9. Cronograma

O cronograma do projeto aprovado previa aprovação em julho de 2026 e início da coleta em
agosto de 2026 — datas já ultrapassadas. O cronograma abaixo é **relativo à data de
submissão desta emenda (S0)** e deve ser convertido em datas absolutas pelo pesquisador
responsável, em conjunto com o orientador.

| Etapa | Prazo relativo | Data-alvo provisória |
| --- | --- | --- |
| S0 — Submissão da emenda na Plataforma Brasil (com TCLE v2 e anexos) | — | novembro/2026 `[PREENCHER]` |
| S1 — Resposta às pendências do parecer, se houver | até +30 dias do parecer | `[PREENCHER]` |
| S2 — Aprovação final da emenda | até +60 dias de S0 (prazo do CEP) | `[PREENCHER]` |
| S3 — Piloto técnico **sem participantes** (calibração, sincronização, checklist) | S2 + 2 semanas | `[PREENCHER]` |
| S4 — Coleta: sessão 1 dos 5–6 participantes | S2 + 3 a 8 semanas | `[PREENCHER]` |
| S5 — Coleta: sessão 2 (≥ 48 h após a sessão 1) | S4 + 1 a 2 semanas | `[PREENCHER]` |
| S6 — Controle de qualidade, extração de pose e fechamento do dataset | S5 + 2 semanas | `[PREENCHER]` |
| S7 — Análise estatística conforme seção 8 | S6 + 4 semanas | `[PREENCHER]` |
| S8 — Redação, devolutiva aos participantes e submissão de artigo | S7 + 6 semanas | `[PREENCHER]` |

**Regra de bloqueio:** nenhuma etapa a partir de S3 ocorre antes de S2. O piloto de S3
usa apenas dados do próprio pesquisador ou dados sintéticos, sem participante recrutado.

---

## 10. Impacto orçamentário (resumo)

| Item novo | Justificativa | Custo estimado |
| --- | --- | --- |
| Mídia de armazenamento dedicada (HD externo) | guarda de vídeo bruto com acesso restrito | `[PREENCHER]` |
| Gel condutor adicional e eletrodos descartáveis | 2 sessões com montagem completa | `[PREENCHER]` |
| Material de fixação do IMU (faixas/fitas hipoalergênicas) | conforto e segurança | `[PREENCHER]` |
| Lanches e ressarcimento de transporte | 2 visitas por participante, visitas mais longas | `[PREENCHER]` |
| Impressão e envelopes de TCLE v2 | duas vias por participante e por sessão | `[PREENCHER]` |
| **Total adicional estimado** | — | R$ 600–1.200 `[PREENCHER]` |

Fonte: IEB/UFSC e recursos próprios, como no projeto aprovado.

---

## 11. Anexos e checklist de submissão

### 11.1. Anexos a preparar

| Anexo | Conteúdo | Status |
| --- | --- | --- |
| **Anexo I** | TCLE versão 2 (com os blocos da seção 6 destacados e checkboxes) | a produzir |
| **Anexo II** | Protocolo detalhado do trial (linha do tempo de 12 s, fases e marcadores) | a produzir a partir desta seção 3.4 |
| **Anexo III** | Lista de equipamentos e certificações (g.Nautilus 32, câmeras, IMU, EMG) | a produzir |
| **Anexo IV** | Plano de análise estatística (seção 8 formalizada) | a produzir |
| **Anexo V** | Plano de guarda, acesso e descarte de dados (seção 7 formalizada) | a produzir |
| **Anexo VI** | Formulário de fadiga/desconforto e de evento adverso | a produzir |
| **Anexo VII** | Carta de encaminhamento da emenda, com justificativa resumida | a produzir |
| **Anexo VIII** | Cronograma e orçamento atualizados | a produzir |

### 11.2. Checklist de submissão (Plataforma Brasil)

- [ ] Cadastro do projeto existente localizado (CAAE e parecer corretos).
- [ ] Tipo de solicitação marcado como **emenda** (não novo projeto).
- [ ] Carta de encaminhamento assinada pelo pesquisador responsável.
- [ ] Justificativa da emenda (seção 2 deste documento, adaptada ao formulário).
- [ ] Protocolo detalhado atualizado (seção 3).
- [ ] TCLE v2 com data e versão, em PDF pesquisável.
- [ ] Tabela comparativa v1 × v2 (seção 4) anexada como material de apoio.
- [ ] Seção de riscos atualizada, com o risco físico novo explicitado (seção 5).
- [ ] Plano de dados/LGPD atualizado (seção 7).
- [ ] Plano de análise estatística atualizado (seção 8).
- [ ] Cronograma revisado (seção 9), coerente com a nova data de início.
- [ ] Orçamento revisado (seção 10).
- [ ] Declaração de infraestrutura do local de coleta.
- [ ] Declaração de que nenhum participante foi recrutado nem coletado antes da aprovação.
- [ ] Se houver dados de piloto prévios (inclusive autoexperimentação): declarar
      explicitamente e descrever uso pretendido.
- [ ] Checklist interno completo (11.3) antes de clicar em "enviar".

### 11.3. Checklist técnico pré-sessão (após aprovação)

- [ ] Câmeras fixas, calibradas no dia; erro de reprojeção mediano ≤ 5 px.
- [ ] Enquadramento sem rosto/tronco, conferido com o participante.
- [ ] Sincronização EEG ↔ câmeras ↔ paradigma verificada (offset ≤ 10 ms).
- [ ] Touca montada, impedância verificada, canais sem sinal morto.
- [ ] EMG nos extensores, bilateral, funcionando.
- [ ] IMU bruto gravando (somente se autorizado no TCLE).
- [ ] Script do paradigma com 125 trials, 5 blocos, sementes registradas.
- [ ] Pausas de 3 min configuradas; escala de fadiga preparada.
- [ ] TCLE v2 assinado em duas vias, com checkboxes de imagem/IMU/descarte marcados.
- [ ] Espaço de armazenamento criptografado disponível e verificado.
- [ ] `meta.yaml`, `manifest.csv` e `quality_report.json` preparados.

---

## 12. Pendências e pontos para decisão do pesquisador

1. **Número do parecer e CAAE** do projeto aprovado — preencher em todo o documento.
2. **Versão final do trial:** confirmar a divisão `cue` (~1 s) + execução (~3 s) dentro do
   bloco de 4 s e se a animação é reapresentada no intervalo de priming ou apenas no cue.
3. **Número de participantes:** 5 ou 6 — a diferença não altera o desenho, mas precisa ser
   coerente com o tamanho amostral declarado ao CEP.
4. **IMU:** incluir como padrão ou apenas como opcional recusável. Recomendação: opcional,
   com recusa sem prejuízo.
5. **Descarte do vídeo bruto:** oferecer como opção do participante (recomendado) ou adotar
   como padrão para todos? A primeira opção preserva o dataset; a segunda é mais protetiva.
6. **Kinect:** usar ou não como terceira vista. Se usado, manter o mesmo esquema de
   `camera_id` e registrar no `meta.yaml`.
7. **Datas absolutas** do cronograma — o cronograma original já venceu; é preciso
   reprogramar formalmente junto ao orientador e ao programa.
8. **Dados de piloto prévios:** verificar se existe alguma gravação anterior (inclusive do
   próprio pesquisador) e, se sim, declarar ao CEP antes da submissão.

---

## 13. Referências de apoio à justificativa

Chaves BibTeX correspondentes já presentes em `library.bib` da tese:

- `kwon2026e2t` — E2T: EEG-to-Trajectory Transformer for Motor Imagery-Based Fully-DoF
  Motion Prediction (IEEE TNSRE, 2026).
- `fu2026sand` — SAND: Spectral-Attention Neural Decoding of Hand Kinematics from
  Low-Frequency EEG Dynamics (IEEE TBME, 2026, early access).
- `wang2023mtrt` — MTRT: Motion Trajectory Reconstruction Transformer (IEEE TNSRE, 2023).
- `m3t2026` — M3T-attention: multi-level multi-scale temporal attention transformer
  (Cognitive Neurodynamics, 2026).
- `reviewtraj2023` — Revisão sobre reconstrução de trajetória motora para BCI baseada em
  EEG (Frontiers in Neuroscience, 2023).
- `imagined3d2019` — Decodificação de trajetórias 3D imaginadas para controle de dois
  braços virtuais (Frontiers in Neurorobotics, 2019).
- `Stinear2005` — Imagética cinestésica, e não visual, modula a excitabilidade corticomotora
  (Experimental Brain Research, 2006; DOI 10.1007/s00221-005-0078-y).
- `imageryvoluntary2003` — Imagética de movimento voluntário ativa representações motoras
  específicas por parte do corpo (Journal of Neurophysiology, 2003).
- `luciw2014way` — WAY-EEG-GAL: registros de EEG durante 3.936 trials de preensão e
  elevação (Scientific Data, 2014).

---

*Documento de trabalho. Não substitui o parecer do CEP nem a versão oficial protocolada na
Plataforma Brasil.*

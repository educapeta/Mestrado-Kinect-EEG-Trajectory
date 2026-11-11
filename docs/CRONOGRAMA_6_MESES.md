# Cronograma de 6 meses — SAND-Pose (ME/MI de 8 s)

## Premissas

- 5–6 participantes saudáveis;
- 2 sessões por participante;
- 100–125 trials por sessão (200–250 por participante);
- trial de 20 s: HOME 2 + ME 8 + animação 2 + MI 8;
- 6 classes (3 objetos × 2 mãos) ou 4 classes (2 objetos × 2 mãos);
- objetivo: pose final/hold + orientação 6D + abertura; velocidade auxiliar;
- offline + avaliação causal (sem malha fechada);
- sem dedos individuais e sem controle online completo.

## Visão de 24 semanas

| Período | Foco | Entregável |
| --- | --- | --- |
| **S1–S2** | Congelar protocolo; escrever emenda CEP; revisar TCLE; orçamento | Emenda pronta para submissão |
| **S3–S4** | Submeter CEP; montar/calibrar câmeras; pipeline de aquisição e labels; teste com objeto de calibração (sem humanos) | Pipeline técnico funcionando |
| **S5–S6** | Responder pendências do CEP; preparar análise; atualizar metodologia da tese | CEP aprovado ou em fase final |
| **S7–S8** | Piloto técnico com 1–2 participantes (após aprovação); QC; ajuste de janelas/labels | Piloto validado |
| **S9–S14** | Coleta principal: 5–6 participantes × 2 sessões (10–12 visitas), com QC após cada sessão | Dataset completo |
| **S11–S18** | Modelagem em paralelo: baseline, SAND-Pose, ablations, LOSO; começar com os primeiros participantes | Resultados principais |
| **S19–S21** | Análise final, figuras, tabelas, estatística; coleta de reposição se necessário | Resultados fechados |
| **S22–S24** | Escrita dos capítulos, revisão, defesa/qualificação, buffer | Versão final |

## Gates de decisão

### Gate 1 — fim do mês 2

**Pergunta:** a emenda ao CEP está submetida e o pipeline técnico está pronto?

- **Sim:** seguir para piloto.
- **Não:** priorizar CEP e reduzir escopo do piloto; não iniciar coleta.

### Gate 2 — fim do mês 3

**Pergunta:** o piloto com 1–2 participantes gerou labels de pose com QC aceitável?

- **Sim:** iniciar coleta principal.
- **Não:** simplificar alvo (pose final sem trajetória), reduzir classes ou trocar câmeras; não escalar um pipeline ruim.

### Gate 3 — fim do mês 4

**Pergunta:** já há 3 participantes coletados com qualidade?

- **Sim:** continuar e começar treino sério.
- **Não:** reduzir para 3–4 participantes ou 100 trials/sessão; replanejar análise.

### Gate 4 — fim do mês 5

**Pergunta:** o modelo aprende pose acima do baseline?

- **Sim:** fechar análise e escrever.
- **Não:** ativar plano B: execução + classificação MI + caracterização da transferência; reportar resultado negativo com rigor.

## O que pode dar errado

| Risco | Impacto | Mitigação |
| --- | --- | --- |
| CEP demora > 8 semanas | atrasa toda a coleta | submeter na S1–S2; ter plano B de escopo; não coletar antes |
| Calibração das câmeras instável | labels ruins | calibrar antes, QC por sessão, objeto de calibração, 3ª câmera |
| Participantes faltam | amostra menor | recrutar 7 para garantir 5–6 |
| Fadiga com trial de 20 s | queda de qualidade | 100 trials/sessão, 5 blocos de 20 ou 4 de 25, pausas 3–5 min |
| Modelo não generaliza | resultado fraco | pré-treino WAY/EEG, modelo compacto, LOSO, plano B |
| Escrita deixada para o fim | atraso na defesa | escrever métodos/introdução em paralelo desde S1 |

## Recomendação de escopo para caber em 6 meses

Fazer:

- 5–6 participantes;
- 2 sessões × 100 trials (200/participante) ou 2 × 125 (250);
- pose final/hold + orientação + abertura;
- velocidade auxiliar;
- offline + causalidade/latência;
- classificação MI como baseline.

Não fazer:

- dedos individuais;
- controle em malha fechada da prótese;
- validação com pacientes;
- modelo online completo em tempo real no robô;
- coleta antes da aprovação da emenda.

## Resposta curta

**Sim, é possível em 6 meses** se:

1. a emenda ao CEP for submetida imediatamente;
2. a coleta começar até o mês 3;
3. o escopo ficar congelado em pose final + orientação + abertura;
4. a modelagem começar com os primeiros participantes, não no fim;
5. houver plano B documentado para a hipótese de transferência.

Se o CEP passar de 2 meses ou a calibração das câmeras atrasar, o cronograma fica muito apertado. Nesse caso, reduza para 4 participantes, 100 trials/sessão e mantenha o núcleo da contribuição.

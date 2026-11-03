# Fase 1.5 -- Auditoria dos relatórios A1/A2 contra os PDFs

**Data:** 05/11/2026  
**Objetivo:** separar o que é fato verificável no PDF do que foi inferência,
imprecisão ou alucinação dos agentes da Fase 1, antes de usar o material na
Fase 2 (arquitetura) e na dissertação.

## 1. Método

- Texto extraído com `pypdf` diretamente dos PDFs em `literatura/`.
- Páginas citadas abaixo são **páginas PDF (1-based)**, não páginas impressas
  do artigo.
- Para cada afirmação central, procurei o termo no texto e conferi a página.
- Páginas/equações citadas nos relatórios A1/A2 foram tratadas como suspeitas
  até confirmação.

## 2. Metadados corrigidos

| Referência | Correção | DOI/URL | Páginas PDF |
|---|---|---|---|
| Mahony, Hamel & Pflimlin (2008) | Título correto: **Nonlinear Complementary Filters on the Special Orthogonal Group** (o A1 chamou de "Complementary Filter Design on SO(3)") | 10.1109/TAC.2008.923738 | 17 |
| Markley, Cheng, Crassidis & Oshman | Ano correto: **2007** (A1 e o plano diziam 2002). Título: Quaternion Averaging | NTRS 20070017872 | 10 |
| Tian et al. (2015) | Upper limb motion tracking with the integration of IMU and Kinect | 10.1016/j.neucom.2015.01.071 | 12 |
| Acharya, Bhat & Kanthi (2022) | Título e autores confirmados; DOI descoberto no rodapé | 10.4316/AECE.2022.04010 | 8 |
| Zhang et al. (2020) | MediaPipe Hands; não há DOI no PDF; usar arXiv | arXiv:2006.10214 | 5 |
| Kusunose et al. (2025) | Applied Sciences 15, 10527 | 10.3390/app151910527 | 19 |
| Xefteris et al. (2025) | Internet of Things, **vol. 29, art. 101465, jan/2025** | 10.1016/j.iot.2024.101465 | 13 |
| Jaruchokphuriwaj & Varagul (2024) | IJMPE 12(7), **pp. 20-23** (não 144-175) | http://ijmpe.iraj.in/paperdetails/21035 | 4 |
| Palani & Panigrahi (2022) | IEEE BIBE 2022; DOI descoberto no PDF | 10.1109/BIBE55377.2022.00035 | 6 |
| Lafayette et al. (2021) | The Virtual Kinect; **SVR 2021**, não SBGames | 10.1145/3488162.3488215 | 9 |

## 3. Auditoria do relatório A1 (Mahony/Markley/Tian/Acharya)

| Afirmação do A1 | Veredito | Evidência no PDF |
|---|---|---|
| Mahony tem variantes direct e passive; a passive desacopla o giro da atitude reconstruída | **Confirmado** | Mahony p. 2 |
| Sem magnetômetro, yaw não tem referência absoluta e deriva | **Confirmado conceitualmente** | Mahony p. 4 (magnetômetro) e p. 13 (curvas de yaw); o artigo não faz um experimento específico "6 eixos sem mag", mas a limitação é consequência direta |
| Título do Mahony é "Complementary Filter Design on SO(3)" | **ERRADO** | Título real na p. 1 do PDF |
| Markley: média = autovetor de M associado ao maior autovalor | **Confirmado** | Markley p. 4 ("eigenvector ... maximum eigenvalue") |
| Markley: equivalência com máxima verossimilhança e pesos | **Confirmado** | Markley p. 2 (MLE) e pp. 2-4 (pesos) |
| Markley ano 2002 | **ERRADO** | NTRS: publicação em 01/01/2007 |
| Markley autores (Markley, Cheng, Crassidis, Oshman) | **Confirmado** | Markley p. 1 |
| Tian: estado [p, v, q], UKF, fusão IMU+Kinect | **Confirmado** | Tian p. 1 (UKF/abstract) e p. 3 (Eq. 1, estado) |
| Tian: transformação Kinect <-> frame global na Sec. 5.1 | **Confirmado** | Tian p. 7, Sec. 5.1 |
| Tian: viés estático de 45 min / modelo exponencial | **Não verificado nesta auditoria** | Não usado na arquitetura por enquanto |
| Acharya: MPU6050 + Kinect para orientação de membro superior | **Confirmado** | Acharya p. 1 |
| Acharya: yaw não observável pelo acelerômetro; roll/pitch sim | **Confirmado conceitualmente** | Acharya p. 3 ("insensitive to rotations about the gravitational field vector") |
| Acharya: Wilcoxon + Spearman; correlações ~0,94-0,96 | **Confirmado** | Acharya p. 1 (Wilcoxon) e p. 6 (rs = 0.9636/0.9394; Kinect 0.9515) |
| Páginas/equações citadas no A1 (ex.: Mahony Eqs. 7/47, pp. 5-6/15-16; Markley Eq. 12 p. 3) | **Não confiáveis como estão** | As páginas corretas devem ser reconferidas caso a caso; a numeração do relatório não bate com a extração |

**Resumo A1:** conceitos aproveitáveis, mas o relatório **não pode ser citado
como fonte final**: título do Mahony errado, ano do Markley errado e páginas/
equações não verificadas.

## 4. Auditoria do relatório A2 (MediaPipe/Kusunose/Xefteris/IJMPE/BIBE/Virtual Kinect)

### 4.1 O problema sistêmico: páginas inventadas
O A2 cita páginas que **não existem** nos PDFs:

| PDF | Páginas reais | Exemplo de citação do A2 |
|---|---|---|
| MediaPipe Hands | 5 | pp. 117-125 |
| IJMPE | 4 (pp. 20-23) | pp. 144-175 |
| Xefteris | 13 | p. 407 / pp. 509-512 |
| Virtual Kinect | 9 | pp. 410-456, 601-672 |
| BIBE | 6 | pp. 321-629 |

Isso invalida **todas** as referências de página/equação do A2 como estão.

### 4.2 Afirmações centrais

| Afirmação do A2 | Veredito | Evidência no PDF |
|---|---|---|
| MediaPipe usa palm detector + hand landmark model e retorna landmarks 2.5D | **Confirmado** | MediaPipe p. 1 |
| Profundidade relativa é aprendida só de imagens sintéticas | **Confirmado** | MediaPipe p. 2 ("relative depth w.r.t. the wrist point being learned only from synthetic images") |
| MediaPipe Hands tem 21 landmarks | **Convenção do modelo/documentação**, não explicitamente no texto do paper | O paper fala em 2.5D landmarks; usar a documentação oficial para o número 21 |
| IJMPE: ponto O = interseção 1-17 e 0-9 | **Confirmado** | IJMPE p. 2, Fig. 2 |
| IJMPE: yaw vem do ângulo 2D no plano da imagem | **Confirmado** | IJMPE p. 2 |
| IJMPE: X do IMU ao longo dos dedos, Z perpendicular à palma | **Confirmado** | IJMPE p. 2 |
| IJMPE: 20 joint points citados (MediaPipe real tem 21) | **Confirmado com ressalva** | IJMPE p. 2 diz "skeleton of 20 hand joint points"; é a descrição dos autores |
| IJMPE: 2,33 mm é erro posicional médio | **Parcialmente correto** | IJMPE p. 3 Tabela 1: X 2,44 mm, Y 2,22 mm, **Z 2,33 mm**; 2,33 é a média do eixo Z, não um erro 3D único. Não usar "2,33 mm" como erro global |
| IJMPE: jig de 0 a 90° em passos de 15° | **Confirmado** | IJMPE p. 3 |
| IJMPE: páginas 144-175 | **ERRADO** | PDF tem 4 páginas, pp. 20-23 |
| Kusunose: features normalizadas e seleção de 22 landmarks | **Confirmado** | Kusunose p. 5 (22 landmarks/selected) e p. 6 (Tabela 2) |
| Kusunose: SHAP identifica triângulo palmar como principal | **Confirmado** | Kusunose p. 1 e p. 8 |
| Kusunose: `norm_palm_size` e `rad_palm_angle` dominam >35% | **NÃO ENCONTRADO** | Não há "35%" no texto extraído; remover até achar a Tabela/figura exata |
| Kusunose: z não usado diretamente, normalizado por tamanho | **Confirmado** | Kusunose p. 5 |
| Xefteris: fusão LSTM + Random Forest com IMU + RGB | **Confirmado** | Xefteris p. 1 |
| Xefteris: latência total 81,41 ms (~12,28 fps), não é tempo-real estrito | **Confirmado; página corrigida** | Xefteris p. 12 (não pp. 509-512) |
| BIBE: Kalman sequencial com múltiplos updates; visão + IMU | **Confirmado** | BIBE p. 4 |
| BIBE: RMSE IMU 6,30°, visão 7,70°, fundido 6,18° | **Confirmado** | BIBE p. 1 |
| BIBE: estado `[ângulo, velocidade angular]` com F = [[1, dt],[0,1]] | **NÃO CONFIRMADO** | O PDF descreve um modelo de estados genérico (Eq. 5), mas não essa matriz específica; tratar como adaptação nossa, não como fato do paper |
| Virtual Kinect: 25 juntas Kinect ↔ 33 MediaPipe, apenas corpo | **Confirmado** | Virtual Kinect p. 2 (25 juntas) e p. 5 (33 landmarks MediaPipe); Tabela 2 p. 6 |
| Virtual Kinect: mapeamento de mãos | **ERRADO/não suportado** | O paper é de pose corporal; não mapeia mãos de 21 landmarks |
| MediaPipe falha com mãos acima da cabeça | **Confirmado** | Xefteris p. 12 |

**Resumo A2:** a síntese conceitual é útil, mas o relatório tem páginas
fabricadas e alguns números específicos não sustentados (35% do SHAP, estado
`[ângulo, taxa]` do BIBE). Não usar como fonte final.

## 5. O que é seguro levar para a Fase 2

1. **Mahony 6 eixos**: roll/pitch observáveis, yaw deriva; visão precisa dar
   referência de yaw. (Mahony pp. 2, 4, 13)
2. **Markley**: média de quaternions = autovetor principal; pesos/MLE;
   cuidado com q e -q, outliers e condição de unicidade. (Markley pp. 2-4)
3. **Tian**: UKF com estado [p, v, q] e transformação Kinect→global; usar como
   arquitetura de referência, não copiar parâmetros. (Tian pp. 1, 3, 7)
4. **Acharya**: IMU bom para tempo/relativo; Kinect bom para posição absoluta;
   yaw do acelerômetro não observável. (Acharya pp. 1, 3, 6)
5. **MediaPipe**: landmarks 2.5D, profundidade relativa não métrica; usar
   depth do Kinect; normal/PCA é adaptação nossa, não do paper.
6. **IJMPE**: referência O 1-17/0-9, yaw 2D, eixos X/Z do IMU; números são de
   RealSense+Witmotion, **não** do Kinect v2; 2,33 mm é média do eixo Z.
7. **Kusunose**: features normalizadas e SHAP; útil para desenho de features,
   mas é pronação/supinação de antebraço, não orientação de palma.
8. **Xefteris**: fusão tardia e limitação de latência (81,41 ms), não tempo-real.
9. **BIBE**: Kalman sequencial e sincronização; RMSE 6,30/7,70/6,18; o modelo
   `[ângulo, taxa]` é adaptação nossa, não fato do paper.
10. **Virtual Kinect**: mapeia corpo (25↔33), não mãos; não usar para 21
    landmarks.

## 6. O que NÃO usar

- Qualquer página/equação dos relatórios A1/A2 como está: reconferir no PDF.
- Título "Complementary Filter Design on SO(3)" (Mahony).
- Markley 2002 (é 2007).
- IJMPE "2,33 mm de erro posicional" como erro 3D global.
- Kusunose ">35%" sem achar a fonte exata.
- BIBE `[ângulo, velocidade]`/matriz F como fato do paper.
- Qualquer conclusão do tipo "o artigo X garante Y" sem PDF + página.

## 7. Implicações para a arquitetura (Fase 2)

- V1: Kinect (pos/depth) + MediaPipe (landmarks/normal) + IMU (Mahony);
  filtro complementar em quaternion; sem JointOrientation/sem estéreo.
- Não usar orientação de junta do Kinect: o body tracking não rastreia na
  montagem atual (documentado em `docs/DIAGNOSTICO_KINECT_BODY.md`).
- A normal da palma por PCA/landmarks é **proposta nossa**; precisa de teste
  sintético e de validação com o Kinect/jig, não é "do artigo".
- Sincronização de timestamps e confiança/edge-on entram como contratos
  explícitos na Fase 2.
- EEG: `FUSED_*` opcionais/versionadas; não substituir `PALM_*`/`IMU_*` sem
  validação.

## 8. Correções já aplicadas

- `literatura/referencias.bib` corrigido (Markley 2007, Mahony título, Xefteris
  vol. 29/2025, Acharya DOI, BIBE DOI, Virtual Kinect SVR 2021).
- Relatórios A1/A2 receberam aviso no topo apontando para esta auditoria.
- `literatura/relatorios/README.md` atualizado.

## 9. Pendências

- Tese exata de Polytechnique Montréal sobre calibração de mão: ainda não
  localizada; o PDF `2022_MarkAsmar.pdf` é outra tese (transcrição de guitarra).
- Se formos citar Tian/Mahony/Markley em detalhe, fazer uma segunda passada
  conferindo equação por equação (esta auditoria priorizou afirmações
  conceituais e metadados).

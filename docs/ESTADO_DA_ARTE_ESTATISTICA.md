# Estado da Arte: Bateria de Estatística e Métricas para BCI e Ground Truth

> **Mestrado em Engenharia Elétrica / Neuroengenharia**  
> Documento metodológico de referência para decodificação contínua de trajetórias (EEG $\to$ Cinemática 3D), classificação em BCI e validação geométrica de tracking óptico/inercial.  
> Baseado em: Müller-Putz et al. (2008), Sassenhagen & Draschkow (2019), Maris & Oostenveld (2007), Benjamini & Hochberg (1995), Wolpaw et al. (2000).  
> Implementação correspondente: `estat_bci.py` e ferramenta de linha de comando `tools/analise_estatistica.py`.

---

## 1. Nível de Acaso Exato vs. Acaso Assintótico em BCI

### 1.1 O Mito dos "50% de Chance" em Amostras Finitas
Em tarefas de classificação binária (e.g., Imagética Motora mão direita vs. mão esquerda), assume-se frequentemente que a probabilidade teórica $p_0 = 0.5$ é o limiar para determinar significância estatística. **Isso é falso para pequenos números de trials.**

Em um protocolo experimental finito com $N$ trials independentes, o número de acertos sob a hipótese nula $H_0$ segue rigorosamente uma distribuição binomial:
$$k \sim \text{Binomial}(N, p_0)$$

O limiar crítico de acertos $k_{\text{crit}}$ para um nível de significância $\alpha$ (padrão $\alpha = 0.05$) é o menor inteiro tal que:
$$P(K \ge k_{\text{crit}} \mid H_0) = \sum_{j=k_{\text{crit}}}^{N} \binom{N}{j} p_0^j (1 - p_0)^{N - j} \le \alpha$$

### 1.2 Tabela Comparativa de Nível de Acaso Exato ($\alpha = 0.05, p_0 = 0.5$)

| Trials ($N$) | Acertos Mínimos ($k_{\text{crit}}$) | Acurácia Crítica Exata | $p$-valor no Ponto | Aprox. Normal Wald ($p_0 + z_{1-\alpha}\sqrt{\frac{p_0(1-p_0)}{N}}$) | Erro da Aproximação |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10** | 9 | **90.0%** | $0.0107$ | 76.0% | $-14.0\%$ (falso positivo se usar Wald) |
| **20** | 15 | **75.0%** | $0.0207$ | 68.4% | $-6.6\%$ |
| **30** | 20 | **66.7%** | $0.0494$ | 65.0% | $-1.7\%$ |
| **40** | 26 | **65.0%** | $0.0403$ | 63.0% | $-2.0\%$ |
| **60** | 37 | **61.7%** | $0.0462$ | 60.6% | $-1.1\%$ |
| **100** | 59 | **59.0%** | $0.0443$ | 58.2% | $-0.8\%$ |
| **200** | 112 | **56.0%** | $0.0476$ | 55.8% | $-0.2\%$ |

**Implicações Práticas:**
1. Em sessões preliminares com $N=20$ trials, obter $14/20 = 70\%$ de acurácia **NÃO É estatisticamente significante** ($p = 0.0577$). O modelo precisa de no mínimo $15/20 = 75\%$ para rejeitar o acaso ao nível de 5%.
2. A aproximação normal subestima drasticamente a exigência de acerto para $N < 50$, gerando falsos positivos na literatura se o teste exato não for aplicado.
3. No caso multiclasse ($C$ classes, $p_0 = 1/C$), a função `estat_bci.acerto_minimo(n, n_classes, alfa)` calcula o limiar exato para qualquer $C$.



---

## 2. Decodificação Contínua de Trajetórias: Métricas e Armadilhas

### 2.1 Pearson Correlation Coefficient (PCC) por Trial vs. Global
Ao avaliar a decodificação de trajetórias contínuas 3D ($X, Y, Z$), o Pearson Correlation Coefficient é a métrica padrão. No entanto, sua aplicação direta aos dados concatenados introduz graves vieses.

- **PCC Global Concatena Trials:** Concatena todos os trials em um único vetor longo $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{N \cdot T}$. Se o participante realiza movimentos com offset constante ou deslocamento gradual de linha de base, a correlação entre as médias dos trials infla artificialmente o PCC global, mascarando decodificações intra-trial espúrias.
- **PCC por Trial (Honesto):**
  $$r_{i, d} = \frac{\sum_{t=1}^T (y_{i,t,d} - \bar{y}_{i,\cdot,d})(\hat{y}_{i,t,d} - \bar{\hat{y}}_{i,\cdot,d})}{\sqrt{\sum_{t=1}^T (y_{i,t,d} - \bar{y}_{i,\cdot,d})^2} \sqrt{\sum_{t=1}^T (\hat{y}_{i,t,d} - \bar{\hat{y}}_{i,\cdot,d})^2}}$$
  O desempenho da sessão é a média dos coeficientes intra-trial: $\bar{r}_d = \frac{1}{N}\sum_{i=1}^N r_{i, d}$.

### 2.2 O Baseline Trivial: Alvo Médio da Condição
Em paradigmas de alcance a alvos pré-definidos (e.g., reaching motor), o participante realiza trajetórias com perfil de velocidade em sino e curvas suaves semelhantes entre repetições para o mesmo alvo.
- Se um modelo ingênuo simplesmente emitir **a trajetória média observada no treino**, o PCC obtido com o sinal real pode atingir valores tão altos quanto **0.70 a 0.90** em eixos onde a trajetória média varia monotonicamente.
- **Regra de Ouro:** O modelo decodificador de EEG só demonstra aprendizado útil de informação neural se o seu PCC superar o baseline gerado pela emissão da média de cada alvo (`estat_bci.pcc_alvo_medio`).
- A comparação pareada entre o vetor de PCC do modelo e o vetor de PCC do baseline médio deve ser realizada com teste de Wilcoxon com sinal pareado (`estat_bci.comparacao_pareada`) ou teste de permutação exato.

### 2.3 Teste de Permutação com Troca de Trials (Trial-Swap Permutation)
Para testar a hipótese nula $H_0$: *"A previsão do modelo no trial $i$ não contém informação temporal específica do EEG do trial $i$, beneficiando-se apenas da cinemática média da condição"*:
1. Mantém-se as previsões do modelo $\hat{\mathbf{Y}} \in \mathbb{R}^{N \times T \times 3}$.
2. Em cada uma das $B$ iterações de Monte Carlo (e.g., $B = 5\,000$), permuta-se a ordem dos trials alvo $\mathbf{Y}_{\pi} = \mathbf{Y}[\pi]$, onde $\pi$ é uma permutação aleatória de $\{1, \dots, N\}$.
3. Se houver blocos de gravação ou múltiplos alvos, a permutação é restrita **dentro de cada bloco** (stratified permutation).
4. Calcula-se a estatística de teste nula $\bar{r}^{(b)}$.
5. O $p$-valor não-paramétrico exato é dado por:
   $$p = \frac{1 + \sum_{b=1}^B \mathbb{I}(\bar{r}^{(b)} \ge \bar{r}_{\text{obs}})}{1 + B}$$
   *(A adição de $+1$ no numerador e denominador é matematicamente obrigatória para evitar $p=0$ e garantir teste exato sob a hipótese nula).*


---

## 3. Testes Baseados em Clusters (Cluster-Based Permutation Tests)

### 3.1 Motivação e Mecânica (Maris & Oostenveld, 2007)
Ao comparar séries temporais de sinais neurais (e.g., potenciais corticais lentos MRCP, curvas de potência ERD/ERS em canais e pontos de tempo), o número de testes pontuais simultâneos causa inflação catastrófica do erro Tipo I (Family-Wise Error Rate). Métodos tradicionais como Bonferroni são excessivamente conservadores devido à forte correlação espaço-temporal dos sinais biológicos.

O teste de permutação baseado em clusters resolve isso sem suposições paramétricas de gaussianidade:
1. Em cada ponto temporal $t$, calcula-se uma estatística bicaudal (e.g., $t$-score pareado entre Condição A e Condição B).
2. Selecionam-se pontos onde $|t| \ge t_{\text{limiar}}$ (formando candidatos contíguos no tempo).
3. Agrupam-se pontos adjacentes em clusters e calcula-se a massa do cluster:
   $$M_k = \sum_{t \in \text{Cluster}_k} |t|$$
   A estatística da amostra é a massa máxima de cluster: $M_{\text{max}} = \max_k M_k$.
4. Sob $H_0$, as condições A e B são permutadas (inversão de sinal pareada por sujeito/trial com probabilidade 0.5) $B$ vezes, recalculando-se $M_{\text{max}}^{(b)}$ em cada iteração para construir a distribuição nula do máximo.
5. Um cluster observado é significante se sua massa exceder o percentil $(1-\alpha)$ da distribuição nula de máximos.

### 3.2 A Advertência Crítica de Sassenhagen & Draschkow (2019)
> **"Cluster-based permutation tests on event-related potentials do not establish significance of onset, latency, or spatial focus."** (Psychophysiology, 2019).

**O que o teste FAZ:**
- O teste avalia a hipótese nula global de intercambiabilidade: *"Existe uma diferença entre as condições A e B em algum ponto do domínio temporal"*.
- Rejeitar $H_0$ garante que a diferença é real e controla a taxa de falsos positivos na família inteira de pontos.

**O que o teste NÃO FAZ (e NUNCA deve ser alegado em teses/artigos):**
- **Limites de início/fim:** O início do cluster significante (e.g., $t = 180\text{ ms}$) **NÃO É** a latência de início (*onset latency*) do efeito neural. O limiar $t_{\text{limiar}}$ é arbitrário; limiares diferentes alteram o início do cluster sem alterar a presença do efeito subjacente.
- **Resolução pontual:** Não se pode afirmar que um instante específico dentro do cluster é estatisticamente diferente de zero isoladamente.
- **Diretriz de redação:** Reportar a extensão do cluster como *"intervalo temporal do cluster abrangendo de $t_1$ a $t_2$, indicando efeito distribuído de modulação motora"*, abstendo-se estritamente de usar termos como *"o efeito começou em $t_1$"*.


---

## 4. Correção para Comparações Múltiplas: Holm vs. Benjamini-Hochberg

Ao avaliar múltiplos canais, múltiplos tempos, 21 nós da mão ou múltiplos métodos de decodificação:

### 4.1 Holm-Bonferroni (Controle Estrito de FWER)
Controla a taxa de erro por família de testes ($\text{FWER} \le \alpha$):
1. Ordenam-se os $p$-valores brutos em ordem crescente: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
2. Compara-se cada $p_{(i)}$ com $\frac{\alpha}{m - i + 1}$.
3. O $p$-valor ajustado monotonicamente é dado por:
   $$\tilde{p}_{(i)} = \max_{k \le i} \left( \min( (m - k + 1) \cdot p_{(k)}, 1.0 ) \right)$$
4. A rejeição é sequencial (step-down). Se o critério falhar para o índice $i$, todas as hipóteses subsequentes $j > i$ são retidas.

### 4.2 Benjamini-Hochberg (FDR - False Discovery Rate)
Indicado para varredura exploratória e famílias maiores (e.g., 21 nós anatômicos do esqueleto da mão), onde tolera-se uma proporção controlada $q \le \alpha$ de falsos positivos entre as descobertas.
1. Ordenam-se os $p$-valores: $p_{(1)} \le \dots \le p_{(m)}$.
2. O limiar crítico é $p_{(k)} \le \frac{k}{m} \alpha$.
3. Os $p$-valores ajustados monotonicamente são calculados em ordem reversa (step-up):
   $$\tilde{p}_{(i)} = \min \left( \tilde{p}_{(i+1)}, \frac{m}{i} p_{(i)} \right)$$

---

## 5. Tamanhos de Efeito e Intervalos de Confiança

$p$-valores sozinhos não informam a magnitude biológica ou geométrica do achado. Todo relatório do pipeline deve incluir tamanho de efeito e intervalo de confiança a 95%.

### 5.1 Cohen's $d_z$ para Amostras Pareadas
Para comparar dois métodos no mesmo conjunto de sujeitos/trials ($D_i = x_{1,i} - x_{2,i}$):
$$d_z = \frac{\bar{D}}{s_D} = \frac{\frac{1}{N}\sum D_i}{\sqrt{\frac{1}{N-1}\sum (D_i - \bar{D})^2}}$$

### 5.2 Hedges' $g$ (Correção para Amostras Pequenas)
O $d$ de Cohen é viesado para cima quando $N < 20$. O $g$ de Hedges aplica o fator de correção não-paramétrico exato via função Gamma:
$$J(df) = \frac{\Gamma(df / 2)}{\sqrt{df / 2} \cdot \Gamma((df - 1) / 2)} \approx 1 - \frac{3}{4\,df - 1}$$
$$g = d \cdot J(df)$$

### 5.3 Intervalos de Confiança
- **Para Médias e Diferenças Arbitrárias:** Percentile Bootstrap com $B \ge 2\,000$ reamostragens (`estat_bci.ic_media_bootstrap`). Não assume distribuição gaussiana.
- **Para Coeficientes de Correlação de Pearson:** Transformação $z$ de Fisher:
  $$z = \text{arctanh}(r) = \frac{1}{2} \ln \left(\frac{1+r}{1-r}\right)$$
  $$\text{SE}_z = \frac{1}{\sqrt{N - 3}}$$
  $$\text{IC}_z = z \pm z_{1-\alpha/2} \cdot \text{SE}_z \implies \text{IC}_r = \tanh(\text{IC}_z)$$

---

## 6. Information Transfer Rate (ITR) Prático

A métrica padrão para taxa de transmissão em BCI é o ITR de Wolpaw (bits/minuto):
$$B = \log_2(N) + P \log_2(P) + (1 - P) \log_2\left(\frac{1 - P}{N - 1}\right) \quad [\text{bits/seleção}]$$
$$\text{ITR} = B \cdot \left(\frac{60}{T}\right) \quad [\text{bits/minuto}]$$
Onde $N$ é o número de classes possíveis, $P$ é a acurácia de classificação equilibrada ($0 \le P \le 1$), e $T$ é a duração total da seleção em segundos (tempo de foco da janela + pausas entre estímulos).

**Exigências Metodológicas:**
1. Se $P \le 1/N$ (desempenho ao acaso ou abaixo), $B = 0$ bits.
2. O tempo $T$ **deve incluir o intervalo entre estímulos (inter-trial interval)**. Ignorar o ITI e considerar apenas a janela ativa de EEG infla artificialmente o ITR em até 300%.

---

## 7. Validação do Tracking de Câmeras e Overlay

No arquivo `overlay_deviation_log.csv`, registram-se amostras temporais com desvios em pixels entre os nós do MediaPipe e a projeção 3D dos nós do esqueleto.

### 7.1 Dependência Temporal entre Frames Sucessivos
- Vídeos gravados a 30 Hz possuem forte autocorrelação temporal: o frame $k+1$ é quase idêntico ao frame $k$.
- Tratar cada frame como um grau de liberdade independente em testes $t$ paramétricos é violação clássica de premissa (pseudo-replicação).
- **Abordagem Correta:**
  1. Comparar as fontes (Kinect vendo vs. Kinect cego/reserva) através de testes de permutação e tamanho de efeito $d$, reportando explicitamente como contraste indicativo de robustez da fusão, e não como amostras gaussianas independentes.
  2. Ajustar os 21 nós da mão conjuntamente via Benjamini-Hochberg FDR para controlar falsos alarmes em nós periféricos das pontas dos dedos.

---

## 8. Guia de Uso da Ferramenta de Linha de Comando

O utilitário `tools/analise_estatistica.py` fornece interface direta para todos os modos:

### Nível de Acaso Exato
```bash
# Para um resultado experimental específico (e.g. 15 acertos em 20 trials, 2 classes)
python tools/analise_estatistica.py --acaso 15/20

# Tabela do limiar crítico exato para múltiplos tamanhos de amostra
python tools/analise_estatistica.py --acaso-tabela 10 20 30 40 60 100
```

### Análise de Previsões de Trajetória do SAND
```bash
# npz contendo chaves 'previsao' (N,T,3) e 'alvo' (N,T,3)
python tools/analise_estatistica.py --pcc-npz resultados_validacao.npz --n-perm 5000 --json relatorio_pcc.json
```

### Log de Desvio do Overlay de Câmeras
```bash
# Analisa desvios médios por nó com FDR e contraste Kinect vs. Backup
python tools/analise_estatistica.py --overlay-log overlay_deviation_log.csv
```

### Comparação de Modelos e Benchmarks (Friedman + Holm)
```bash
# CSV longo com colunas 'sujeito', 'metodo', 'pcc'
python tools/analise_estatistica.py --pcc-tabela benchmark_modelos.csv
```

### Auto-Teste Integrado do Pipeline
```bash
python tools/analise_estatistica.py --auto-teste
```



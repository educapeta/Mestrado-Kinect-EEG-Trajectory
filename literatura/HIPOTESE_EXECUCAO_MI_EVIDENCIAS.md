# Evidências para a hipótese “execução como seeding da MI”

## Hipótese da tese

> A execução motora pode ser usada como supervisão auxiliar/referência para
> aprender uma representação de pose que seja decodificável a partir de MI,
> reduzindo a dependência de labels de pose medidos durante a MI.

Ela tem três componentes que precisam ser sustentados separadamente:

1. **MI e execução compartilham circuitos** (base neurofisiológica);
2. **é possível decodificar cinemática/trajetória de MI** (base BCI);
3. **execução antes da MI pode servir de priming/referência** (base experimental).

Nenhum artigo prova a hipótese completa no cenário desta tese. A contribuição é
justamente testá-la com pose final + orientação + abertura.

## Evidências por componente

| Artigo | O que sustenta | O que NÃO sustenta |
| --- | --- | --- |
| E2T (Kwon et al., 2026, DOI 10.1109/TNSRE.2026.3683431) | ME seguida de MI; previsão de velocidade 3D na MI contra trajetória de referência da ME; priming por execução | Não mede pose na MI; a referência da MI é a ME; não prova replay temporal exato |
| SAND (Fu et al., 2026, DOI 10.1109/TBME.2026.3694935) | Decodificação de trajetória 2D/3D a partir de EEG de execução (WAY-EEG-GAL); arquitetura espectral+Transformer | Não é MI; não prova transferência execução→MI |
| M3T-attention (2026, DOI 10.1007/s11571-025-10403-1) | Transformer multi-escala para trajetória 3D de mão; referência de arquitetura | Execução; não resolve seeding de execução para MI |
| MTRT (Wang et al., 2023, DOI 10.1109/TNSRE.2023.3275172) | Reconstrução de trajetória com restrições geométricas; posições de ombro/cotovelo/punho | Sem orientação; execução, não MI |
| Decoding Imagined 3D Arm Movement Trajectories (2019, DOI 10.3389/fnbot.2019.00094) | MI de trajetórias 3D de braço pode ser decodificada; controle de braços virtuais | Pilot; não é pose de mão/garra; não resolve execução→MI |
| A comprehensive review on motion trajectory reconstruction (2023, DOI 10.3389/fnins.2023.1086472) | Revisão do estado da arte e limitações de decodificação de trajetória | Não valida a hipótese específica |
| Kinesthetic, but not visual, motor imagery modulates corticomotor excitability (2006, DOI 10.1007/s00221-005-0078-y) | MI cinestésica recruta M1; MI visual não é equivalente | Não é decoding; justifica usar instrução cinestésica e evitar vídeo detalhado |
| Imagery of voluntary movement... (2003, DOI 10.1152/jn.01113.2002) | Imaginar movimento ativa representações motoras body-part-specific semelhantes à execução | fMRI; não garante decodificação em EEG |
| Motor Imagery and Motor Execution: A Narrative Review (2025/2026, PubMed 41141194) | Revisão de assinaturas EEG, diferenças metodológicas e aplicações translacionais | Não é um experimento de seeding |
| Toward Practical Wrist BCIs: actual and imagined movements (2025, DOI 10.1109/ACCESS.2025.3630223) | Classificação de movimentos reais e imaginados de punho | Classificação; não pose contínua |
| IACKD dataset (2026, DOI 10.1038/s41597-026-07146-x) | Dataset EEG + cinemática de mão em controle unimanual; modelo de dados | Não é MI; útil para desenho de labels/QC |

## O que o E2T realmente mostra

- desenho: trial com ME seguida de MI, quatro direções 3D;
- referência: trajetória medida na ME;
- alvo: velocidade 3D; a MI é avaliada contra a referência da ME;
- resultado: correlação média ~0,77 (ME) e ~0,71 (MI);
- limitação: não há medida de movimento durante a MI; a correspondência
  MI↔ME é uma **suposição experimental**, reforçada pelo priming e pelo
  paradigma fixo.

Isso é suficiente para propor a hipótese, mas não para afirmar que a MI de
qualquer pessoa reproduz a ME.

## Como escrever isso na tese

Evite:

> “O E2T provou que a MI segue a execução.”

Prefira:

> “O E2T adota a trajetória da execução como referência para a MI e demonstra
> que um decodificador de velocidade 3D consegue reconstruir essa referência a
> partir de MI após priming por execução. Este trabalho estende essa lógica
> para pose final, orientação e abertura, e testa explicitamente a transferência
> execução→MI.”

## Lacunas que a tese pode atacar

1. E2T usa velocidade 3D; não pose completa/orientação/abertura.
2. E2T usa quatro direções e um braço; não três objetos e duas mãos.
3. E2T não mede a pose da MI; a tese pode comparar alvo instruído, pose final da
   ME e desempenho em MI.
4. E2T não testa generalização entre sujeitos/sessões com LOSO.
5. SAND não testa MI; a tese combina SAND-Pose + seeding por execução.

## Referências locais (Citavi)

PDFs correspondentes estão em
`C:\Users\Usuario\OneDrive\Desktop\Mestrado_Projeto\Citavi\Mestrado_BCI\Citavi Attachments\`.
Os principais: `E2T_...pdf`, `SAND_...pdf`, `MTRT_...pdf`,
`s11571-025-10403-1 (1).pdf` (M3T), `sdata201447.pdf` (WAY-EEG-GAL).

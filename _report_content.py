# -*- coding: utf-8 -*-
"""Textos do Relatorio Tecnico de Bolsas (PPGNeuro/UFSC) - 2o semestre.

Periodo: 01/03/2026 a 31/08/2026.
"""

TITULO = (
    "Classificação de Padrões Neurais em Eletroencefalograma (EEG) com Redes Neurais "
    "Convolucionais (CNN) para Aplicação em Interface Cérebro-Máquina de Mão Biônica"
)

RESUMO_PROPOSTA = [
    "O projeto desenvolve e avalia uma Interface Cérebro-Máquina (ICM) não invasiva "
    "baseada em eletroencefalografia (EEG) para o controle de uma prótese de mão biônica "
    "de código aberto, com o objetivo de ampliar a funcionalidade e a qualidade de vida "
    "de pessoas com amputação de membro superior.",
    "A proposta original combinava EEG e rastreamento ocular (eyetracking). Após análise "
    "de viabilidade, o eyetracking foi suspenso e o esforço concentrado na modalidade "
    "EEG — decisão aprovada pelo orientador e registrada no relatório do 1º período. O "
    "objetivo central foi mantido: decodificar a intenção motora a partir do EEG e "
    "traduzi-la em comando para a prótese.",
    "A aquisição é feita com o amplificador g.Nautilus (g.tec, 32 canais, 500 Hz), em "
    "paradigma de 12 s por trial com seis condições (objeto × mão) e duas fases por trial "
    "— execução motora (ME) e imagética motora (MI) —, precedidas de vídeo de priming. A "
    "verdade de referência (ground truth) do movimento é medida por um sistema "
    "multimodal construído no próprio projeto: Kinect v2 (RGB + profundidade), webcams "
    "auxiliares calibradas em estéreo e duas luvas inerciais (ESP32 + MPU6050) fixadas no "
    "dorso das mãos.",
]

ALTERACOES_DIFICULDADES = [
    "1) Evolução do objeto de modelagem (ajuste metodológico). A proposta inicial previa "
    "um classificador discreto de padrões de imagética motora (CNN do tipo EEGNet, três "
    "classes). A revisão sistemática conduzida no período mostrou que, para o controle "
    "protético, a formulação de maior valor clínico é a predição de trajetória contínua "
    "(e não a decisão discreta). A arquitetura foi reformulada para uma rede com atenção "
    "esparsa e camadas convolucionais (SAND), que mapeia janelas de EEG para a trajetória "
    "tridimensional da mão (30 pontos × 3 eixos, em metros). O título do projeto foi "
    "mantido, mas o alvo passou de detecção para trajetória. A mudança foi acompanhada de "
    "validação em base pública (BCI Competition IV – Graz 2a/2b), da definição de alvos "
    "alternativos (posição e velocidade) e de um filtro de Kalman que funde a velocidade "
    "prevista com a aceleração medida pelo MPU6050.",
    "2) Dependência da aprovação do Comitê de Ética em Pesquisa (CEP). A documentação "
    "ética (TCLE versão 5, projeto detalhado, folha de rosto, declaração institucional "
    "com base na Resolução 510/2016 e modelo de anonimização de dados) foi consolidada e "
    "submetida entre abril e maio de 2026. A coleta com participantes não pôde ser "
    "iniciada dentro do período e foi deslocada para o próximo semestre. Para não "
    "interromper o desenvolvimento, o pipeline foi validado com sessões sintéticas "
    "geradas pelo próprio projeto e com base pública.",
    "3) Complexidade do sistema de verdade (ground truth). Obter posição e orientação da "
    "mão com precisão sub-centimétrica exigiu o desenvolvimento de uma cadeia própria de "
    "calibração e diagnóstico: calibração estéreo Kinect × webcam auxiliar com rejeição "
    "de poses inconsistentes, alinhamento do esqueleto da mão (MediaPipe) entre câmeras, "
    "tratamento das limitações do fluxo de profundidade do Kinect e fusão com duas luvas "
    "inerciais. Todas as causas-raiz e correções estão documentadas no arquivo "
    "HISTORICO_ITERACOES.txt do repositório.",
    "4) Infraestrutura. O amplificador g.Nautilus é utilizado em regime de colaboração "
    "com o Instituto de Engenharia Biomédica (IEB/UFSC), o que impôs janelas de "
    "disponibilidade para os testes de bancada.",
]

RESULTADOS = [
    "a) Qualificação de mestrado. A qualificação foi realizada em 26/03/2026, com "
    "apresentação do projeto detalhado à banca examinadora, cumprindo etapa regimental "
    "do Programa.",
    "b) Documentação ética consolidada e submetida. Foram produzidos e submetidos o TCLE "
    "(versão 5), o projeto detalhado, a folha de rosto, a declaração institucional "
    "(Resolução 510/2016) e o modelo de anonimização de dados. A submissão ocorreu em "
    "maio de 2026; o parecer está pendente, o que condiciona o início da coleta humana.",
    "c) Sistema multimodal de verdade (ground truth) concluído e validado. Calibração "
    "estéreo Kinect × webcam auxiliar com erro de reprojeção reduzido de 30–70 px "
    "(estado inicial) para 0,58–1,45 px, após correção da ordenação canônica dos cantos "
    "do tabuleiro e uso de RANSAC com atribuição de poses por solvePnP; alinhamento "
    "nó-a-nó entre o esqueleto da mão da câmera auxiliar e o do Kinect; correção do "
    "fluxo de profundidade do Kinect na camada de aquisição; e firmware versionado das "
    "duas luvas inerciais (ESP32 + MPU6050, filtro de Kalman e transmissão UDP).",
    "d) Programa de aquisição offline concluído. Sessão de 12 s por trial, cinco blocos "
    "de 50 trials, painel de impedâncias ao vivo, vigia de perda de enlace do EEG, vídeos "
    "de priming com a trajetória 3D sobreposta e questionário do participante. O EEG é "
    "gravado cru a 500 Hz (32 canais + marcador) em arquivo separado do movimento "
    "(~30 Hz), e a sincronização EEG↔movimento é feita por reta entre a amostra exata de "
    "cada evento e o relógio monotônico comum aos dois arquivos.",
    "e) Rede de trajetória (SAND) implementada e validada. Modelo, treino e inferência "
    "assíncrona em tempo real, com filtro de Kalman para reconstrução de posição, "
    "regularizador anatômico (envelope de alcance e limites articulares) e alvos de "
    "posição e velocidade. Validação em base pública (BCI Competition IV) e geração de "
    "sessões sintéticas que permitem executar todo o pipeline sem hardware.",
    "f) Verificação automatizada. Aproximadamente 40 suítes de teste cobrem protocolo "
    "experimental, calibrações, fusão inercial, alvos, treino, overlay e cancelamento de "
    "sessão, todas executáveis sem hardware.",
    "g) Testes de bancada com hardware real. Registros simultâneos de g.Nautilus e das "
    "luvas ESP32/MPU6050 realizados em agosto de 2026, comprovando a integração física da "
    "cadeia de aquisição.",
]

CONCLUSOES = [
    "O período correspondeu ao fechamento da etapa de construção do algoritmo. O sistema "
    "de verdade (ground truth) multimodal está operacional, o programa de aquisição "
    "offline está completo e a rede de trajetória está implementada e validada em base "
    "pública e em sessões sintéticas. A qualificação foi concluída em março de 2026 e a "
    "documentação ética foi submetida em maio de 2026.",
    "O principal fator que limitou o avanço completo do cronograma foi a dependência do "
    "parecer do CEP, que deslocou a aquisição de dados humanos para o próximo semestre. "
    "Esse deslocamento não comprometeu o desenvolvimento: as decisões de projeto, a "
    "calibração e a verificação foram integralmente exercitadas com dados sintéticos e "
    "públicos, de modo que a entrada em coleta ocorrerá sobre um pipeline já testado. "
    "Considerando o estágio atual, o cronograma permanece compatível com o prazo "
    "regimental de defesa.",
]

PERSPECTIVAS = [
    "Para o próximo período (3º semestre), as metas são:",
    "• Concluir a tramitação ética e iniciar a aquisição offline de EEG e movimento com "
    "participantes saudáveis, seguindo o protocolo já definido (seis condições, cinco "
    "blocos de 50 trials).",
    "• Pré-treinar a rede de trajetória em base pública de longo prazo (WAY-EEG-GAL) e "
    "treinar com os dados reais coletados, incluindo avaliação leave-one-subject-out.",
    "• Refinar o filtro de Kalman e o regularizador anatômico com os dados reais e "
    "avaliar o desempenho da inferência em tempo real.",
    "• Integrar a saída da rede ao protótipo de mão biônica e conduzir testes piloto de "
    "usabilidade.",
    "• Redigir e submeter artigo científico com a metodologia e os primeiros resultados.",
    "A necessidade técnica desta etapa é direta: sem dados reais de EEG sincronizados ao "
    "movimento medido, não é possível validar a decodificação de trajetória nem calibrar "
    "o sistema para uso online. Os dados do próximo semestre são, portanto, o insumo "
    "essencial para a dissertação.",
]

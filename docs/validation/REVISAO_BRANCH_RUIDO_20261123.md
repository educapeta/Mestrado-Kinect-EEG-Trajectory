# Revisão independente — branch `resultados/ruido-eeg-20261123`

**Data:** 23/11/2026
**Branch:** `resultados/ruido-eeg-20261123`
**Commit de código:** `7343180`

## 1. O que foi verificado

- README e JSONs das duas sessões;
- recálculo de 60 Hz e correlação 59–61 Hz diretamente nas 8 amostras de 10 s;
- confirmação do congelamento da sessão 1 em C/D/E.

## 2. Integridade

| sessão | condições válidas | observação |
|---|---|---|
| 17:02 | A, B | C/D/E com std = 0 (stream congelado) |
| 17:15 | C, D, E | sinal real; sem congelamento |

O congelamento da sessão 1 foi confirmado: C/D/E têm std = 0 e todos os canais
repetem os mesmos valores. **Essas três condições não devem ser usadas.**

## 3. Recálculo independente nas amostras de 10 s

| amostra | std (µV) | RMS60 (µV) | corr 59–61 Hz |
|---|---:|---:|---:|
| sessao1_A | 625,0 | 216,8 | +0,995 |
| sessao1_B | 1977,1 | 1620,4 | +0,995 |
| sessao1_C/D/E | 0,0 | 0,0 | congelado |
| sessao2_C | 2515,2 | 1138,6 | +0,996 |
| sessao2_D | 2031,5 | 1718,5 | +0,997 |
| sessao2_E | 478,9 | 108,6 | +0,990 |

A correlação de +0,99 na faixa 59–61 Hz confirma **interferência de rede em modo
comum**. Os valores de 10 s não batem exatamente com os 60 s (como o README já
avisa), mas a direção é a mesma.

## 4. Achados que se sustentam

### 4.1 Bateria × tomada (D → E, mesma posição)

| | D (tomada) | E (bateria) | fator |
|---|---:|---:|---:|
| RMS60 | 1727,1 µV | 96,7 µV | **17,9× menor** |
| RMS50 | 5,35 µV | 1,84 µV | 2,9× menor |

O efeito é **grande e coerente**: a bateria reduz drasticamente o 60 Hz. Isso é
compatível com loop de terra/fonte chaveada.

Ressalva: D e E não tiveram estado da tela registrado; a comparação é interna à
sessão 2, mas ainda carece de repetição.

### 4.2 Proximidade do desktop

| comparação | longe | perto | fator |
|---|---:|---:|---:|
| A → B (bateria) | 246,9 | 1129,5 | **4,6× pior perto** |
| C → D (tomada) | 957,6 | 1727,1 | **1,8× pior perto** |

As duas comparações internas apontam na mesma direção: perto do desktop piora.
Mas A/B e C/D são sessões diferentes; a magnitude não deve ser comparada entre
sessões.

### 4.3 Banda larga 40–80 Hz

O B (bateria + perto) teve PSD 40–55 = 200,6 e 65–80 = 158,7, muito acima de
A (3,7 / 2,2). **Não se repetiu na sessão 2** (2,0–7,5). Portanto, é um achado
isolado, possivelmente movimento/cabo/postura; não deve ser generalizado.

## 5. Limitações metodológicas

1. **Não houve uma sessão única A–E**: A/B e C/D/E são sessões diferentes; A×C e
   B×D cruzam sessões e têm drift 3–5× diferente.
2. **Estado da tela não registrado** na sessão 2; E não isola tela de desktop.
3. **Uma única exposição por condição**; não há repetição/randomização.
4. **Ordem fixa** A→B→C→D→E; efeitos de ordem/fadiga não controlados.
5. Sessão 1 congelou sem vigia de enlace; a aquisição não detectou.
6. O `std` cru é dominado por drift de baixa frequência; para 60 Hz, usar
   `rms_60` e correlação 59–61 Hz, não a correlação crua.

## 6. Próximo teste recomendado

1. Uma sessão única com **A–E repetidos 2–3 ciclos**, ordem randomizada,
   pausas curtas.
2. Registrar em cada condição: bateria/tomada, tela ligada/desligada, distância.
3. Adicionar **F**: bateria + perto + tela desligada (para separar de E).
4. Repetir D↔E alternando pelo menos 3 vezes para confirmar o fator ~18×.
5. Implementar **vigia de enlace**: após cada condição, verificar
   `std > 0` e fluxo de amostras; abortar/avisar se congelar.
6. Só depois decidir se o ambiente está aceitável para o MI.

## 7. Conclusão

A branch é útil e bem documentada. O achado mais forte é a redução de ~18× do
60 Hz na bateria, coerente com problema de terra/fonte. Mas ainda há ~97 µV de
60 Hz na melhor condição, então **a bateria melhora, mas não resolve
completamente**. A comparação A–E completa ainda precisa ser refeita em uma
única sessão, com repetição e registro do estado da tela.

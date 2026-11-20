# Offset marcador × EEG: teste exploratório

**Data:** 19/11/2026
**Fonte:** `piloto_me_mi_s01` + `s02`.
**Script:** `src/mestradopy/tools/analisa_offset_marcador.py`

## 1. JSON vs CSV

Comparei os tempos de `mi_start` no JSON (relativos a `session_start`) com os
marcadores 780 gravados no CSV:

| sessão | diferença mediana | min–max | desvio | diferença dos intervalos |
|---|---:|---:|---:|---:|
| s01 | +0,922 s | 0,902–0,932 | 0,006 s | ~1 ms |
| s02 | +0,882 s | 0,869–0,891 | 0,005 s | ~1 ms |

A diferença constante de ~0,9 s é compatível com a gravação começar antes do
`session_start` registrado no JSON. **Não há deriva**: os intervalos entre
marcadores batem com os do JSON em ~1 ms.

## 2. Varredura com janela de 4 s

| offset | BA |
|---:|---:|
| 0,0 s | 0,475 |
| +0,6 s | 0,593 |
| +0,7 s | 0,619 |
| +1,0 s | **0,627** |

Problema: com janela de 4 s, offsets positivos fazem a janela **passar do fim
da MI e entrar no ITI**. Não é um teste válido de offset.

## 3. Varredura com janela de 2 s

| offset | BA | leitura |
|---:|---:|---|
| −1,0 s | **0,675** | janela entra na ME/transição — **inválido para MI** |
| −0,5 s | 0,492 | transição |
| 0,0 s | 0,458 | MI inicial |
| +0,4 s | 0,592 | MI inicial/média |
| +1,7 s | 0,600 | MI final |
| +2,0 s | 0,567 | MI final |

Dentro da MI verdadeira (offset 0 a +2 s), o máximo foi ~0,60 em +1,7 s, sem
pico em zero; o early (0 s) ficou em 0,458.

## 4. Interpretação

1. **Não há evidência de offset constante entre marcador e EEG**: intervalos
   JSON/CSV batem em ~1 ms e a varredura não tem um pico consistente em zero
   ou em um valor fixo.
2. **O ganho em offsets negativos é contaminação por ME/transição**: a janela
   “melhora” porque inclui o movimento executado, não MI.
3. **O ganho em offsets positivos grandes é contaminação por fase tardia/ITI**.
4. O problema dos primeiros 2 s de MI é **fisiológico/paradigmático**, não um
   simples deslocamento de marcador.

## 5. Conclusão

A hipótese de offset temporal **não é descartada por princípio**, mas nos dados
atuais ela **não se sustenta**. Qualquer melhora obtida deslocando a janela para
−1 s ou +1 s vem de incluir ME/ITI, não de alinhar a MI.

Para investigar de verdade, seria necessário um marcador externo independente
(flash/som registrado simultaneamente no EEG e no vídeo/estímulo) ou um teste de
latência do g.Pype. Sem isso, o teste de offset por classificação fica circular:
qualquer deslocamento que inclua ME ou ITI pode aumentar o BA.

## 6. Limitações

- varredura exploratória com múltiplos offsets;
- sem marcador externo de verdade;
- 2 sessões, 120 trials, 3 canais;
- janelas de 2 s com passo 0,5 s;
- o efeito ME não é uma medida de MI.

# Trajetória suavizada por landmark — kinect

Deslocamento físico: 50.0 cm; tolerância: 2.0 cm (mediana do erro).

Landmarks dentro da tolerância: **18/21**

| Landmark | Desloc. mediano | Erro mediano | Erro p95 | Caminho integrado | OK |
|---:|---:|---:|---:|---:|:---:|
| 0 | 0.4960 m | 0.75 cm | 10.53 cm | 0.5960 m | SIM |
| 1 | 0.5104 m | 1.34 cm | 10.35 cm | 0.6095 m | SIM |
| 2 | 0.5036 m | 1.35 cm | 11.15 cm | 0.6201 m | SIM |
| 3 | 0.5091 m | 1.44 cm | 11.98 cm | 0.6470 m | SIM |
| 4 | 0.5247 m | 3.40 cm | 15.60 cm | 0.7407 m | NÃO |
| 5 | 0.5192 m | 2.18 cm | 10.20 cm | 0.6087 m | NÃO |
| 6 | 0.5123 m | 1.65 cm | 10.62 cm | 0.6414 m | SIM |
| 7 | 0.5139 m | 1.64 cm | 11.13 cm | 0.6505 m | SIM |
| 8 | 0.5130 m | 1.59 cm | 10.61 cm | 0.6715 m | SIM |
| 9 | 0.5058 m | 1.22 cm | 10.22 cm | 0.5890 m | SIM |
| 10 | 0.5101 m | 1.81 cm | 10.57 cm | 0.6292 m | SIM |
| 11 | 0.5142 m | 2.05 cm | 10.17 cm | 0.6592 m | NÃO |
| 12 | 0.5085 m | 1.67 cm | 7.90 cm | 0.6873 m | SIM |
| 13 | 0.5013 m | 0.80 cm | 10.15 cm | 0.5870 m | SIM |
| 14 | 0.5064 m | 1.47 cm | 9.88 cm | 0.6129 m | SIM |
| 15 | 0.5106 m | 1.79 cm | 6.81 cm | 0.6613 m | SIM |
| 16 | 0.5078 m | 1.78 cm | 5.54 cm | 0.6793 m | SIM |
| 17 | 0.4967 m | 0.81 cm | 10.02 cm | 0.5887 m | SIM |
| 18 | 0.4977 m | 1.13 cm | 9.89 cm | 0.6168 m | SIM |
| 19 | 0.4972 m | 0.97 cm | 9.44 cm | 0.6127 m | SIM |
| 20 | 0.4969 m | 1.22 cm | 9.36 cm | 0.6386 m | SIM |

## Interpretação

- deslocamento = distância entre as medianas do hold inicial e final;
- caminho integrado = soma das distâncias entre amostras suavizadas;
- se o caminho for muito maior que 50 cm, há ruído/curvatura; se for muito menor, o movimento foi menor que o alvo.

# Teste AHRS 2x MPU6050 -- vetores 3D (item 142)

Firmware separado do firmware das luvas, baseado no projeto
[0015/ThatProject -- Esp32_MPU6050_MahonyARHS/MahonyAHRS_ESP32](https://github.com/0015/ThatProject/tree/master/Esp32_MPU6050_MahonyARHS/MahonyAHRS_ESP32).

Em vez do "dado" 3D do projeto original, este firmware publica os quaternions
**corpo -> mundo** dos dois MPUs na serial; o visualizador no PC desenha **um
vetor 3D por sensor**. Cada vetor é fixo no corpo do sensor e é "tareado" no
primeiro quadro: com o sensor deitado (face paralela à mesa), ele começa
apontando em **-Z (chão)** e gira junto com o sensor.

## Hardware / endereços
- Os dois MPU6050 no **mesmo barramento I2C** (como no firmware das luvas):
  - **0x68 = direita** (vetor laranja);
  - **0x69 = esquerda** (vetor ciano).
- O firmware varre pares de GPIO (robusto à fiação da bancada); na bancada a
  ordem é **SCL=D21 / SDA=D22**, mas o par encontrado é impresso no monitor.
- Se os endereços estiverem trocados, mude `build_flags` em `platformio.ini`.

## Gravar
Feche qualquer programa que use a COM4. Coloque a placa em download mode
(segure **BOOT**, aperte/solte **EN**, solte BOOT) e rode:
```powershell
cd C:\mestradopy
C:\Users\Usuario\.platformio\penv\Scripts\pio.exe run -d firmware\esp32_teste_ahrs -t upload
```
Depois de testar, volte ao firmware das luvas:
```powershell
C:\Users\Usuario\.platformio\penv\Scripts\pio.exe run -d firmware\esp32_duas_imus -t upload
```

## Rodar
1. Deixe as **duas IMUs paradas e com a face paralela à mesa** ao ligar: o boot
   calibra bias do giroscópio e offsets do acelerômetro (~4 s) e aquece o
   Mahony por ~1,5 s.
2. Abra o visualizador:
```powershell
cd C:\mestradopy
.\.venv\Scripts\python.exe tools\visualiza_vetores_ahrs.py --port COM4
```
   (ou `--simular` para ver a demonstração sem hardware; `--listar-portas`
   lista as portas.)
3. Gire/incline cada MPU. Os vetores (laranja = direita, ciano = esquerda)
   devem acompanhar a rotação física. O primeiro quadro zera ambos em **-Z**;
   a tecla **Z** re-tara a orientação atual.
4. Teclas: **Z** re-tarar; **C** recalibrar no ESP (IMUs paradas/deitadas);
   **R** reiniciar a atitude; **Q/ESC** sair.

## Protocolo serial (115200 baud)
```
# AHRS_VETOR v1 | q=corpo->mundo (w,x,y,z) | vetores zerados no 1o quadro
# A,t_us,qwL,qxL,qyL,qzL,qwR,qxR,qyR,qzR,axL,ayL,azL,axR,ayR,azR
A,1234567,1.000000,0.000000,...,0.0100,-0.0200,0.9900,...
```
`L` = esquerda (0x69), `R` = direita (0x68). Linhas `E,0,D` / `E,0,E`
indicam falha de leitura I2C de um lado.

## Interpretação / limitações
- É um AHRS de **6 eixos** (girômetro + acelerômetro), sem magnetômetro: o
  **yaw deriva** lentamente; isso é esperado. O importante é o tilt/roll e a
  coerência do vetor ao girar.
- Um vetor que começa exatamente no eixo vertical **não muda** ao girar em
  torno desse mesmo eixo vertical (física normal); incline/gire nos outros
  eixos para ver a rotação 3D.
- O vetor é **fixo no corpo** (como uma seta pintada no sensor). Ele não é uma
  direção de gravidade "sempre para baixo": ao inclinar o sensor, inclina junto.
- Se quiser medir a orientação absoluta da luva, o caminho continua sendo a
  calibração `T`/`f` com o Kinect (itens 127+), não este teste.

## Troubleshooting

- **A janela abre, mas fica "AGUARDANDO QUADROS A"**: a COM4 está enviando
  linhas `1,...`/`2,...` do **firmware das luvas**, não do teste. Feche o tool
  das luvas, grave `esp32_teste_ahrs` de novo (com BOOT) e confira com
  `pio device monitor -d firmware/esp32_teste_ahrs`: tem de aparecer
  `[ahrs] TESTE AHRS...`, a calibração e `# AHRS_VETOR v1`.
- **A janela não aparece**: rode primeiro `tools\visualiza_vetores_ahrs.py
  --simular --duracao 5` para testar o matplotlib. Se o simulado funcionar, a
  janela real pode ter aberto atrás do PowerShell. O script agora imprime o
  backend e as linhas cruas com `--debug`.
- **Porta ocupada / PermissionError**: feche o tool das luvas, o `pio device
  monitor` e qualquer outro programa na COM4; o CP210x desta bancada não
  aceita duas aberturas.
- **Só um vetor se mexe**: confira o endereço I2C (0x68/0x69) e a fiação; o
  monitor mostra o par SDA/SCL encontrado e os erros `E,0,D`/`E,0,E`.
- **Vetor deriva devagar**: normal em AHRS 6 eixos (sem magnetômetro); é yaw
  drift. Use `Z` para re-tarar.

## Modo console (sem janela)

Se a janela 3D nao aparecer (Qt/Tcl/ambiente grafico), use `--texto`: ele
imprime os dois vetores no console a cada 0,25 s, sem matplotlib:
```powershell
.\.venv\Scripts\python.exe tools\visualiza_vetores_ahrs.py --port COM4 --texto --duracao 15
```
A primeira linha `[texto] tara aplicada...` aparece quando a calibracao termina;
depois cada linha mostra `L=(x,y,z) ang=... | R=(...) | qL=... | qR=...`.
Gire cada MPU e veja os valores mudarem. Ctrl+C encerra.
Depois tente de novo a janela:
```powershell
.\.venv\Scripts\python.exe tools\visualiza_vetores_ahrs.py --port COM4
```
(aguarde ~8-10 s apos o reset; o boot calibra antes dos quadros `A`).
Se `--backend tkagg` falhar com erro de Tcl, use o modo janela padrao (Qt).

## Visualizador
`tools/visualiza_vetores_ahrs.py` usa a matemática de
`tools/ahrs_vetores_core.py` (também testada por `test_ahrs_vetores.py`).
O eixo Z do mundo aponta para dentro da cena; **-Z = chão**. O piso cinza em
z=-1,05 serve de referência.

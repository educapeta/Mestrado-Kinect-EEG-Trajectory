# AHRS -- teste de orientação dos dois MPU6050

Tudo o que é do teste AHRS está nesta pasta. O resto de `C:\mestradopy` é o
projeto das luvas/Kinect.

## Estrutura
```
AHRS/
  README.md                  <- este arquivo
  AHRS_menu.bat              <- menu principal (1..7)
  AHRS_texto.bat             <- vetores no console (sem janela)
  AHRS_grafico.bat           <- janela 3D OpenCV (com console)
  AHRS_grafico_sem_console.bat <- janela 3D OpenCV sem console
  AHRS_grafico_log.bat       <- janela 3D gravando log CSV
  preview.png                <- como a janela é
  logs/                      <- ahrs_*.csv (ignorados pelo git)
  firmware/esp32_teste_ahrs/ <- projeto PlatformIO (Mahony 6 eixos)
  tools/visualiza_vetores_ahrs.py
  tools/ahrs_vetores_core.py
  tools/analisa_log_ahrs.py
  test_ahrs_vetores.py
```

## Uso rápido
1. Gravar o firmware de teste (precisa BOOT):
   `AHRS_menu.bat` -> opção 4, ou
   `pio run -d AHRS\firmware\esp32_teste_ahrs -t upload`.
2. Deixar as duas IMUs **paradas e deitadas** no boot (calibração ~4 s).
3. Abrir `AHRS_menu.bat`:
   - opção 2: janela 3D OpenCV;
   - opção 7: janela 3D **gravando log** em `AHRS/logs/`;
   - opção 1: texto no console (fallback).
4. Mexer os sensores; fechar com Q/ESC.
5. Analisar o log:
   `.venv\Scripts\python.exe AHRS\tools\analisa_log_ahrs.py`.

## Voltar às luvas
`AHRS_menu.bat` -> opção 5, ou
`pio run -d firmware\esp32_duas_imus -t upload`.

Guia detalhado do firmware/protocolo: `AHRS/firmware/esp32_teste_ahrs/README.md`.

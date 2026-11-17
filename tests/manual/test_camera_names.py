"""Teste do modulo `camera_names` (nome de camera = verdade; indice = palpite).

Sem hardware: a enumeracao do Windows e' injetada no cache do modulo. O que este
teste protege e' a regra que faltava no dia da bancada: a webcam do LAPTOP nunca
pode ser tratada como camera auxiliar, e o nome de cada indice tem de ser
legivel/avisavel.

    .venv\\Scripts\\python.exe test_camera_names.py
"""
import camera_names as cn

# --- 1) classificacao por nome (o que decide "e' a do laptop?") -------------
assert cn.e_do_laptop("Integrated Camera"), "Integrated Camera e' do laptop"
assert cn.e_do_laptop("Integrated Webcam"), "Integrated Webcam e' do laptop"
assert cn.e_do_laptop("Webcam do laptop"), "texto em portugues tambem"
assert not cn.e_do_laptop("SIGMA-W780M"), "externa da bancada nao e' laptop"
assert not cn.e_do_laptop("Trust USB Camera"), "externa da bancada nao e' laptop"
assert not cn.e_do_laptop(None) and not cn.e_do_laptop(""), "vazio nao e' laptop"

# --- 2) mapa/nome/aviso seguindo a enumeracao injetada ----------------------
cn._CACHE = ["SIGMA-W780M", "Integrated Camera", "Trust USB Camera"]
assert cn.mapa_de_cameras() == {0: "SIGMA-W780M", 1: "Integrated Camera",
                                2: "Trust USB Camera"}
assert cn.nome_da_camera(1) == "Integrated Camera"
assert cn.nome_da_camera(9) is None, "indice fora da lista devolve None"
assert cn.nome_da_camera("1") == "Integrated Camera", "aceita string"
assert cn.indices_fora_do_laptop() == [0, 2]
aviso = cn.aviso_laptop(1)
assert aviso is not None and "LAPTOP" in aviso and "Integrated Camera" in aviso
assert cn.aviso_laptop(0) is None, "externa nao gera aviso"

# --- 3) resumo pronto para imprimir ----------------------------------------
linhas = cn.resumo([0, 1, 9])
assert linhas[0] == "idx 0: SIGMA-W780M", linhas[0]
assert "LAPTOP" in linhas[1], linhas[1]
assert "(nao existe)" in linhas[2], linhas[2]
completo = cn.resumo()
assert len(completo) == 3 and "LAPTOP" in completo[1]

# --- 4) sem enumeracao: degrada em silencio (nunca levanta) -----------------
cn._CACHE = []
assert cn.indices_fora_do_laptop() == []
assert cn.nome_da_camera(0) is None
assert cn.aviso_laptop(0) is None
assert "nenhum nome" in cn.resumo()[0]
cn._CACHE = None            # ultima coisa do processo: nao importa re-enumerar

print("TESTE CAMERA NAMES: OK (nome decide; sem enumeracao nada levanta)")

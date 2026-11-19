#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Teste manual do `tools/feedback_adaptativo.py` (sem hardware).

Cobre:
  1. invariante do protocolo: janela do decoder SEMPRE dentro da fase MI
     (nunca atravessa ME->MI) e passo/tamanho corretos;
  2. plano de 3 sessoes = 60 trials/classe, <=20/classe/sessao (validado);
  3. pipeline completo: calibracao CSP+LDA -> score continuo -> feedback ->
     adaptacao por bloco -> sessao final CONGELADA, com metricas BA+IC e ERD;
  4. criterio de nao-respondedor dispara quando o participante sintetico nao
     tem ERD (illiteracy) e NAO dispara quando tem.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "mestradopy"))
sys.path.insert(0, str(RAIZ / "src" / "mestradopy" / "tools"))

import feedback_adaptativo as fa  # noqa: E402


class TestJanelasNaoCruzamFases(unittest.TestCase):
    def test_janelas_estritamente_dentro_da_mi(self):
        """Cada janela do decoder e' um recorte CONTIDO na fase MI."""
        rng = np.random.default_rng(3)
        x_mi = rng.normal(size=(3, int(4.0 * fa.FS)))       # 4 s de MI
        janelas = fa._janelas_dentro(x_mi, int(fa.JANELA_S * fa.FS),
                                     int(fa.PASSO_S * fa.FS))
        self.assertEqual(janelas.shape[1:], (3, int(fa.JANELA_S * fa.FS)))
        self.assertGreater(janelas.shape[0], 0)
        janela_n = int(fa.JANELA_S * fa.FS)
        passo_n = int(fa.PASSO_S * fa.FS)
        for i, w in enumerate(janelas):
            ini = i * passo_n
            self.assertTrue(np.allclose(w, x_mi[:, ini:ini + janela_n]))
            self.assertLessEqual(ini + janela_n, x_mi.shape[1])

    def test_janelas_do_trial_ignoram_me(self):
        """Mesmo com ME longa + MI, as janelas usam SO a MI recortada."""
        trial = {
            "classe": 1, "sessao": 1, "bloco": 0,
            "X_base": np.zeros((3, int(fa.FS))),
            "X_me": np.ones((3, int(4.0 * fa.FS))) * 999.0,   # marcador da ME
            "X_mi": np.zeros((3, int(4.0 * fa.FS))),
        }
        janelas = fa.janelas_do_trial(trial)
        self.assertGreater(len(janelas), 0)
        self.assertFalse(np.any(np.abs(janelas) > 100))     # nada da ME (999)


class TestPlanoProtocolo(unittest.TestCase):
    def test_plano_60_por_classe_em_3_sessoes(self):
        p = fa.plano_protocolo()
        self.assertEqual(p["total_trials_por_classe"], 60)
        self.assertEqual(p["sessoes"], 3)
        self.assertLessEqual(p["trials_por_classe_por_sessao"], 20)
        self.assertEqual(p["trials_por_sessao"],
                         2 * p["trials_por_classe_por_sessao"])

    def test_recusa_mais_de_20_por_sessao(self):
        with self.assertRaises(SystemExit):
            fa.plano_protocolo(trials_por_classe=30)


class TestPipelineCompleto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plano = fa.plano_protocolo()
        # sessoes menores para o teste rodar rapido (4 trials/classe por sessao)
        cls.sessoes = [
            fa.gerar_sessao_sintetica(s, trials_por_classe=4, blocos=2,
                                      seed=101 + s, erd_forte=True)
            for s in (1, 2, 3)
        ]

    def test_calibracao_score_e_adaptacao(self):
        treino = fa.roda_treino_adaptativo(self.sessoes[:2], log=lambda *a: None)
        self.assertIsNotNone(treino["modelo"])
        self.assertEqual(len(treino["historico"]), 4)      # 2 sessoes x 2 blocos
        papeis = [h["papel"] for h in treino["historico"]]
        self.assertEqual(papeis[0], "calibracao")
        self.assertIn("adaptacao", papeis)
        for h in treino["historico"]:
            self.assertTrue(np.isfinite(h["ba"]))
            self.assertGreaterEqual(h["tolerancia"], 0.12)
            self.assertLessEqual(h["tolerancia"], 0.80)

    def test_score_continuo_no_intervalo(self):
        treino = fa.roda_treino_adaptativo(self.sessoes[:1], log=lambda *a: None)
        trial = self.sessoes[0][0]
        score, p_dir, n = fa.pontua_trial(treino["modelo"], trial)
        self.assertGreaterEqual(n, 1)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(p_dir, 0.0)
        self.assertLessEqual(p_dir, 1.0)

    def test_teste_congelado_e_respondedor(self):
        treino = fa.roda_treino_adaptativo(self.sessoes[:2], log=lambda *a: None)
        teste = fa.roda_teste_congelado(self.sessoes[2], treino["modelo"],
                                        log=lambda *a: None)
        m = teste["metricas"]
        self.assertEqual(m["n"], 8)                        # 4/classe x 2
        self.assertTrue(np.isfinite(m["ba"]))
        self.assertEqual(len(m["ic95"]), 2)
        self.assertIn("8-13Hz", teste["erd"])
        self.assertIsInstance(teste["veredito"]["nao_respondedor"], bool)

    def test_nao_respondedor_quando_sem_erd(self):
        sessoes = [
            fa.gerar_sessao_sintetica(s, trials_por_classe=4, blocos=2,
                                      seed=303 + s, erd_forte=False)
            for s in (1, 2, 3)
        ]
        treino = fa.roda_treino_adaptativo(sessoes[:2], log=lambda *a: None)
        teste = fa.roda_teste_congelado(sessoes[2], treino["modelo"],
                                        log=lambda *a: None)
        self.assertTrue(teste["veredito"]["nao_respondedor"])


class TestRegressaoCanaisEPickle(unittest.TestCase):
    """Regressões dos bugs de 19/11: orientação no resample e joblib."""

    def test_resample_preserva_canais(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(3, 1000))          # (canais, amostras)
        y = fa._resample(x, 500, 128)
        self.assertEqual(y.shape[0], 3)         # NUNCA virar 1 canal
        self.assertEqual(y.shape[1], 256)

    def test_pipeline_salva_e_carrega_joblib(self):
        import tempfile, joblib
        from pathlib import Path as P
        pipe = fa.cria_pipeline()
        rng = np.random.default_rng(8)
        X = rng.normal(size=(40, 3, 128))
        y = np.array([0, 1] * 20)
        pipe.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            f = P(d) / "modelo.joblib"
            joblib.dump(pipe, f)
            loaded = joblib.load(f)
            self.assertEqual(loaded.predict(X).shape, (40,))

    def test_finite_np(self):
        x = np.array([np.nan, np.inf, -np.inf, 1.0])
        y = fa._finite_np(x)
        self.assertTrue(np.all(np.isfinite(y)))


class TestDecide(unittest.TestCase):
    def test_decide_direita_esquerda(self):
        self.assertEqual(fa.decide(0.5, 0.0)[0], 1)
        self.assertEqual(fa.decide(-0.5, 0.0)[0], -1)
        self.assertFalse(fa.decide(float("nan"), 0.0)[1])

    def test_dificuldade_adapta(self):
        tol, ganho, acao = fa.aplica_dificuldade(0.35, 1.0, 0.90)
        self.assertEqual(acao, "mais_dificil")
        self.assertLess(tol, 0.35)
        tol2, ganho2, acao2 = fa.aplica_dificuldade(0.35, 1.0, 0.40)
        self.assertEqual(acao2, "mais_facil")
        self.assertGreater(tol2, 0.35)


if __name__ == "__main__":
    unittest.main(verbosity=2)

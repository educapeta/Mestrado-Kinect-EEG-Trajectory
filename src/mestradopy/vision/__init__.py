"""Pacote `vision/` do projeto: adaptadores de modelos de visao.

Conteudo:
  - `handmvnet_adapter`: adaptador do modelo multiview HandMvNet (ou da
    triangulacao classica como fallback) para o formato do projeto;
  - `handmvnet_synthetic`: geracao de frames/vistas sinteticas calibradas para
    smoke test sem hardware.

Os modulos sao importados de forma explicita (`from vision.handmvnet_adapter
import HandMvNetAdapter`) porque o resto do projeto usa imports `flat` com
`src/mestradopy` no `sys.path`; manter o `__init__` sem imports evita pagar o
custo de carregar opencv/numpy em quem so' precisa do pacote.
"""

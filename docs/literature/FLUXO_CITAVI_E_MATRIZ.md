# Citavi + repositório: fluxo de trabalho da literatura

## 1. O que dá e o que não dá para fazer

O projeto Citavi `Mestrado_BCI.ctv6` é um banco **SQLite** (confirmado pelo
cabeçalho `SQLite format 3`). Isso significa que:

- **dá para ler** títulos, autores, ano, DOI, categorias, keywords, notas e
  anotações com segurança, com o Citavi fechado;
- **não é seguro editar** o `.ctv6` diretamente: o Citavi mantém estado
  próprio, há sincronização OneDrive e um arquivo inconsistente pode corromper
  a biblioteca inteira;
- a Citavi não oferece uma API pública de escrita para projetos locais.

Conclusão: **Citavi continua sendo o gerenciador oficial**. O repositório é a
camada de trabalho compartilhada (índice, matriz, notas, BibTeX).

## 2. Fluxo recomendado

```text
Citavi (master)
  ├─ exporta BibTeX/RIS
  ├─ exporta PDFs/anotações quando necessário
  └─ recebe imports de novas referências

Repositório literatura/
  ├─ INDICE.md              # visão curada por tema
  ├─ referencias.bib        # metadados BibTeX
  ├─ matriz_literatura.csv  # matriz de leitura
  ├─ notas/<chave>.md       # 1 nota por artigo
  └─ FLUXO_CITAVI_E_MATRIZ.md
```

### Rotina semanal

1. No Citavi: **File → Export → BibTeX** (ou RIS) para o projeto.
2. Salvar em `literatura/referencias.bib` (substituir/atualizar).
3. Rodar o script de índice (a implementar):

   ```powershell
   .\.venv\Scripts\python.exe tools\atualiza_literatura.py --bib literatura/referencias.bib
   ```

4. Preencher a linha correspondente em `matriz_literatura.csv`.
5. Escrever uma nota curta em `literatura/notas/<chave>.md` só para os
   artigos que entram na fundamentação.
6. Referências novas sugeridas pelo agente: gerar um `.ris` para importar no
   Citavi, nunca escrever no `.ctv6`.

## 3. Categorias sugeridas no Citavi

- `1-EEG-decoding`
- `2-Datasets`
- `3-IMU-MediaPipe`
- `4-Hand-pose-Quaternion`
- `5-Robot-hand-control`
- `6-Methods-Transformers`
- `7-Review`
- `8-Para-ler`

Não renomeie categorias existentes sem exportar antes; a Citavi usa IDs
internos, e a organização visual pode ficar inconsistente.

## 4. Matriz de literatura

Cabeçalho de `matriz_literatura.csv`:

```csv
id,ano,autores,titulo,venue,doi,pdf,pergunta,metodo,dataset,entrada,saida,metricas,limitacoes,uso_no_projeto,status
```

Regras:

- `id`: `primeiroautor_ano` (ex.: `luciw_2014`);
- `pdf`: caminho relativo em `literatura/` quando o PDF estiver local;
- `pergunta`: o que o artigo investiga, em uma frase;
- `metodo`: arquitetura/estatística principal;
- `dataset`: nome, sujeitos, canais, tarefa;
- `entrada`/`saida`: EEG, IMU, vídeo, pose etc.;
- `metricas`: correlação, RMSE, acurácia, erro angular;
- `limitacoes`: o que o artigo **não** resolve;
- `uso_no_projeto`: `fundamentacao`, `metodo`, `baseline`, `dataset`,
  `descartado`;
- `status`: `lido`, `lido-parcial`, `triagem`, `a-ler`.

## 5. O que o agente pode fazer sem tocar no Citavi

- ler o `.ctv6` em modo somente leitura (com o Citavi fechado);
- gerar um CSV/Markdown de conferência;
- atualizar `INDICE.md`, `referencias.bib` e `matriz_literatura.csv`;
- escrever notas e sínteses;
- gerar `.ris`/`.bib` para você importar no Citavi.

## 6. O que o agente não deve fazer

- editar o `.ctv6` com o Citavi aberto;
- mover/renomear PDFs dentro de `Citavi Attachments/`;
- sobrescrever anotações do Citavi;
- commitar PDFs no Git (direitos autorais e dados privados).

Se quiser, o próximo passo é eu implementar `tools/atualiza_literatura.py`
(leitura do `.ctv6` + geração de `INDICE.md`/`matriz_literatura.csv`) com um
teste usando uma cópia do banco.

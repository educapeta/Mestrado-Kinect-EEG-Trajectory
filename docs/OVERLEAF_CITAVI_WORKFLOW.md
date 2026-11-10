# Overleaf + Citavi + repositório: fluxo de trabalho

## 1. Resumo

| Caminho | Requer | Como o agente participa |
| --- | --- | --- |
| **Overleaf Git integration** | plano premium/Overleaf Commons | clona o projeto, edita `.tex`/`.bib`, faz commit e push |
| **Overleaf GitHub sync** | premium + projeto criado a partir do GitHub | edita o repositório GitHub; Overleaf puxa/empurra pelo botão |
| **ZIP manual** | plano free | edita uma cópia local, você sobe o ZIP/arquivos de volta |
| **API do Overleaf** | não existe API pública de escrita | não usar soluções não oficiais |

O agente **não acessa a interface web do Overleaf** sozinho. Ele trabalha em uma
cópia Git local ou em arquivos que você baixa/envia.

## 2. Caminho recomendado: Git integration do Overleaf

No projeto Overleaf, abrir **Integrations → Git** e copiar a URL. No Overleaf
Cloud ela tem o formato:

```text
https://git.overleaf.com/<PROJECT_ID>
```

Clonar uma vez (o token de autenticação é do Overleaf, não a senha da conta):

```powershell
cd C:\mestradopy
git clone https://git.overleaf.com/<PROJECT_ID> tese_overleaf
cd tese_overleaf
```

Fluxo diário:

```powershell
# 1. sempre puxar antes de editar (o coautor pode ter mexido no Overleaf)
git pull

# 2. editar .tex e referencias.bib localmente

# 3. conferir
latexmk -pdf main.tex

# 4. enviar para o Overleaf
git add -A
git commit -m "capitulo 2: revisao e citacoes"
git push
```

Depois do push, o projeto Overleaf recebe o commit; recarregue a página.

### Limitações que importam para tese

- Git integration do Overleaf é **premium** (ou Overleaf Commons da universidade);
- **não suporta branches**;
- não suporta Git LFS nem symlinks;
- push/pull pode deslocar **track changes/comentários** do Overleaf — não misture
  edição pesada via Git com revisão por “Track Changes”;
- renomear/mover arquivo no Git pode apagar metadados de comentários;
- faça commits pequenos e sempre `pull` antes.

### Token de autenticação

O Overleaf fornece um token para o Git; use o gerenciador de credenciais do
Windows/Git Credential Manager. **Nunca** salve token em `.tex`, `.bib` ou no
repositório.

## 3. Alternativa: GitHub synchronization

O Overleaf também tem **GitHub sync**, mas com uma limitação decisiva: **não é
possível ligar um projeto Overleaf existente a um repositório GitHub
existente**. Só dá para:

- criar um projeto Overleaf novo a partir de um repositório GitHub; ou
- criar um repositório GitHub novo a partir de um projeto Overleaf.

A sincronização é manual (botão **GitHub** no menu do projeto): push/pull não
acontece sozinho. Também é premium e tem limite de ~7 MB de conteúdo editável
para o sync.

Se a tese já existe no Overleaf, o caminho natural é **Git integration**, não
GitHub sync.

## 4. Caminho free: ZIP/arquivos

Se não houver plano premium:

1. no Overleaf: **Menu → Download → Source**;
2. descompactar em `C:\mestradopy\tese_overleaf`;
3. editar/compilar localmente com `latexmk`;
4. subir de volta os arquivos alterados (ou o ZIP) pelo menu do Overleaf.

Funciona, mas não é colaborativo em tempo real. O agente pode editar a cópia
local e gerar um diff/patch para você aplicar.

## 5. BibTeX/Citavi: uma única fonte de verdade

O Citavi continua sendo o gerenciador oficial. O fluxo:

```text
Citavi (Mestrado_BCI.ctv6)
  └─ File → Export → BibTeX
        ↓
tese_overleaf/referencias.bib      ← fonte que o LaTeX usa
        ↓
\cite{chave} no .tex
```

Regras:

- exportar **BibTeX** (não RIS) para o LaTeX;
- manter um único `referencias.bib` no projeto Overleaf;
- não editar as chaves à mão se o Citavi for reexportar; escolha chaves estáveis
  no Citavi (`autor_ano_palavra`);
- se usar `biblatex`, no preâmbulo:

  ```latex
  \usepackage[backend=biber,style=apa]{biblatex}
  \addbibresource{referencias.bib}
  ```

  e no fim:

  ```latex
  \printbibliography
  ```

- se usar BibTeX clássico: `\bibliography{referencias}` + `\bibliographystyle{...}`;
- o `literatura/matriz_literatura.csv` é para leitura/organização, **não** entra
  no LaTeX;
- `literatura/INDICE.md` e notas não precisam ir para o Overleaf (mas podem,
  como anexo/apêndice, se você quiser).

### Checagem de citações

Antes de submeter, rodar:

```powershell
latexmk -pdf main.tex
# ou, no Overleaf, olhar o log de citações indefinidas
```

Um script simples para comparar `\cite{...}` do `.tex` com as chaves do `.bib`
pode ser implementado em `tools/checa_citacoes.py`; ele evita citação órfã e
entrada não usada.

## 6. Como trabalhar com o agente

- **Com Git integration:** o agente clona, edita, compila e faz push no clone
  local. Você revisa no Overleaf e puxa quando quiser.
- **Sem Git:** você baixa o ZIP, coloca em `tese_overleaf/`; o agente edita e
  você sobe de volta.
- **Citações:** o agente pode atualizar `referencias.bib`, escrever texto com
  `\cite{...}` e checar chaves, desde que o `.bib` esteja no projeto.
- **Citavi:** o agente **não edita** o `.ctv6`; ele lê/exporta ou gera `.ris`
  para você importar.

## 7. Decisão prática

Pergunte a si mesmo:

1. O projeto Overleaf já existe? → **Git integration** (se premium).
2. A universidade dá Overleaf Commons? → provavelmente sim; teste
   **Integrations → Git**.
3. Sem premium? → ZIP + compilação local; sem promessa de sincronização ao vivo.
4. Quer manter Citavi como master? → exporte BibTeX e mantenha só o
   `referencias.bib` no Overleaf.

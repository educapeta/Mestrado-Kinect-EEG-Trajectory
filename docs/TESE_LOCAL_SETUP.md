# Tese local grátis: LaTeX + VS Code + Git (sem Overleaf premium)

## 1. Recomendação

Para escrever a tese com citações e com o agente trabalhando junto, o caminho
mais simples e gratuito é:

```text
MiKTeX/TeX Live  +  VS Code + LaTeX Workshop  +  Git/GitHub privado
```

O Overleaf deixa de ser o editor principal; se quiser, ele continua como
visualizador/backup via ZIP, ou você usa uma compilação na nuvem pelo GitHub
Actions.

### O que já existe na máquina

- Git: `C:\Program Files\Git\cmd\git.exe` (instalado);
- VS Code: instalado;
- LaTeX: **não instalado** (nem `pdflatex`, `latexmk`, `biber`).

## 2. Instalar o LaTeX (free)

Duas opções:

### MiKTeX (mais fácil no Windows)

- Download: https://miktex.org/download
- Instalação padrão; ele baixa pacotes sob demanda.
- Na primeira compilação, aceite instalar pacotes automaticamente.

### TeX Live (mais completo, download maior)

- Download: https://tug.org/texlive/
- Bom se você não quiser dependência de rede no futuro.

Qualquer um dos dois instala `latexmk`, `pdflatex` e `biber`.

### Extensão do VS Code

Instalar **LaTeX Workshop**:

```powershell
code --install-extension James-Yu.latex-workshop
```

Opcional: **LaTeX Utilities** para contadores de palavras e limpeza.

## 3. Estrutura sugerida da tese

```text
C:\mestradopy\tese\
  main.tex
  capitulos/
    introducao.tex
    fundamentacao.tex
    metodologia.tex
    resultados.tex
    conclusao.tex
  referencias.bib
  figuras/
  anexos/
  .gitignore
  .vscode/settings.json
```

`main.tex` mínimo com `biblatex`:

```latex
\documentclass[12pt,a4paper]{report}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazil]{babel}
\usepackage{graphicx}
\usepackage[backend=biber,style=apa]{biblatex}
\addbibresource{referencias.bib}

\begin{document}
\tableofcontents
\include{capitulos/introducao}
\printbibliography
\end{document}
```

Se o template da universidade usar BibTeX clássico, mantenha
`\bibliography{referencias}` + `\bibliographystyle{...}`.

## 4. Compilar localmente

```powershell
cd C:\mestradopy\tese
latexmk -pdf -interaction=nonstopmode -synctex=1 main.tex
```

O `latexmk` roda biber/BibTeX automaticamente conforme o template.

No VS Code: `Ctrl+Alt+B` (LaTeX Workshop) e `Ctrl+Alt+V` para visualizar o PDF.

## 5. Git/GitHub privado (grátis)

```powershell
cd C:\mestradopy\tese
git init
git add -A
git commit -m "estrutura inicial da tese"
```

Criar um repositório **privado** no GitHub e:

```powershell
git remote add origin https://github.com/<usuario>/<repo>.git
git branch -M main
git push -u origin main
```

Colaboração: você e o orientador usam o mesmo repositório; o agente edita a
cópia local e faz commit/push.

`.gitignore` sugerido para a tese:

```gitignore
# builds
*.aux
*.bbl
*.bcf
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.run.xml
*.synctex.gz
*.toc
main.pdf

# dados/figuras grandes e privadas
dados/
*.zip
*.rar
```

Não versionar PDFs gigantes de literatura nem dados de participantes.

## 6. Compilação grátis na nuvem (GitHub Actions)

Se quiser um “Overleaf de compilação” sem instalar LaTeX em todo computador,
adicione `.github/workflows/latex.yml`:

```yaml
name: build-thesis
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: xu-cheng/latex-action@v3
        with:
          root_file: main.tex
      - uses: actions/upload-artifact@v4
        with:
          name: tese-pdf
          path: main.pdf
```

Cada push gera um PDF para baixar nos *artifacts* (minutos grátis no GitHub;
repositório público é ilimitado).

## 7. Citações: Citavi → BibTeX → LaTeX

1. No Citavi: **File → Export → BibTeX**.
2. Salvar/atualizar `tese/referencias.bib`.
3. Citar no texto: `\cite{chave}` ou `\textcite{chave}` (biblatex).
4. Compilar; o log mostra chaves indefinidas.
5. O `literatura/matriz_literatura.csv` não entra no LaTeX; é organização.

Se o Citavi gerar chaves instáveis, fixe-as no Citavi antes de exportar.

## 8. Alternativas sem instalar nada

| Opção | Limite |
| --- | --- |
| Overleaf free | sem Git; ZIP manual; colaboração limitada |
| GitHub + GitHub Actions | edita `.tex` no navegador/VS Code; PDF no artifact |
| CoCalc (free tier) | editor LaTeX online, recursos limitados |
| Overleaf Community Edition | grátis, mas exige Docker/servidor; para usuário avançado |

Para o nosso fluxo com agente, **local + Git** é o que funciona melhor: o
arquivo está no disco, eu edito, compilo e valido as citações.

## 9. Próximo passo

1. instalar MiKTeX ou TeX Live;
2. instalar LaTeX Workshop;
3. criar `C:\mestradopy\tese`;
4. colocar o template da universidade e o `referencias.bib`;
5. compilar uma vez;
6. criar o repositório GitHub privado.

Se você baixar o ZIP do Overleaf atual, eu monto a estrutura local, adapto o
template e faço a primeira compilação.

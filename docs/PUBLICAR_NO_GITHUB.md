# Publicar no GitHub sem que ninguém edite (guia para quem nunca usou)

## 1. A verdade sobre "impedir edição"

**Ninguém pode editar o seu repositório sem você convidar** — isso vale até em
repositório **público**. O que as outras pessoas podem fazer em um repo público:

| Ação | Público | Privado |
|---|---|---|
| Ler o código pelo navegador | sim | não (só convidados) |
| **Editar / enviar commits para o seu repo** | **não** (sempre) | **não** (só convidados com papel Write) |
| Fazer *fork* (cópia na conta dela) | sim | não |
| Abrir *issue* / *pull request* (sugestão) | sim | só convidados |
| Aceitar a sugestão | **só você** | **só você** |

Ou seja: **público = leitura livre, escrita só sua**. Se você quer que ninguém
nem *leia*, use **privado**. Um *fork* não pode ser impedido em repo público, mas
a cópia fica na conta da outra pessoa — o seu repositório continua seu.

Três controles extras:

- **Arquivar** (Settings → *Archive this repository*): deixa o repo **somente
  leitura para todo mundo, inclusive para você**. É o "congelado" definitivo.
- **Colaboradores** (Settings → Collaborators): convide o orientador com papel
  **Read** — ele lê e clona, mas não edita.
- **Licença** (arquivo `LICENSE`): define o que os outros podem *fazer* com o
  código (usar, modificar, citar). **Sem licença** = todos os direitos
  reservados (ninguém tem permissão de reutilizar). Se quiser permitir reuso com
  citação, use **MIT** ou **Apache-2.0**.

## 2. Passo a passo (caminho recomendado: site + 3 comandos)

**Passo 1 — conta:** se ainda não tem, crie em <https://github.com>.

**Passo 2 — criar o repositório VAZIO no site:**
1. Clique no **+** (canto superior direito) → **New repository**.
2. Nome sugerido: `mestrado-groundtruth-eeg`
3. Descrição: `Ground truth 3D da mão (Kinect + MediaPipe + IMU) acoplado a EEG g.Nautilus`
4. **Público** ou **Privado** (você escolhe; veja a tabela acima).
5. **NÃO** marque "Add a README file", "Add .gitignore" nem "Choose a license"
   (o projeto já tem README, .gitignore e já tem commits).
6. **Create repository**. Copie a URL que aparece, do tipo
   `https://github.com/SEU-USUARIO/mestrado-groundtruth-eeg.git`

**Passo 3 — enviar o código (no terminal do VS Code, dentro de `C:\Mestradopy`):**

```powershell
cd C:\Mestradopy
git remote add origin https://github.com/SEU-USUARIO/mestrado-groundtruth-eeg.git
git branch -M main
git push -u origin main
```

Na primeira vez, o Windows abre uma **janela para você entrar no GitHub pelo
navegador** (Git Credential Manager — já está configurado na sua máquina). Não
precisa criar token nem digitar senha no terminal. Depois disso, os envios são
automáticos.

**Passo 4 — conferir:** abra a URL do repositório no navegador. Devem aparecer
os 39 arquivos (não deve haver `.venv`, `gravacoes/`, `data/`).

**Passo 5 — atualizar depois** (sempre que mexer no código):

```powershell
git add -A
git commit -m "descricao curta da mudanca"
git push
```

> Alternativa sem terminal: instalar o **GitHub Desktop**
> (<https://desktop.github.com>), `File → Add local repository` → `C:\Mestradopy`
> → **Publish repository**. Para atualizar: escrever a mensagem, **Commit to
> main** e **Push origin**.

## 3. O que vai e o que NÃO vai

Vai (39 arquivos, **13,5 MB**): os 20 programas/testes, `README.md`,
`HISTORICO_ITERACOES.txt`, `imu.py`, `docs/`, `tools/`, as calibrações `.npz` e
os 2 modelos `.task` (7,5 + 5,5 MB; garantem que o clone funcione offline).

**Não vai** (garantido pelo `.gitignore`): `.venv/` (36 mil arquivos!),
`gravacoes/` (dados das sessões), `data/`, `results/`, `legacy/`, `*_priming/`,
`*.csv`, `*.avi`, `*.pt`.

Verificado antes de publicar: **nenhuma chave de API** e **nenhum dado de
participante** está versionado.

## 4. Atualizando depois (um comando: `tools/publicar.ps1`)

```powershell
cd C:\Mestradopy
powershell -ExecutionPolicy Bypass -File tools\publicar.ps1 "descreva a mudanca em uma linha"
```

> O `-ExecutionPolicy Bypass` é necessário porque o Windows está com a política
> **Restricted** (bloqueia scripts). Se preferir, rode **uma vez**
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e depois use só
> `.\tools\publicar.ps1 "mensagem"`.

O script faz, em ordem:
1. mostra o que mudou desde o último commit;
2. **bloqueia** se algum arquivo novo passar de 50 MB (protege contra subir dado
   de sessão por engano);
3. `git add -A` + `git commit -m "mensagem"`;
4. `git pull --rebase` (integra o que você editou **pelo site do GitHub**) e
   `git push`.

Se houver conflito, ele **cancela tudo** e avisa — nada é perdido: o commit fica
salvo localmente.

### "Editei algo pelo site do GitHub e agora o push é recusado"
Isso é normal: o repositório remoto ficou "na frente" do seu computador
(`rejected ... fetch first`). A correção é integrar antes de enviar:

```powershell
git pull --rebase origin main
git push
```

O script acima já faz isso sozinho.

### Pelo VS Code (sem terminal)
Aba **Source Control** (Ctrl+Shift+G) → escreva a mensagem → **Commit** →
**Sync Changes**.

## 5. Cuidados que valem para sempre

1. **Nunca** commite chave de API/token (DeepSeek, Cline, etc.). Se colar uma em
   um arquivo e commitar, ela fica no histórico — teria que reescrever o
   histórico e trocar a chave. (Hoje o repositório está limpo.)
2. **Nunca** publique dados de participantes (LGPD/CEP): os arquivos
   `*_participante.json` e os CSVs de sessão ficam fora do git, no disco de
   dados.
3. **Backup em 3 lugares**: `C:\Mestradopy` (trabalho) + GitHub (código) +
   `D:\Mestrado_Dados` (dados). O GitHub **não** é backup dos dados.

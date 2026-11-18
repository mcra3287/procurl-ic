# ProCuRL-Target — reprodução para a IC

Baseado na implementação oficial de **Proximal Curriculum with Task Correlations for Deep Reinforcement Learning**, Tzannetos, Kamalaruban e Singla, IJCAI 2024. Esta etapa prepara a reprodução em Point Mass Sparse.

- [Artigo](https://www.ijcai.org/proceedings/2024/556)
- [Repositório dos autores](https://github.com/machine-teaching-group/ijcai2024-proximal-curriculum-target-rl)
- Origem: `9c7df278fb2520188d4dafb10f5888acec67bd4d`, preservado na branch `main` do fork e na tag local `upstream-ijcai2024` do pacote.
- [Mudanças e protocolo](docs/REPRODUCTION.md)
- [Validação executada](reports/SMOKE_STATUS.md)
- [Protocolo completo de PointMass e suíte de 25 sementes](docs/FULL_POINTMASS.md)
- [README original](docs/README-upstream.md)

## Instalação

Clone a branch de reprodução:

```bash
git clone --branch reproduction-pointmass https://github.com/mcra3287/procurl-ic.git
cd procurl-ic
```

Use Python **3.10**, como os autores. Com [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.10
uv venv --python 3.10
```

No Linux, instale primeiro PyTorch CPU para evitar dependências CUDA:

```bash
uv pip install --python .venv/bin/python 'torch==1.13.1+cpu' \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
```

Depois, tanto no Linux quanto no macOS:

```bash
uv pip install --python .venv/bin/python -r requirements-pointmass.txt
```

Verificado em Linux/CPU. A execução no macOS ainda precisa ser verificada. O arquivo `requirements-pointmass-lock-linux.txt` registra as versões instaladas nesta sessão e não é um lock portátil entre sistemas. O requirements original foi preservado.

## Reprodução com orçamento completo

Para executar o ProCuRL-Target nas duas configurações de PointMass, com **25 sementes por configuração** e **dois milhões de passos por execução**:

```bash
.venv/bin/python run_pointmass_suite.py run --workers 2
```

São 50 treinos. Para começar com um treino completo em cada configuração, use `--seeds 0`. O executor salva logs, checkpoints, curvas e tabelas com média e erro padrão entre sementes. Veja o [protocolo completo](docs/FULL_POINTMASS.md). O padrão de `run_pointmass.py` também é o perfil `paper`. A conclusão das execuções e a comparação com as curvas publicadas precisam ser verificadas; o teste curto anterior não comprova reprodução de desempenho.

## Teste curto opcional

A partir da raiz do projeto:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_pointmass.py train --profile smoke \
  --curriculum proxcorl --target-type single-task --seed 420 \
  --output results/smoke-proxcorl-420
```

O perfil smoke executa **10.240 passos**, duas atualizações PPO, mantendo **20 mil tarefas de treino** e **400 amostras-alvo**. Valida execução; não reproduz as curvas do artigo. A pasta de saída deve ser nova para evitar sobrescrita de resultados.

No código original, `proxcorl` corresponde ao **ProCuRL-Target** e `procurl-val` à variante sem os termos relacionados ao alvo.

## Tentativa de reprodução completa

```bash
.venv/bin/python run_pointmass.py train --profile paper \
  --curriculum proxcorl --target-type single-task --seed 0 \
  --output results/paper-pms1t-proxcorl-seed0 --tensorboard
```

São dois milhões de passos, com os parâmetros de `configs/pointmass-paper.json`. Para PM-S:2G, use `--target-type double-mode-gaussian`. O artigo utiliza 25 seeds em cada configuração de PointMass: uma única execução não reproduz sua média. O perfil completo ainda não foi executado nesta etapa.

A mesma entrada aceita `iid`, `target`, `procurl-val`, `spdl` e `plr`. Esses concorrentes ainda precisam ser validados individualmente. `currot` exige as dependências adicionais do projeto original.

## Recarregar e avaliar

```bash
.venv/bin/python run_pointmass.py evaluate \
  --checkpoint results/smoke-proxcorl-420/final_model.zip \
  --target-type single-task --seed 100420 --episodes-per-task 10
```

Cada execução salva configuração efetiva, versões, hashes dos datasets, métricas, avaliações periódicas, monitores, melhor checkpoint e modelo final. Os resultados ficam fora do Git.

## Origem e publicação

O código herdado permanece atribuído aos autores. Não atribuímos uma licença nova ao código deles: o repositório consultado não apresenta licença explícita. O histórico original e `main.py` foram preservados.

O próximo passo científico é reproduzir o baseline com orçamento completo. Depois, a hipótese da IC deve entrar em alterações separadas e ser comparada sob o mesmo protocolo.

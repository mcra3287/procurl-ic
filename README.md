# ProCuRL-Target no PointMass · reprodução da IC

Reimplementação do **professor que escolhe tarefas** no PointMass do artigo de Tzannetos, Kamalaruban e Singla (IJCAI 2024). O professor da IC e o dos autores podem treinar o mesmo agente PPO, sobre a mesma física e os mesmos conjuntos de tarefas. Na versão da IC, o primeiro arquivo a ler é [pointmass_ic.py](pointmass_ic.py).

- [Resultados e comparação com o artigo](reports/COMPARACAO.md)
- [Diferenças em relação ao original e motivo de cada mudança](docs/DIFERENCAS.md)
- [Curvas da Figura 2 e método de extração](reference/README.md)
- [Artigo](https://ijcai.org/proceedings/2024/0556.pdf) · [Versão com apêndices](https://arxiv.org/abs/2405.02481) · [Código dos autores](https://github.com/machine-teaching-group/ijcai2024-proximal-curriculum-target-rl)

## Resultado medido

Na última janela da curva, aos dois milhões de passos, a IC teve **69,8% de sucesso em PM-S:1T** (cinco sementes) e **64,1% em PM-S:2G** (três). Na Figura 2, as médias dos autores foram aproximadamente **88,8%** e **95,8%**, respectivamente, com 25 sementes por cenário. Uma semente em cada cenário terminou com zero sucesso. Na comparação controlada em PM-S:1T, semente 0, os dois professores produziram retornos idênticos nas 80 avaliações periódicas. Os [gráficos, resultados individuais e checkpoints](reports/COMPARACAO.md) permitem examinar a diferença.

## Instalação

No terminal do Ubuntu ou do macOS, com Git e [uv](https://docs.astral.sh/uv/) disponíveis:

```bash
git clone --branch ic/pointmass-curriculo https://github.com/mcra3287/procurl-ic.git
cd procurl-ic
uv python install 3.10
uv venv --python 3.10
```

No Linux, instale primeiro o PyTorch para CPU:

```bash
uv pip install --python .venv/bin/python 'torch==1.13.1+cpu' --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
```

Em ambos os sistemas, instale as dependências e execute os testes:

```bash
uv pip install --python .venv/bin/python -r requirements-pointmass.txt
.venv/bin/python -m unittest discover -s tests -v
```

Validado em Linux/CPU. O comando para macOS usa o pacote de PyTorch da plataforma e ainda precisa de validação em um Mac.

## Treino e comparação

Uma semente, orçamento completo (2 milhões de passos), alvo único:

```bash
.venv/bin/python run_pointmass_suite.py run --targets single-task --curricula proxcorl-ic --seeds 0
```

A suíte também agrega e compara os resultados quando o treino termina. Para comparar professor da IC e professor dos autores na mesma configuração, inclua ambos em `--curricula proxcorl-ic proxcorl`. Para os dois painéis do artigo com 25 sementes **por painel**, execute:

```bash
.venv/bin/python run_pointmass_suite.py run --curricula proxcorl-ic --workers 2
```

O último comando agenda 50 treinos independentes com 100 milhões de interações de treino solicitadas mais as avaliações. Ele pula as execuções completas já existentes na pasta de saída. Uma saída parcial é preservada e sinalizada. O número de workers afeta a concorrência e o uso de RAM.

Para regerar a comparação dos treinos locais completos:

```bash
.venv/bin/python run_pointmass_suite.py aggregate --curricula proxcorl-ic
```

Os resultados locais ficam em `results/pointmass-full`. Abra o PNG da configuração para ver a curva da IC sobreposta à publicada. No CSV terminado em `-vs-paper.csv`, a coluna `ours_minus_paper` mede a diferença em retorno no mesmo passo. Um valor de −0,10 corresponde a dez pontos percentuais a menos de sucesso neste ambiente de recompensa binária.

Para conferir a instalação em dois rollouts antes de iniciar uma bateria longa:

```bash
.venv/bin/python run_pointmass.py train --profile smoke --curriculum proxcorl-ic --output results/smoke-ic
```

O ambiente, a política e o código original mantêm a atribuição aos autores. A branch `main` do fork guarda o original e `reproduction-pointmass` guarda a adaptação anterior. A hipótese de pesquisa com novos sinais para o currículo pode ser medida posteriormente usando esta reprodução como controle.

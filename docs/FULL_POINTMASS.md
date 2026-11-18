# Reprodução do ProCuRL-Target no PointMass

Este protocolo executa a implementação oficial com orçamento completo. A preparação do código não significa que os resultados publicados já foram reproduzidos. A contribuição nova da IC continua fora desta etapa.

## Escopo

| Item | Protocolo |
|---|---|
| Método principal | ProCuRL-Target, `proxcorl` no código dos autores |
| Configurações | PM-s:1T (`single-task`) e PM-s:2G (`double-mode-gaussian`) |
| Sementes | 25 por configuração; escolhemos 0 a 24 |
| Orçamento | 2.000.000 interações de treino solicitadas por execução |
| Total do método principal | 50 treinos, 100 milhões de interações solicitadas |
| PPO | rollout 5.120; batch 128; dez épocas; entropia 0,01; beta do currículo 130 |
| Rede | uma camada compartilhada 64, um ramo da política 64 e um ramo do crítico 64 |
| Dados | CSVs oficiais, 20.000 tarefas de treino e 400 amostras-alvo |
| Avaliação | estocástica, a cada 25.000 passos; 100 episódios em 1T e dez por tarefa do CSV em 2G |
| Curvas | média e ± um erro padrão entre sementes de treinamento; suavização de dois pontos |

O SB3 completa rollouts inteiros: 2.000.000 solicitados resultam em 2.001.920 passos efetivos. Isso também ocorre com o orçamento do script original. Os passos de avaliação são adicionais e não atualizam o PPO. Os valores exatos de sementes dos autores não foram encontrados no repositório consultado; nossa lista 0–24 é registrada, sem alegar igualdade bit a bit com as execuções deles.

Fonte do protocolo: [artigo, Figura 2 e Apêndice D](https://arxiv.org/html/2405.02481v1), mais o `main.py` oficial para orçamento, frequência e número de episódios de avaliação. O apêndice fundamenta arquitetura, batch, entropia e beta. As figuras do artigo precisam ser comparadas com as curvas produzidas; executar 25 sementes, por si só, não comprova concordância.

## Executar

Use o ambiente Python instalado conforme o README, a partir da raiz do repositório. O perfil padrão de `run_pointmass.py` agora é `paper`; o teste curto exige `--profile smoke` explícito.

Primeiro treino completo em cada configuração:

```bash
.venv/bin/python run_pointmass_suite.py run --seeds 0 --workers 2
```

Para executar as 25 sementes em ambas as configurações:

```bash
.venv/bin/python run_pointmass_suite.py run --workers 2
```

Para conferir o plano sem treinar:

```bash
.venv/bin/python run_pointmass_suite.py plan
```

Para gerar tabelas e figuras a partir dos treinos já concluídos:

```bash
.venv/bin/python run_pointmass_suite.py aggregate
```

O número de workers controla quantos treinos independentes executam simultaneamente, sem vetorizar ambientes nem mudar o PPO. Reduza para um worker se faltar memória. Não se presume que os processos continuem após encerrar uma máquina ou sessão.

Resultados ficam em `results/pointmass-full/<configuração>/<método>/seed-<n>/`. Cada execução tem configuração, versões, hashes dos dados, monitores, avaliações periódicas, melhor checkpoint e modelo final. O log do processo fica ao lado da pasta da semente. O agregador produz PNG, PDF, CSVs sem suavização e com suavização de dois pontos, além de `summary.json` com contagem explícita de sementes concluídas.

Uma pasta incompleta nunca é sobrescrita automaticamente. Mova-a para outro nome antes de repetir uma execução interrompida. A retomada da suíte pula apenas treinos completos e compatíveis; não retoma um PPO interrompido a partir do meio. O agregador rejeita misturar commits, versões ou datasets diferentes no mesmo grupo. Com uma única semente, nenhum erro padrão entre sementes é apresentado.

## Controles e limites da primeira implementação

A reprodução completa **do método ProCuRL-Target** é o conjunto de 50 treinos acima. Reproduzir **todos os experimentos comparativos de PointMass do artigo** também exige os concorrentes; são escopos distintos. Não chamamos as 50 execuções de reprodução de todos os baselines.

O executor aceita os controles disponíveis sem solvers adicionais:

```bash
.venv/bin/python run_pointmass_suite.py run --workers 2 \
  --curricula proxcorl procurl-val iid target
```

Esse comando agenda 200 treinos. CURROT, SPDL, PLR e Gradient não integram esta suíte validada. CURROT e Gradient requerem uma preparação adicional; nenhum resultado desses métodos é inventado ou tratado como reproduzido. Para testar nossa futura contribuição, ProCuRL-Target será o baseline principal, no mesmo ambiente e com o mesmo orçamento.

## O que veio dos autores e o que mudou

- **Herdado:** dinâmica e recompensa do PointMass; CSVs; kernel; regra de seleção; callbacks do professor; rede personalizada; implementação do PPO do SB3.
- **Acrescentado:** entradas dedicadas de treino e suíte; registros de execução; relatórios e gráficos; testes; dependências específicas para PointMass.
- **Ajustado:** assinatura e tipos exigidos pelo Gymnasium; semeadura explícita; importação opcional dos solvers de CURROT; parâmetros PPO explícitos conforme o experimento do artigo.
- **Não alterado:** regra científica do currículo, recompensa, física, horizonte de cem passos, arquitetura e dados.

Os ajustes de tipos, semeadura e parâmetros podem mudar a trajetória numérica em relação a simplesmente executar o `main.py` original com seus padrões. Em particular, o ramo original sem W&B omitia parâmetros definidos no ramo com W&B: reutilizar esse caminho sem corrigir batch e entropia não corresponde ao protocolo descrito no artigo. A referência preservada em `main` permite inspecionar todas as diferenças. Trata-se de **reprodução baseada no código dos autores**, e não de uma reimplementação independente nem da contribuição original da IC.

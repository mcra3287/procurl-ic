# O que mudou e por quê

A base é o commit `9c7df278fb2520188d4dafb10f5888acec67bd4d` dos autores. O fork já tinha a adaptação de PointMass em `70e8aa383d1b4b36c64e050ca1ba8830aca563a1`. A branch `ic/pointmass-curriculo` acrescenta um professor escrito para a IC. O diff completo, inclusive cada linha de documentação e testes, pode ser aberto no [GitHub](https://github.com/mcra3287/procurl-ic/compare/main...ic/pointmass-curriculo).

## Que regra estamos reproduzindo?

Em cada tarefa de treino de contexto $c$, lemos o valor do crítico no estado inicial e limitamos essa previsão ao intervalo $[0,1]$. Chamamos o resultado de $p_c$. Um valor intermediário dá mais peso à tarefa pelo termo $p_c(1-p_c)$. Fazemos a mesma leitura para o contexto-alvo $t$. O peso relativo da tarefa de treino no código público é

$$
w(c)=\exp\!\left(130\, p_c(1-p_c)\max_{t}\left[p_t(1-p_t)e^{-\lVert c-t\rVert_2}\right]\right).
$$

O professor sorteia $c$ com probabilidade $w(c)/\sum_j w(j)$. Aqui entram tanto a dificuldade atual quanto a proximidade de um alvo em que o agente tem espaço para aprender. Mantemos o máximo porque o objetivo é comparar com a implementação que acompanha o artigo. O kernel usa as três coordenadas originais, sem padronização.

Há uma distinção verificável entre texto e código. A equação (4) do artigo escreve uma distribuição conjunta sobre pares $(t,c)$. Somar as probabilidades de todos os pares que contêm $c$ produz a distribuição marginal de $c$. Já a função publicada `pick_curr_id_proxcorl` toma o máximo sobre os alvos antes da exponencial e sorteia apenas $c$. As duas construções podem dar probabilidades diferentes em PM-S:2G. Em PM-S:1T, repetir o mesmo alvo dá a mesma distribuição de tarefas de treino nas duas construções. O repositório público não inclui os logs originais por semente que permitiriam confirmar qual executável gerou cada curva.

Fontes: [artigo, seção 3.2 e Figura 2](https://ijcai.org/proceedings/2024/0556.pdf) · [seleção oficial](https://github.com/machine-teaching-group/ijcai2024-proximal-curriculum-target-rl/blob/9c7df278fb2520188d4dafb10f5888acec67bd4d/abstract_classes/AbstractGymWrapper.py#L238) · [cálculo do termo-alvo](https://github.com/machine-teaching-group/ijcai2024-proximal-curriculum-target-rl/blob/9c7df278fb2520188d4dafb10f5888acec67bd4d/abstract_classes/AbstractTeacher.py).

## Mudanças introduzidas nesta versão

| Parte | Original | Nossa versão | Motivo e efeito esperado |
|---|---|---|---|
| Professor | Seleção e estimativa dos valores distribuídas entre o wrapper e o callback genéricos | `pointmass_ic.py` implementa previsão, pesos, sorteio e atualização em `ICPointMass` e `ICTeacher` | Validar o código que escrevemos contra o dos autores usando o mesmo agente. O executor identifica os dois como `proxcorl-ic` e `proxcorl`. |
| Consultas ao crítico | Uma inferência por tarefa de treino e duas pelo mesmo valor de cada alvo na expressão | Inferência em lote para os estados iniciais, uma leitura por valor, sem construir grafo de derivadas | Reduzir chamadas à rede. Operações em lote e arredondamentos podem alterar trajetórias de treino. |
| Distâncias | Uma linha por alvo com distâncias a todas as tarefas | Distância euclidiana via SciPy em blocos de até 32 alvos, mantendo o maior termo por tarefa | Evitar reter ou percorrer a matriz de 400 × 20.000 a cada escolha. O construtor herdado ainda aloca a matriz inicial de zeros, que a nossa classe libera. |
| Alvos repetidos | Computa os 400 termos mesmo quando o alvo é idêntico | Gera as 400 amostras com o amostrador original e avalia uma única vez cada contexto distinto | Em PM-S:1T os 400 alvos são idênticos e repetir um termo não altera o máximo. O sorteio original em PM-S:2G continua sendo feito a cada atualização. |
| Sorteio por episódio | Recalcula produtos, máximos e exponenciais a cada reset e invoca `random.choices` com pesos | Calcula pesos cumulativos ao atualizar o professor e reutiliza-os em `random.choices` | Durante o rollout, o professor mantém os mesmos pesos. A primeira escolha, antes de o professor observar a política, continua uniforme. |
| Precisão numérica | Combina saídas do crítico e arrays NumPy em operações escalares | Converte os valores para `float64` na fórmula. A rede continua com a precisão do agente | Reduzir arredondamentos na exponencial. O teste compara probabilidades com tolerância numérica. |
| Periodicidade | No início e a cada 5.120 passos de treino, antes da atualização PPO daquele rollout | A mesma cadência em um callback específico | Mover o cálculo para depois da atualização usaria outro crítico e afetaria o currículo. |
| Execução lado a lado | O código original expõe o professor dos autores | Os executores individuais e de múltiplas sementes aceitam os dois professores e identificam cada saída. O padrão desta branch é `proxcorl-ic` | Impedir que as curvas de implementações diferentes sejam misturadas. |
| Dependência | W&B 0.15.2 instalado sem restrição direta para Sentry | Fixamos `sentry-sdk==2.75.0` em `requirements-pointmass.txt` | Nesta instalação, Sentry 3.0.0 interrompeu o processo antes do treino com `AttributeError: module 'sentry_sdk' has no attribute 'Hub'`. Com 2.75.0 o treino iniciou. |
| Teste, comparação e entrega | O repositório dos autores contém código e parâmetros | Teste da distribuição contra o método de sorteio original, CSV digitalizado da Figura 2, curvas sobrepostas, diferença em cada passo, documentação e resultados desta execução | Validar a fórmula separadamente do desempenho do agente e permitir auditoria dos resultados. O agregador verifica a quantidade esperada de tarefas, os retornos binários, os dados, o código de treino e os metadados de execução. Uma edição apenas de gráficos ou textos permite aproveitar os treinos compatíveis. |

## Adaptações que já existiam no fork e entram no resultado

| Parte | Diferença frente ao original | Por que importa |
|---|---|---|
| Entrada de treino | `run_pointmass.py` carrega apenas PointMass e escolhe perfis `paper` ou `smoke`. O `main.py` dos autores continua disponível | Evita carregar os ambientes que não participam deste experimento. O smoke serve para testar instalação. |
| Gymnasium | O original já importava Gymnasium. O fork ajustou `reset(seed, options)`, observações em `float32`, recompensa em `float`, término em `bool`, truncamento em `False` e reinício do índice de avaliação com semente | Satisfazer o contrato da biblioteca. O limite de 100 passos continua sendo tratado como terminal, preservando o comportamento de bootstrap original. |
| PPO | O `main.py` passa batch 128 e entropia 0,01 no ramo com W&B. Sem W&B omite esses argumentos, ativando os padrões do SB3 (64 e 0). Nosso executor passa sempre os valores declarados para PointMass | Evitar que a escolha de logging mude o agente. A taxa de aprendizado 0,0003 e os demais padrões compatíveis também são explícitos. |
| Currículo | O argumento genérico no `main.py` começa em β=110. O arquivo de configuração de PointMass usa β=130 | O apêndice D.2 especifica 130 para os dois cenários de PointMass. |
| Sementes | O original entrega a semente ao PPO. O fork semeia Python, NumPy e PyTorch antes de construir os ambientes e aceita sementes no reset. A avaliação final adicional usa semente de treino + 100.000 | Tornar repetível inclusive a inicialização anterior ao PPO. As sementes 0–24 são uma escolha nossa. A lista exata dos autores não consta no repositório consultado. |
| Avaliação e persistência | Avaliações periódicas estocásticas, sempre salvas por passo. O fork adicionou logs locais, versão, hashes, Monitor, melhor modelo e modelo final por pasta, além de avaliação final estocástica separada | Comparar a curva de treino com a Figura 2. O desempenho do melhor checkpoint e a avaliação adicional do modelo final respondem a perguntas distintas. |
| Dependências | Python 3.10 explicitado em `.python-version`, requisitos específicos para PointMass e registro de versões Linux. Os imports dos solvers de CURROT foram adiados até a escolha desse método | Instalar o necessário para PPO/PointMass sem exigir solvers de outros currículos. |
| Suíte e relatórios | `run_pointmass_suite.py` organiza condições, sementes e workers, preserva pastas parciais, valida orçamento e grade de avaliação e calcula média/erro padrão entre sementes. Arquivos auxiliares incluem configuração JSON, testes, documentação histórica, status de smoke e regras de `.gitignore` | Prevenir sobrescrita de treinos e confusão entre variância dos episódios e variância entre agentes treinados. |

## Controles usados na comparação

Reutilizamos a física, colisões, recompensa binária, dados CSV, política ator–crítico e implementação de PPO do SB3. A observação combina posição e velocidade nos dois eixos com três variáveis de contexto. A rede usa uma camada compartilhada de 64 unidades e dois ramos de 64. Usamos γ=0,99, GAE=0,95, clipping=0,2, dez épocas e uma thread de CPU por agente. O orçamento de dois milhões resulta em 2.001.920 passos efetivos por execução porque o último rollout se completa. O ponto do gráfico em dois milhões é medido durante esse rollout.

Em PM-S:1T, o professor usa o alvo nominal (0,9; 0,5; 3,5), e o CSV oficial de avaliação contém (0,9011806; 0,5; 3,5277016). Preservamos essa diferença para medir o agente na mesma tarefa de teste do código publicado. Em PM-S:2G, cada avaliação periódica percorre os 100 contextos de teste dez vezes. Em PM-S:1T são 100 episódios por avaliação. Os passos de avaliação ficam fora do orçamento de treinamento.

Este estudo cobre os dois cenários PointMass da Figura 2. SGR, MiniGrid e Bipedal Walker, assim como a bateria completa de currículos concorrentes, ficam fora dos resultados medidos aqui. Os arquivos originais desses ambientes continuam no histórico do projeto.

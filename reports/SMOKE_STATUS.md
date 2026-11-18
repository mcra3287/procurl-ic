# Estado da validação — 17/11/2026

## Executado

| Verificação | Resultado |
|---|---|
| Ambiente isolado | Python 3.10.22, Linux/CPU, PyTorch 1.13.1+cpu, SB3 2.2.1 |
| Testes de contrato e regressão | Cinco testes passaram |
| ProCuRL-Target (`proxcorl`), PM-S:1T | Treino de 10.240 passos concluído, seed 420 |
| Pool de tarefas | 20.000 tarefas de treino e 400 amostras-alvo, sem redução |
| PPO efetivo no checkpoint | n_steps=5120, batch_size=128, ent_coef=0.01 |
| Checkpoint | Salvo e recarregado em processo independente |
| Repetição com seed 420 | Parâmetros da política idênticos bit a bit e registros de avaliação idênticos |
| Avaliação final | Retorno médio 0,0; zero acertos em dez episódios; ações estocásticas |
| Duração nesta máquina | Aproximadamente catorze segundos por teste curto |

Esses resultados validam o fluxo de treino, seleção de tarefas, avaliação e persistência. **Não demonstram reprodução das curvas nem desempenho competitivo.** Foram feitas apenas duas atualizações PPO. Nenhum resultado de baseline concorrente ou da contribuição da IC foi produzido.

A pasta `reports/smoke-proxcorl-420` contém os metadados e métricas da primeira execução. Checkpoints e avaliações brutas acompanham o pacote de entrega em `results/`, mas não são versionados. Os experimentos foram executados sobre o commit de origem com alterações locais registradas como `working_tree_dirty: true`, antes do commit da preparação.

## Ainda não executado

- Treino completo de dois milhões de passos.
- Média sobre 25 seeds.
- PM-S:2G e concorrentes.
- Instalação e execução no macOS.
- Implementação da hipótese da IC.

Não há inferência sobre superioridade científica nesta etapa.

# Reprodução: mudanças e limites

> Registro histórico da primeira adaptação. A implementação da IC e a comparação com a Figura 2 estão nas [diferenças atuais](DIFERENCAS.md).

## Origem

Repositório: https://github.com/machine-teaching-group/ijcai2024-proximal-curriculum-target-rl

Commit-base: `9c7df278fb2520188d4dafb10f5888acec67bd4d`.
Referência original no GitHub: branch `main`. Tag local no pacote: `upstream-ijcai2024`. Branch de trabalho: `reproduction-pointmass`.

Nenhuma contribuição de pesquisa nova está implementada. O ambiente, dados, cálculo do kernel, regra de seleção, professor e política são os dos autores, com os ajustes abaixo.

## Ajustes

1. **Entrada dedicada:** `run_pointmass.py` reutiliza PointmassWrapper, PointMassTeacher e CustomActorCriticPolicy. Evita importar ambientes não utilizados. `main.py` foi preservado.
2. **PPO explícito:** o caminho original sem W&B omite batch_size, learning_rate, ent_coef e demais parâmetros definidos para o caminho com W&B. A nova entrada passa a configuração explícita do experimento. Não é uma melhoria proposta do currículo.
3. **Dependências opcionais:** imports de CURROT movidos para sua inicialização. ProCuRL-Target pode executar sem Gurobi/cyipopt/GP.
4. **Interface Gymnasium:** reset aceita seed/options; observações float32; truncamento False em vez de None; recompensa e término têm tipos explícitos.
5. **Física e horizonte preservados:** o original trata o limite de cem passos como término. Essa semântica foi mantida, sem modificar silenciosamente o bootstrap do PPO.
6. **Seeds:** Python, NumPy e PyTorch são semeados antes da construção dos ambientes. Preservamos o gerador global NumPy usado na dinâmica, sem uma migração de RNGs.
7. **Rastreabilidade:** pasta nova por execução, versões, configuração efetiva, hashes dos dados e checkpoints. Avaliação final estocástica, com seed registrada.
8. **Ambiente isolado:** Python 3.10, versões principais fixadas e PyTorch CPU no Linux. Requirements original intacto.

Os testes exercitam Gymnasium, colisão, sucesso, horizonte e repetição da transição com seed. O checker usa um wrapper contextual, pois o ambiente mantém limites internos 4D para a física e oferece observações 7D por meio dos wrappers.

## Configuração

Smoke: 10.240 passos; 20 mil tarefas; 400 amostras-alvo.

Paper: dois milhões de passos, rollout de 5.120, batch de 128, dez épocas, entropia 0,01, beta 130, rede compartilhada 64 e ramos 64/64. Demais valores em `configs/pointmass-paper.json`.

PM-S:1T: cem episódios por avaliação no perfil paper. PM-S:2G: dez episódios por tarefa do CSV de avaliação. Uma avaliação estocástica final usa seed treino + 100000. As avaliações periódicas usam EvalCallback, como o código original.

O perfil curto não sustenta conclusões sobre superioridade ou reprodução de desempenho. Instalação no macOS, outros ambientes e outros baselines permanecem fora da validação desta etapa.

## Próximos experimentos

1. Confirmar a instalação na máquina de execução do projeto.
2. Executar PM-S:1T com orçamento completo e conferir curvas/protocolo.
3. Validar concorrentes no mesmo fluxo.
4. Executar várias seeds e PM-S:2G.
5. Implementar a hipótese da IC separadamente e repetir os controles.

Manter constantes os dados, parâmetros do agente, orçamento de interações, frequência de avaliação e escolha do checkpoint. Reportar o custo adicional do professor. Correções futuras do código-base devem ser registradas e aplicadas consistentemente às versões comparadas.

# Atividade: 1. Data — ANN & DL (2025.2)

Esta página documenta meu notebook `codes.ipynb` para a atividade **1. Data** da disciplina *Artificial Neural Networks and Deep Learning* (ANN & DL). A atividade é composta por três exercícios e segue exatamente as instruções oficiais do curso. Consulte o enunciado para detalhes dos requisitos e critérios de avaliação.

---

## Visão geral

- **Objetivo**: gerar/inspecionar dados e prepará-los para redes neurais, incluindo normalização adequada para camadas com `tanh`.
- **Entrega**: página no GitHub Pages com explicações, código e visualizações solicitadas.
- **Dataset real (Ex. 3)**: *Spaceship Titanic*, cujo alvo `Transported` indica se o passageiro foi transportado para outra dimensão.

---

## Ambiente e execução

1. **Abrir o notebook**: `codes.ipynb`.
2. **Executar as células na ordem** (Kernel → Restart & Run All) para gerar os gráficos e tabelas.
3. **Exportar gráficos** (se necessário) para incluir na página: use `plt.savefig('figs/<nome>.png', dpi=150)` antes de `plt.show()`.

> Observação: a página foi escrita considerando o conteúdo já presente no notebook (células de geração de dados e visualização nos Exercícios 1 e 2, e seção de pré-processamento para o Exercício 3).

---

## Exercício 1 — *Exploring Class Separability in 2D*

### O que fiz
- **Geração de dados 2D**: criação de um dicionário `data` com chaves `x`, `y`, `class` e `color`, seguido de conversão para `DataFrame` (`df`).  
- **Visualização**: `df.plot.scatter(x='x', y='y', c='color', colormap='viridis')` para destacar classes por cor.
- **Análise**: discussão textual sobre sobreposição e possíveis fronteiras de decisão lineares entre grupos (registrada em célula markdown).

### Como reproduzir
- Execute as três primeiras células do notebook (importações, criação do `DataFrame` e `scatter`). Garanta que cada classe tenha uma cor distinta.
- **Responda às perguntas do enunciado**: descreva distribuição/overlap e esboce (ou explique) as possíveis *decision boundaries*.

**Checklist (rubrica do curso)**  
☑ Dados gerados e *scatter* claro.  
☑ Análise de separabilidade e fronteiras coerentes.  

---

## Exercício 2 — *Non-Linearity in Higher Dimensions*

### O que fiz
- **Semente reprodutível** com NumPy (ex.: `rng = np.random.default_rng(42)`).
- **Geração 5D** para duas classes com médias/covariâncias distintas (ver enunciado para os valores).  
- **Redução de dimensionalidade (PCA)**: projeção para 2D e *scatter* colorido por classe.

> Dica: conferi que a projeção 2D evidencie relações não lineares, motivando arquiteturas com ativações não lineares (MLP) em vez de modelos lineares simples.  

**Checklist (rubrica do curso)**  
☑ Dados 5D gerados conforme parâmetros.  
☑ PCA aplicado e *scatter* claro.  
☑ Discussão sobre (não) linearidade e por que redes com não linearidades são adequadas.  

---

## Exercício 3 — *Preparing Real-World Data for a Neural Network*

### Dados
- **Fonte**: Kaggle — *Spaceship Titanic*.  
- **Alvo (`Transported`)**: binário; indica se o passageiro foi transportado para outra dimensão.  
- **Exemplos de atributos**: `HomePlanet`, `CryoSleep`, `Destination` (categóricos) e `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` (numéricos), além de `VIP`, `Cabin`, `Name`.

> Dica: o conjunto costuma estar razoavelmente balanceado entre `True`/`False` para `Transported`, o que ajuda na avaliação de classificadores.  

### Pipeline de pré-processamento adotado
1. **Carregamento e inspeção**: `df.info()`, `df.isna().sum()` para mapear *missing* por coluna.  
2. **Tratamento de ausentes**:  
   - Numéricos: imputação com mediana (robusto a *outliers*).  
   - Categóricos: imputação com *most frequent* (modo) ou categoria explícita `"Unknown"` quando apropriado — documente a escolha.  
3. **Codificação categórica**: *one-hot encoding* para `HomePlanet`, `CryoSleep`, `Destination`, `VIP` (usar `.astype(str)` se necessário antes do `get_dummies`).  
4. **Escalonamento para `tanh`**:  
   - Justificativa: `tanh` é centrada em zero e satura fora de ~±2; padronizar (média 0, desvio 1) ou normalizar para `[-1, 1]` acelera o treino e evita saturação.  
   - Implementação: `StandardScaler()` **ou** normalização min-max seguida de mapeamento para `[-1, 1]`. O enunciado aceita ambos; documentei a opção utilizada.  
5. **Visualizações**: histogramas de `Age` e/ou `FoodCourt` **antes e depois** do escalonamento para evidenciar o efeito da transformação.  

### Saídas esperadas
- Tabela (ou *print*) com contagem de *missing* por coluna.  
- Lista das colunas numéricas vs. categóricas com a estratégia aplicada.  
- Gráficos comparando distribuições pré vs. pós-escalonamento.  
- `df_final` pronto para treino; `Transported` separado em `y`/`transported` e *features* em `X`/`df_train` (seu notebook pode optar por nomes equivalentes).

**Checklist (rubrica do curso)**  
☑ Dados carregados e descritos corretamente.  
☑ *Missing*, *encoding* e *scaling* (centrado para `tanh`) implementados e justificados.  
☑ Visualizações mostrando o impacto do pré-processamento.  

---

## Decisões e justificativas

- **Imputação**: mediana/mode reduzem viés com *outliers* e preservam categorias; quando a semântica faz sentido, uso de `"Unknown"` evita descartar linhas.  
- **One-hot** em categóricas: mantém relações não ordinais entre categorias (evita codificação numérica arbitrária).  
- **Escalonamento centrado**: com `tanh`, entradas centradas (média≈0) ajudam a manter ativações na zona quase linear e gradientes estáveis — melhorando convergência.  

---

## Conclusões

- **Ex. 1**: a distribuição 2D mostra regiões com sobreposição parcial; fronteiras estritamente lineares não separam todos os grupos com folga.  
- **Ex. 2**: a projeção 2D (via PCA) evidencia padrões não lineares, motivando MLPs com ativações não lineares.  
- **Ex. 3**: com imputação adequada, *one-hot* e escalonamento centrado, o *dataset* fica pronto para uma MLP com camadas `tanh`, atendendo às exigências do enunciado.  

---

## Referências

- Enunciado oficial e rubrica da atividade (versão **2025.2**).  
- Descrição do *Spaceship Titanic* e do alvo `Transported`.  
- Exemplos de significado das colunas e guias de EDA no *Spaceship Titanic*.  

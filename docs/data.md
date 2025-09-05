# Atividade: 1. Data — ANN & DL (2025.2)

Esta página documenta meu notebook `codes.ipynb` para a atividade **1. Data** da disciplina *Artificial Neural Networks and Deep Learning* (ANN & DL). A atividade é composta por três exercícios e segue exatamente as instruções oficiais do curso. Consulte o enunciado para detalhes dos requisitos e critérios de avaliação.

---


## Exercício 1 — *Exploring Class Separability in 2D*

## Generate  the data:

```python
data = {'x': [], 'y':[], 'class':[], 'color': []}


mean_std_dev = [([2,3], np.diag([0.8, 2.5])) ,
([5,6], np.diag([1.2, 1.9])),
([8,1], np.diag([0.9, 0.9])),
([15,4], np.diag([0.5, 2.0]))]
mean = [2, 3]
cov = np.diag([0.8, 2.5])
for i in range(0, 4):

    mean = mean_std_dev[i][0]
    cov = mean_std_dev[i][1]

    x, y = np.random.multivariate_normal(mean, cov, 100).T
    classe = []
    for c in range(0, len(x)):
        data['x'].append(x[c])
        data['y'].append(y[c])
        data['class'].append(f"class_{i}")
        if i == 0:
            data['color'].append((1.0, 0.0, 0.0))
        if i == 1:
            data['color'].append((1.0,0.0,1.0,))
        if i == 2:
            data['color'].append((1.0, 1.0, 0.0))
        if i == 3:
            data['color'].append((1.0, 0.5, 0.5))
```

### Ploting the Data
![Texto alternativo](docs/img/imagem.png)

### Analyze and Draw Boundaries
a. Percebe-se que as duas classes mais a direita estão bme disntintas. Enquanto, que as duas a mais esquerda, vermelho e roxo, apresentam um overlap entre elas.

b.Observando o plot precisa de tres  linear boundays, para poder distinguir, uma vez que com uma apenas daria para separar entre duas metade  esquerdo e direito, e teria duas classes em cada.Além de que com duas um lado ficaria com duas clssses, sendo necessário então uma terceira linha para separar estas duas.

c.
![Texto alternativo](docs/img/imagem2.png)
---

## Exercício 2 — *Non-Linearity in Higher Dimensions*

## Generate  the data:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
data = {
    'x1': [], 'x2': [], 'x3': [], 'x4': [], 'x5': [],
    'class': [], 'color': []
}
params = [
    (
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([
            [1.0, 0.8, 0.1, 0.0, 0.0],
            [0.8, 1.0, 0.3, 0.0, 0.0],
            [0.1, 0.3, 1.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 1.0, 0.2],
            [0.0, 0.0, 0.0, 0.2, 1.0],
        ]),
        "class_A",
        (1.0, 0.0, 0.0), 
    ),
    (
        np.array([1.5, 1.5, 1.5, 1.5, 1.5]),
        np.array([
            [1.5, -0.7, 0.2, 0.0, 0.0],
            [-0.7, 1.5, 0.4, 0.0, 0.0],
            [0.2, 0.4, 1.5, 0.6, 0.0],
            [0.0, 0.0, 0.6, 1.5, 0.3],
            [0.0, 0.0, 0.0, 0.3, 1.5],
        ]),
        "class_B",
        (0.0, 0.0, 1.0), 
    ),
]

n_per_class = 500

for mean, cov, label, color in params:
    samples = rng.multivariate_normal(mean, cov, size=n_per_class)
    for s in samples:
        data['x1'].append(s[0])
        data['x2'].append(s[1])
        data['x3'].append(s[2])
        data['x4'].append(s[3])
        data['x5'].append(s[4])
        data['class'].append(label)
        data['color'].append(color)


df = pd.DataFrame(data)

```

### Visualize the data 
![](docs/imgs/image1.png)

### Analyzethe plot
a. Observando percebe-se que os vermelhos tem a tendencia a ficar no lado esquerdo, enquanto os azuis tendem a ficar no lado direito. No entanto, isto não é verdade para todos,uma vez que é possível encontrar alguns pontos invadindo os territórios, ou seja, se desenhasse uma linha no meio não teria uma acuracia muito boa, por conta desta mistura entre as classes.

b.Este tipo de data impoe um desafio,uma vez que o modelo teria dificuldade para pegar esses corner cases, uma vez que precisaria de um grande numero de linhas para contornar, o que também pode pender para um overfitting.

c.
![](docs/imgs/image2.png)

---

## Exercício 3 — *Preparing Real-World Data for a Neural Network*

### Describe the Data
- O objetivo do dataset "Spaceship Titanic" é prever se os passageiros foram transportados para outra dimensão, sendo que a coluna "Transported" representa isto, ou seja se nela estiver o valor 1 significa que o passageiro foi tranportado, caso seja 0 não.

- Features numericos: Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
- Features categóricos: PassengerId, HomePlanet, CryoSleep, Cabin, Destination, VIP, Name

-Colunas com "missing values" e as quantidades
Colunas com faltantes:
CryoSleep       217
ShoppingMall    208
VIP             203
HomePlanet      201
Name            200
Cabin           199
VRDeck          188
FoodCourt       183
Spa             183
Destination     182
RoomService     181
Age             179




## Referências

- Descrição do *Spaceship Titanic* e do alvo `Transported`: https://www.kaggle.com/competitions/spaceship-titanic/data.  


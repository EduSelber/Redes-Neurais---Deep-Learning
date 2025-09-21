# Atividade: 3. MLP

---
## Exercício 1
## A brief description of my approach

Implemento, em código direto e sem funções, o fluxo completo forward → loss → backward → update de um MLP com tanh na camada escondida e na saída, usando MSE para uma única amostra. Imprimo todas as grandezas intermediárias (pré-ativações, ativações, gradientes de pesos e vieses) e faço a atualização com η = 0,1, para validar numericamente a cadeia de derivadas que será reutilizada nos demais exercícios.

## Code:

```python

import numpy as np


x = np.array([0.5, -0.2], dtype=float)
y = 1.0

W1 = np.array([[0.3, -0.1],
               [0.2,  0.4]], dtype=float)
b1 = np.array([ 0.1, -0.2], dtype=float)

W2 = np.array([0.5, -0.3], dtype=float)  
b2 = 0.2
eta_update = 0.1


z1 = W1 @ x + b1
h1 = np.tanh(z1)
u2 = W2 @ h1 + b2
yhat = np.tanh(u2)


L = (y - yhat)**2

print("Forward pass:")
print("z1 =", z1)
print("h1 = tanh(z1) =", h1)
print("u2 =", u2)
print("yhat = tanh(u2) =", yhat)
print("Loss L =", L)


dL_dyhat = 2*(yhat - y)               
d_tanh_u2 = 1 - np.tanh(u2)**2
dL_du2 = dL_dyhat * d_tanh_u2


dL_dW2 = dL_du2 * h1                 
dL_db2 = dL_du2


dL_dh1 = dL_du2 * W2
d_tanh_z1 = 1 - np.tanh(z1)**2
dL_dz1 = dL_dh1 * d_tanh_z1            
dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1

print("\nBackward pass (gradients):")
print("dL/dyhat =", dL_dyhat)
print("dL/du2   =", dL_du2)
print("dL/dW2   =", dL_dW2)
print("dL/db2   =", dL_db2)
print("dL/dh1   =", dL_dh1)
print("dL/dz1   =", dL_dz1)
print("dL/dW1   =\n", dL_dW1)
print("dL/db1   =", dL_db1)


W2_new = W2 - eta_update * dL_dW2
b2_new = b2 - eta_update * dL_db2
W1_new = W1 - eta_update * dL_dW1
b1_new = b1 - eta_update * dL_db1

print("\nUpdated parameters (eta = 0.1):")
print("W2_new =", W2_new)
print("b2_new =", b2_new)
print("W1_new =\n", W1_new)
print("b1_new =", b1_new)

```
## Result:
```bash
Forward pass:
z1 = [ 0.27 -0.18]
h1 = tanh(z1) = [ 0.26362484 -0.17808087]
u2 = 0.38523667817130075
yhat = tanh(u2) = 0.36724656264510797
Loss L = 0.4003769124844312

Backward pass (gradients):
dL/dyhat = -1.265506874709784
dL/du2   = -1.0948279147135995
dL/dW2   = [-0.28862383  0.19496791]
dL/db2   = -1.0948279147135995
dL/dh1   = [-0.54741396  0.32844837]
dL/dz1   = [-0.50936975  0.31803236]
dL/dW1   =
 [[-0.25468488  0.10187395]
 [ 0.15901618 -0.06360647]]
dL/db1   = [-0.50936975  0.31803236]

Updated parameters (eta = 0.1):
W2_new = [ 0.52886238 -0.31949679]
b2_new = 0.30948279147136
W1_new =
 [[ 0.32546849 -0.1101874 ]
 [ 0.18409838  0.40636065]]
b1_new = [ 0.15093698 -0.23180324]

```
---
## Exercício 2
## A brief description of my approach

Gero um conjunto binário com make_classification, em que a classe 0 tem 1 cluster e a classe 1 tem 2 clusters. Treino um MLP 2–32–1 escrito inline tanh na escondida, sigmoid na saída e Binary Cross-Entropy como perda. O treinamento é via mini-batch gradient descent, com retropropagação manual dos gradientes e avaliação por acurácia no conjunto de teste.

## Code:

```python

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

n_total = 1000
X0, y0 = make_classification(
    n_samples=n_total//2, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, n_classes=2, class_sep=1.3, flip_y=0.0, random_state=7
)
y0 = (y0==0).astype(int) 

X1a, y1a = make_classification(
    n_samples=n_total//4, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, n_classes=2, class_sep=1.3, flip_y=0.0, random_state=8
)
X1b, y1b = make_classification(
    n_samples=n_total - (n_total//2 + n_total//4), n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, n_classes=2, class_sep=1.3, flip_y=0.0, random_state=9
)
y1a[:] = 1
y1b[:] = 1

X = np.vstack([X0, X1a, X1b]).astype(float)
y = np.hstack([y0, y1a, y1b]).astype(int)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


in_features = 2
h = 32

W1 = rng.normal(0, 1, (h, in_features)) / np.sqrt(in_features)
b1 = np.zeros((h,))
W2 = rng.normal(0, 1, (1, h)) / np.sqrt(h)
b2 = np.zeros((1,))

lr = 5e-2
epochs = 300
batch_size = 64


ytr_ = ytr.reshape(-1, 1).astype(float)
N = Xtr.shape[0]
eps = 1e-12

for ep in range(1, epochs+1):
    idx = rng.permutation(N)
    Xs = Xtr[idx]; ys = ytr_[idx]

    for i in range(0, N, batch_size):
        xb = Xs[i:i+batch_size]
        yb = ys[i:i+batch_size]

      
        z1 = xb @ W1.T + b1
        h1 = np.tanh(z1)

        u2 = h1 @ W2.T + b2 
        yhat = 1.0 / (1.0 + np.exp(-u2))

        yclip = np.clip(yhat, eps, 1-eps)
        loss = -np.mean(yb*np.log(yclip) + (1-yb)*np.log(1-yclip))

    
        B = yb.shape[0]
        dL_dyhat = (yhat - yb) / (yclip*(1-yclip)) / B
        dL_du2 = dL_dyhat * (yhat*(1-yhat))  

        dL_dW2 = dL_du2.T @ h1
        dL_db2 = dL_du2.sum(axis=0)

        dL_dh1 = dL_du2 @ W2
        dL_dz1 = dL_dh1 * (1 - np.tanh(z1)**2)

        dL_dW1 = dL_dz1.T @ xb
        dL_db1 = dL_dz1.sum(axis=0)


        W2 -= lr * dL_dW2
        b2 -= lr * dL_db2
        W1 -= lr * dL_dW1
        b1 -= lr * dL_db1

    if ep % 50 == 0 or ep == 1:
        print(f"epoch {ep:3d} | loss {loss:.4f}")


z1 = Xte @ W1.T + b1
h1 = np.tanh(z1)
u2 = h1 @ W2.T + b2
yhat = 1.0 / (1.0 + np.exp(-u2))
ypred = (yhat.ravel() >= 0.5).astype(int)
acc = (ypred == yte).mean()
print(f"\nTest accuracy (binary): {acc:.3f}")


```
## Result:
```bash
epoch   1 | loss 0.4612
epoch  50 | loss 0.2585
epoch 100 | loss 0.1581
epoch 150 | loss 0.2104
epoch 200 | loss 0.1864
epoch 250 | loss 0.2356
epoch 300 | loss 0.1400

Test accuracy (binary): 0.890

```
---
## Exercício 3
## A brief description of my approach

Formulo um problema multiclasse (3 classes, 4 features), construindo 2/3/4 clusters por classe pela união de subconjuntos. Treino um MLP 4–64–3 também inline, trocando a cabeça para softmax + cross-entropy. Para estabilidade numérica, subtraio o máximo por amostra antes do exp e aplico clipping nas probabilidades; a lógica de treino e retroprop permanece a mesma do Ex. 2, evidenciando reuso conceitual.

## Code:

```python

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(123)

n_total = 1500
n_per = n_total // 3


sizes0 = [n_per//2, n_per - n_per//2]
X0_parts = []
for k, ns in enumerate(sizes0):
    Xk, yk = make_classification(n_samples=ns, n_features=4, n_informative=4, n_redundant=0,
                                 n_classes=2, n_clusters_per_class=1, class_sep=1.5,
                                 flip_y=0.0, random_state=30 + k)
    yk[:] = 0
    X0_parts.append((Xk, yk))
X0 = np.vstack([p[0] for p in X0_parts]); y0 = np.hstack([p[1] for p in X0_parts])


sizes1 = [n_per//3, n_per//3, n_per - 2*(n_per//3)]
X1_parts = []
for k, ns in enumerate(sizes1):
    Xk, yk = make_classification(n_samples=ns, n_features=4, n_informative=4, n_redundant=0,
                                 n_classes=2, n_clusters_per_class=1, class_sep=1.5,
                                 flip_y=0.0, random_state=40 + k)
    yk[:] = 1
    X1_parts.append((Xk, yk))
X1 = np.vstack([p[0] for p in X1_parts]); y1 = np.hstack([p[1] for p in X1_parts])


sizes2 = [n_per//4, n_per//4, n_per//4, n_per - 3*(n_per//4)]
X2_parts = []
for k, ns in enumerate(sizes2):
    Xk, yk = make_classification(n_samples=ns, n_features=4, n_informative=4, n_redundant=0,
                                 n_classes=2, n_clusters_per_class=1, class_sep=1.5,
                                 flip_y=0.0, random_state=50 + k)
    yk[:] = 2
    X2_parts.append((Xk, yk))
X2 = np.vstack([p[0] for p in X2_parts]); y2 = np.hstack([p[1] for p in X2_parts])

X = np.vstack([X0, X1, X2]).astype(float)
y = np.hstack([y0, y1, y2]).astype(int)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

C = 3
ytr_oh = np.eye(C)[ytr]
yte_oh = np.eye(C)[yte]


in_features = 4
h = 64
out_features = 3

W1 = rng.normal(0, 1, (h, in_features)) / np.sqrt(in_features)
b1 = np.zeros((h,))
W2 = rng.normal(0, 1, (out_features, h)) / np.sqrt(h)
b2 = np.zeros((out_features,))

lr = 5e-2
epochs = 300
batch_size = 64
eps = 1e-12

N = Xtr.shape[0]

for ep in range(1, epochs+1):
    idx = rng.permutation(N)
    Xs = Xtr[idx]; ys = ytr_oh[idx]

    for i in range(0, N, batch_size):
        xb = Xs[i:i+batch_size]
        yb = ys[i:i+batch_size]


        z1 = xb @ W1.T + b1
        h1 = np.tanh(z1)

        U = h1 @ W2.T + b2
        Umax = U.max(axis=1, keepdims=True)
        ex = np.exp(U - Umax)
        yhat = ex / ex.sum(axis=1, keepdims=True)

        yclip = np.clip(yhat, eps, 1-eps)
        loss = -np.mean(np.sum(yb * np.log(yclip), axis=1))

        B = yb.shape[0]
        dL_dU = (yhat - yb) / B 

        dL_dW2 = dL_dU.T @ h1
        dL_db2 = dL_dU.sum(axis=0)

        dL_dh1 = dL_dU @ W2
        dL_dz1 = dL_dh1 * (1 - np.tanh(z1)**2)

        dL_dW1 = dL_dz1.T @ xb
        dL_db1 = dL_dz1.sum(axis=0)

        W2 -= lr * dL_dW2
        b2 -= lr * dL_db2
        W1 -= lr * dL_dW1
        b1 -= lr * dL_db1

    if ep % 50 == 0 or ep == 1:
        print(f"epoch {ep:3d} | loss {loss:.4f}")

z1 = Xte @ W1.T + b1
h1 = np.tanh(z1)
U = h1 @ W2.T + b2
Umax = U.max(axis=1, keepdims=True)
ex = np.exp(U - Umax)
yhat = ex / ex.sum(axis=1, keepdims=True)
ypred = np.argmax(yhat, axis=1)
acc = (ypred == yte).mean()
print(f"\nTest accuracy (3 classes): {acc:.3f}")



```
## Result:
```bash
epoch   1 | loss 0.8970
epoch  50 | loss 0.5367
epoch 100 | loss 0.3963
epoch 150 | loss 0.3406
epoch 200 | loss 0.3817
epoch 250 | loss 0.3475
epoch 300 | loss 0.2155

Test accuracy (3 classes): 0.817
```
---
## Exercício 4
## A brief description of my approach

Repito o cenário do Ex. 3, mas aprofundo o modelo para 4–64–32–3, adicionando uma segunda camada escondida. Mantenho a implementação inline e os mesmos cuidados numéricos (softmax estabilizado + CE), comparando a acurácia de teste com a versão menos profunda para observar o impacto da profundidade sob uma rotina de treino e backprop idêntica.

## Code:

```python

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(321)


n_total = 1500
n_per = n_total // 3


sizes0 = [n_per//2, n_per - n_per//2]
X0_parts = []
for k, ns in enumerate(sizes0):
    Xk, yk = make_classification(n_samples=ns, n_features=4, n_informative=4, n_redundant=0,
                                 n_classes=2, n_clusters_per_class=1, class_sep=1.5,
                                 flip_y=0.0, random_state=70 + k)
    yk[:] = 0
    X0_parts.append((Xk, yk))
X0 = np.vstack([p[0] for p in X0_parts]); y0 = np.hstack([p[1] for p in X0_parts])


sizes1 = [n_per//3, n_per//3, n_per - 2*(n_per//3)]
X1_parts = []
for k, ns in enumerate(sizes1):
    Xk, yk = make_classification(n_samples=ns, n_features=4, n_informative=4, n_redundant=0,
                                 n_classes=2, n_clusters_per_class=1, class_sep=1.5,
                                 flip_y=0.0, random_state=80 + k)
    yk[:] = 1
    X1_parts.append((Xk, yk))
X1 = np.vstack([p[0] for p in X1_parts]); y1 = np.hstack([p[1] for p in X1_parts])


sizes2 = [n_per//4, n_per//4, n_per//4, n_per - 3*(n_per//4)]
X2_parts = []
for k, ns in enumerate(sizes2):
    Xk, yk = make_classification(n_samples=ns, n_features=4, n_informative=4, n_redundant=0,
                                 n_classes=2, n_clusters_per_class=1, class_sep=1.5,
                                 flip_y=0.0, random_state=90 + k)
    yk[:] = 2
    X2_parts.append((Xk, yk))
X2 = np.vstack([p[0] for p in X2_parts]); y2 = np.hstack([p[1] for p in X2_parts])

X = np.vstack([X0, X1, X2]).astype(float)
y = np.hstack([y0, y1, y2]).astype(int)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


C = 3
ytr_oh = np.eye(C)[ytr]

in_features = 4
h1_size = 64
h2_size = 32
out_features = 3

W1 = rng.normal(0, 1, (h1_size, in_features)) / np.sqrt(in_features)
b1 = np.zeros((h1_size,))

W2 = rng.normal(0, 1, (h2_size, h1_size)) / np.sqrt(h1_size)
b2 = np.zeros((h2_size,))

W3 = rng.normal(0, 1, (out_features, h2_size)) / np.sqrt(h2_size)
b3 = np.zeros((out_features,))

lr = 3e-2
epochs = 350
batch_size = 64
eps = 1e-12

N = Xtr.shape[0]

for ep in range(1, epochs+1):
    idx = rng.permutation(N)
    Xs = Xtr[idx]; ys = ytr_oh[idx]

    for i in range(0, N, batch_size):
        xb = Xs[i:i+batch_size]
        yb = ys[i:i+batch_size]


        z1 = xb @ W1.T + b1
        h1 = np.tanh(z1)

        z2 = h1 @ W2.T + b2
        h2 = np.tanh(z2)

        U = h2 @ W3.T + b3 
        Umax = U.max(axis=1, keepdims=True)
        ex = np.exp(U - Umax)
        yhat = ex / ex.sum(axis=1, keepdims=True)

       
        yclip = np.clip(yhat, eps, 1-eps)
        loss = -np.mean(np.sum(yb * np.log(yclip), axis=1))

        
        B = yb.shape[0]
        dL_dU = (yhat - yb) / B  

        dL_dW3 = dL_dU.T @ h2
        dL_db3 = dL_dU.sum(axis=0)

        dL_dh2 = dL_dU @ W3
        dL_dz2 = dL_dh2 * (1 - np.tanh(z2)**2)

        dL_dW2 = dL_dz2.T @ h1
        dL_db2 = dL_dz2.sum(axis=0)

        dL_dh1 = dL_dz2 @ W2
        dL_dz1 = dL_dh1 * (1 - np.tanh(z1)**2)

        dL_dW1 = dL_dz1.T @ xb
        dL_db1 = dL_dz1.sum(axis=0)

        W3 -= lr * dL_dW3
        b3 -= lr * dL_db3
        W2 -= lr * dL_dW2
        b2 -= lr * dL_db2
        W1 -= lr * dL_dW1
        b1 -= lr * dL_db1

    if ep % 50 == 0 or ep == 1:
        print(f"epoch {ep:3d} | loss {loss:.4f}")


z1 = Xte @ W1.T + b1
h1 = np.tanh(z1)
z2 = h1 @ W2.T + b2
h2 = np.tanh(z2)
U = h2 @ W3.T + b3
Umax = U.max(axis=1, keepdims=True)
ex = np.exp(U - Umax)
yhat = ex / ex.sum(axis=1, keepdims=True)
ypred = np.argmax(yhat, axis=1)
acc = (ypred == yte).mean()
print(f"\nTest accuracy (deeper, 2 hidden): {acc:.3f}")


```
## Result:
```bash
epoch   1 | loss 0.9812
epoch  50 | loss 0.7336
epoch 100 | loss 0.6125
epoch 150 | loss 0.5455
epoch 200 | loss 0.3711
epoch 250 | loss 0.4223
epoch 300 | loss 0.4229
epoch 350 | loss 0.4302

Test accuracy (deeper, 2 hidden): 0.700
```
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression

# X = [horas_estudo, faltas]
X = np.array([
    [1, 10],
    [2,  8],
    [3,  7],
    [6,  3],
    [7,  2],
    [8,  1]
])

# 0 = reprovado
# 1 = aprovado
y = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])

modelo = LogisticRegression()

dobras = LeaveOneOut()

scores = cross_val_score(
    modelo,
    X,
    y,
    cv=dobras,
    scoring="accuracy"
)

print(scores)
print("Acurácia média:", scores.mean())

'''
cross_val_predict() → "o que o modelo previu?"
cross_val_score() → "quão bem o modelo previu?"

            X                    y

Aluno 1 → [1h estudo, 10 faltas] → 0
Aluno 2 → [2h estudo,  8 faltas] → 0
Aluno 3 → [3h estudo,  7 faltas] → 0
Aluno 4 → [6h estudo,  3 faltas] → 1
Aluno 5 → [7h estudo,  2 faltas] → 1
Aluno 6 → [8h estudo,  1 falta ] → 1

            Treino                    Validação

Fold 1 → alunos 2,3,4,5,6      →      aluno 1
Fold 2 → alunos 1,3,4,5,6      →      aluno 2
Fold 3 → alunos 1,2,4,5,6      →      aluno 3
Fold 4 → alunos 1,2,3,5,6      →      aluno 4
Fold 5 → alunos 1,2,3,4,6      →      aluno 5
Fold 6 → alunos 1,2,3,4,5      →      aluno 6
'''
"""
Aprendizado de Maquina - IFSP Presidente Epitacio
Aula 1, Parte 2 - MINI-LABORATORIO (45 min)  ***ARQUIVO DO ALUNO***

NOME: ______________________________  PRONTUARIO: ______________  DATA: __/__/2026

INSTRUCOES
  - Preencha os trechos marcados com  # TODO
  - Use SEMPRE random_state=42
  - Rode o arquivo ate ele executar sem erro:   python lab01_exercicios.py
  - Responda as perguntas de interpretacao no bloco RELATORIO, no fim do arquivo
  - Entregue este arquivo executado (+ curva_k.png se fizer o bonus)

VALE 10,0 PONTOS - integra a nota do Trabalho 2 (Pratico)
  codigo 60%  |  relatorio 40%
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEMENTE = 42


def secao(n, titulo):
    print()
    print("=" * 70)
    print(f"EXERCICIO {n} - {titulo}")
    print("=" * 70)


# ============================================================ EXERCICIO 1 (1,0)
secao(1, "INSPECAO DO CONJUNTO DE DADOS")

dados = load_iris()
X = dados.data
y = dados.target

# TODO 1a: imprima o formato (shape) de X e de y
# TODO 1b: imprima os nomes dos atributos (dados.feature_names)
#          e os nomes das classes (dados.target_names)
# TODO 1c: monte um DataFrame com X e imprima .describe() arredondado com .round(2)
# TODO 1d: imprima a contagem de exemplos por classe
#          dica: pd.Series(y).value_counts()  ou  np.bincount(y)


# ============================================================ EXERCICIO 2 (2,0)
secao(2, "DIVISAO E PRIMEIRO MODELO (k=5)")

# TODO 2a: divida em treino (70%) e teste (30%)
#          use test_size=0.30, random_state=SEMENTE, stratify=y
X_treino, X_teste, y_treino, y_teste = (None, None, None, None)   # <- substitua

# TODO 2b: treine um KNeighborsClassifier com n_neighbors=5

# TODO 2c: preveja em X_teste e imprima:
#          - a acuracia (accuracy_score)
#          - a matriz de confusao (confusion_matrix)
#          - o classification_report com target_names=dados.target_names


# ============================================================ EXERCICIO 3 (2,5)
secao(3, "VARREDURA DE k")

VALORES_K = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
resultados = []   # guarde tuplas (k, acuracia_treino, acuracia_teste)

# TODO 3: para cada k em VALORES_K:
#           - treine um KNN com aquele k
#           - calcule a acuracia NO TREINO e NO TESTE
#           - imprima uma linha:  k=..  treino=0.xxx  teste=0.xxx
#           - guarde em `resultados`
#
# Depois responda no RELATORIO os itens (a), (b), (c) e (d) da apostila.


# ============================================================ EXERCICIO 4 (2,0)
secao(4, "O EFEITO DA ESCALA")

# TODO 4a: crie copias de X_treino e X_teste com a coluna 0 multiplicada por 1000
#          dica: X_tr_ruim = X_treino.copy();  X_tr_ruim[:, 0] *= 1000

# TODO 4b: treine um KNN(k=5) nos dados distorcidos e imprima a acuracia.
#          Compare com a do Exercicio 2.

# TODO 4c: use make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
#          nos dados distorcidos e imprima a acuracia. Comparou?


# ============================================================ EXERCICIO 5 (1,5)
secao(5, "COMPARACAO DE ALGORITMOS")

modelos = {
    "K-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Arvore de decisao": DecisionTreeClassifier(random_state=SEMENTE),
    "Naive Bayes": GaussianNB(),
}

# TODO 5: treine os tres na MESMA divisao do Exercicio 2 e monte uma tabela
#         com o nome do modelo e a acuracia no teste.
#         No relatorio: quantas flores correspondem a 1 ponto percentual?


# ============================================================ EXERCICIO 6 (1,0)
secao(6, "PROVA DE SOBREAJUSTE")

# TODO 6a: treine DecisionTreeClassifier(random_state=SEMENTE) SEM limitar profundidade
#          imprima acuracia no treino e no teste
#          imprima tambem a profundidade obtida: modelo.get_depth()

# TODO 6b: treine DecisionTreeClassifier(max_depth=2, random_state=SEMENTE)
#          imprima acuracia no treino e no teste

# TODO 6c: relacione os quatro numeros com sobreajuste e subajuste no relatorio


# ============================================================ BONUS (+1,0)
secao("BONUS", "CURVA DE VALIDACAO (opcional)")

# TODO bonus: com matplotlib, plote acuracia de treino e de teste em funcao de k
#             (use a lista `resultados` do Exercicio 3), com titulo, rotulos dos
#             eixos e legenda. Salve como curva_k.png
#
# import matplotlib
# matplotlib.use("Agg")          # necessario se a maquina nao tiver interface grafica
# import matplotlib.pyplot as plt
# ...
# plt.savefig("curva_k.png", dpi=150, bbox_inches="tight")


# ============================================================ RELATORIO
"""
=======================================================================
RELATORIO  (10 a 15 linhas - vale 40% da nota)
=======================================================================

Ex.1 - O conjunto esta balanceado? Como voce sabe?
    R:

Ex.2 - Entre quais classes ocorreram os erros? Por que era previsivel,
       considerando as medias por especie do Exercicio 1?
    R:

Ex.3 (a) Qual k deu a melhor acuracia de teste?
    R:
Ex.3 (b) O que acontece com a acuracia de TREINO quando k=1, e por que
         esse valor era previsivel antes de rodar?
    R:
Ex.3 (c) O que acontece com k=51, e qual fenomeno da Parte 1 isso ilustra?
    R:
Ex.3 (d) Qual o problema metodologico de escolher k pela acuracia no teste?
    R:

Ex.4 - Por que o K-NN e sensivel a escala, e por que a padronizacao
       recupera o desempenho?
    R:

Ex.5 - As diferencas entre os tres modelos sao grandes? Um ponto percentual
       corresponde a quantas flores? Da para declarar um vencedor?
    R:

Ex.6 - Relacione os quatro numeros com sobreajuste e subajuste.
    R:

=======================================================================
"""

print("""
=======================================================================
Arquivo executado. Confira se todos os TODO foram preenchidos e se o
bloco RELATORIO esta respondido antes de entregar.
=======================================================================""")

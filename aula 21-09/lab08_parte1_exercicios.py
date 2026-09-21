"""
Aprendizado de Maquina - IFSP Presidente Epitacio
Aula 8, Parte 1 - MINI-LABORATORIO  ***ARQUIVO DO ALUNO***

"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEMENTE = 42
pd.set_option("display.width", 100)


def secao(n, titulo):
    print()
    print("=" * 70)
    print(f"EXERCICIO {n} - {titulo}")
    print("=" * 70)


def gerar_dados_emprestimo(n=450, semente=42):
    """Mesmo dataset das aulas de preparacao de dados: solicitacoes de
    emprestimo, com valores ausentes, uma coluna categorica e outliers."""
    rng = np.random.default_rng(semente)

    idade = rng.normal(38, 12, n).clip(18, 75).round().astype(float)
    score_credito = rng.normal(650, 80, n).clip(300, 850).round().astype(float)
    tempo_emprego = rng.normal(7, 5, n).clip(0, 35).round().astype(float)

    renda = rng.lognormal(mean=8.15, sigma=0.45, size=n)
    idx_outlier = rng.choice(n, size=5, replace=False)
    renda[idx_outlier] *= rng.uniform(6, 10, size=5)
    renda = renda.round(2)

    cidades = np.array(["SP", "RJ", "MG", "Outra"])
    cidade = rng.choice(cidades, size=n, p=[0.42, 0.23, 0.15, 0.20])

    renda_c = np.clip(renda, None, np.percentile(renda, 95))
    z = (
        0.05 * (score_credito - 650)
        + 0.0008 * (renda_c - renda_c.mean())
        + 0.12 * (tempo_emprego - 7)
        + rng.normal(0, 0.35, n)
    )
    prob_aprovado = 1 / (1 + np.exp(-z))
    aprovado = (rng.uniform(0, 1, n) < prob_aprovado).astype(int)

    df = pd.DataFrame({
        "idade": idade,
        "renda": renda,
        "tempo_emprego": tempo_emprego,
        "score_credito": score_credito,
        "cidade": cidade,
        "aprovado": aprovado,
    })

    mask_idade = rng.uniform(0, 1, n) < 0.08
    mask_renda = rng.uniform(0, 1, n) < 0.10
    mask_cidade = rng.uniform(0, 1, n) < 0.06
    df.loc[mask_idade, "idade"] = np.nan
    df.loc[mask_renda, "renda"] = np.nan
    df.loc[mask_cidade, "cidade"] = None
    return df


COL_NUM = ["idade", "renda", "tempo_emprego", "score_credito"]
COL_CAT = ["cidade"]


def montar_preprocessador(escalar=True):
    """Imputacao + (escala) nos numericos; imputacao + one-hot na categorica.
    Tudo dentro de um ColumnTransformer: as estatisticas (mediana, media,
    desvio, categorias) sao aprendidas SO no treino."""
    passos_num = [("imp", SimpleImputer(strategy="median"))]
    if escalar:
        passos_num.append(("esc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(passos_num), COL_NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), COL_CAT),
    ])


def criar_modelos():
    """Os 5 classificadores da demonstracao + random forest (6o modelo)."""
    return {
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Regressao Logistica": LogisticRegression(max_iter=1000),
        "Arvore de Decisao (max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=SEMENTE),
        "Naive Bayes (Gaussiano)": GaussianNB(),
        "SVM (kernel=rbf)": SVC(kernel="rbf", random_state=SEMENTE),
        "Random Forest (100 arvores)": RandomForestClassifier(n_estimators=100, random_state=SEMENTE),
    }



# ============================================================ EXERCICIO 1 (1,5)
secao(1, "OS DADOS E O PISO (BASELINE)")

# TODO 1a: gere o dataset com gerar_dados_emprestimo(n=450, semente=SEMENTE)
#          e separe X (todas as colunas menos "aprovado") e y (a coluna "aprovado")
df = gerar_dados_emprestimo(n=450, semente=SEMENTE)
X = df.drop(columns="aprovado")
y = df["aprovado"]

# TODO 1b: divida em treino e teste com train_test_split(test_size=0.25,
#          random_state=SEMENTE, stratify=y) e imprima o tamanho de cada parte
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=SEMENTE, stratify=y
)
print(f"Treino: {len(X_treino)} linhas | Teste: {len(X_teste)} linhas")

# TODO 1c: treine um DummyClassifier(strategy="most_frequent") no treino e
#          imprima sua acuracia no teste (esse e o "piso" da comparacao)
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_treino, y_treino)
acc_base = accuracy_score(y_teste, baseline.predict(X_teste))
print(f"Baseline (classe mais comum): acuracia = {acc_base:.3f}")


# ============================================================ EXERCICIO 2 (2,5)
secao(2, "OS 6 MODELOS NO MESMO TREINO E TESTE")

# TODO 2a: para cada modelo de criar_modelos(), monte um Pipeline com dois
#          passos: ("prep", montar_preprocessador()) e ("modelo", modelo).
#          Treine no treino e guarde as previsoes no teste.
resultados = {}
predicoes = {}
modelos_treinados = {}
for nome, modelo in criar_modelos().items():
    pipe = Pipeline([
        ("prep", montar_preprocessador()),
        ("modelo", modelo),
    ])
    pipe.fit(X_treino, y_treino)
    predicoes[nome] = pipe.predict(X_teste)
    modelos_treinados[nome] = pipe

# TODO 2b: monte uma tabela (DataFrame) com uma linha por modelo e as
#          colunas acuracia, precisao, recall e f1 (classe positiva = 1,
#          aprovado). Imprima a tabela arredondada em 3 casas.
for nome, y_prev in predicoes.items():
    resultados[nome] = {
        "acuracia": accuracy_score(y_teste, y_prev),
        "precisao": precision_score(y_teste, y_prev, pos_label=1, zero_division=0),
        "recall": recall_score(y_teste, y_prev, pos_label=1, zero_division=0),
        "f1": f1_score(y_teste, y_prev, pos_label=1, zero_division=0),
    }
tabela = pd.DataFrame(resultados).T
print(tabela.round(3).to_string())

# TODO 2c: imprima o nome do modelo com o MAIOR F1
melhor_modelo = tabela["f1"].idxmax()
print(f"\nMelhor modelo pelo F1: {melhor_modelo}")


# ============================================================ EXERCICIO 3 (2,0)
secao(3, "MATRIZ DE CONFUSAO DO MELHOR MODELO")

# TODO 3a: calcule a matriz de confusao do modelo com maior F1 (Exercicio 2c)
#          e extraia VN, FP, FN, VP com  tn, fp, fn, tp = cm.ravel()
cm = confusion_matrix(y_teste, predicoes[melhor_modelo])
tn, fp, fn, tp = cm.ravel()
print("Matriz de confusao:")
print(cm)
print(f"VN={tn}, FP={fp}, FN={fn}, VP={tp}")

# TODO 3b: calcule NA MAO (so com tp, fp, fn) a precisao e o recall e
#          compare com os valores da tabela do Exercicio 2
precisao_mao = tp / (tp + fp) if tp + fp else 0.0
recall_mao = tp / (tp + fn) if tp + fn else 0.0
print(f"Precisao (manual): {precisao_mao:.3f} | tabela: {tabela.loc[melhor_modelo, 'precisao']:.3f}")
print(f"Recall (manual): {recall_mao:.3f} | tabela: {tabela.loc[melhor_modelo, 'recall']:.3f}")


# ============================================================ EXERCICIO 4 (2,0)
secao(4, "SEM PADRONIZAR: QUEM SOFRE?")

# TODO 4: repita o treino do k-NN, do SVM e da Arvore de Decisao usando
#         montar_preprocessador(escalar=False) (sem StandardScaler).
#         Imprima, para cada um, a acuracia COM escala, SEM escala e a
#         diferenca entre elas.
nomes_sem_escala = [
    "k-NN (k=5)",
    "SVM (kernel=rbf)",
    "Arvore de Decisao (max_depth=4)",
]
for nome in nomes_sem_escala:
    pipe = Pipeline([
        ("prep", montar_preprocessador(escalar=False)),
        ("modelo", criar_modelos()[nome]),
    ])
    pipe.fit(X_treino, y_treino)
    acc_sem = accuracy_score(y_teste, pipe.predict(X_teste))
    acc_com = resultados[nome]["acuracia"]
    print(f"{nome}: com escala={acc_com:.3f} | sem escala={acc_sem:.3f} | diferenca={acc_sem - acc_com:+.3f}")


# ============================================================ EXERCICIO 5 (2,0)
secao(5, "CINCO SORTEIOS, CINCO CAMPEOES?")

# TODO 5a: para semente em range(5), refaca a divisao treino/teste
#          (test_size=0.25, random_state=semente, stratify=y), treine os 6
#          modelos e guarde a acuracia de cada um no teste
nomes = list(criar_modelos().keys())
acuracias = {nome: [] for nome in nomes}
for semente in range(5):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=semente, stratify=y
    )
    for nome, modelo in criar_modelos().items():
        pipe = Pipeline([
            ("prep", montar_preprocessador()),
            ("modelo", modelo),
        ])
        pipe.fit(X_tr, y_tr)
        acuracias[nome].append(pipe.score(X_te, y_te))

tabela_sorteios = pd.DataFrame(acuracias)
print("Acuracias por sorteio:")
print(tabela_sorteios.round(3).to_string(index=False))

# TODO 5b: para cada sorteio, descubra quem foi o campeao (maior acuracia).
#          Em caso de EMPATE, todos os empatados contam como campeoes.
#          Dica: tabela.eq(tabela.max(axis=1), axis=0)
campeoes = tabela_sorteios.eq(tabela_sorteios.max(axis=1), axis=0)
print("\nCampeoes por sorteio:")
for sorteio, linha in campeoes.iterrows():
    nomes_campeoes = linha[linha].index.tolist()
    print(f"Sorteio {sorteio}: {', '.join(nomes_campeoes)}")

# TODO 5c: imprima quantas vezes (de 5) cada modelo foi campeao e a
#          amplitude (maximo - minimo) da acuracia de cada modelo
resumo_sorteios = pd.DataFrame({
    "campeao_em": campeoes.sum(),
    "amplitude": tabela_sorteios.max() - tabela_sorteios.min(),
})
print("\nResumo dos sorteios:")
print(resumo_sorteios.round(3).to_string())


# ============================================================ RELATORIO
"""
=======================================================================
RELATORIO 
=======================================================================

Ex.1 - Qual a acuracia do baseline e o que ela significa? Por que o
       parametro stratify=y foi usado na divisao?
     R: A acuracia foi 0,522, servindo como piso da comparacao. O stratify=y
         mantem a proporcao de aprovados e negados no treino e no teste.

Ex.2 - Qual modelo teve o maior F1? A diferenca dele para o segundo
       colocado e grande ou pequena? Com um teste de 113 exemplos, voce
       diria que essa diferenca e confiavel? Justifique.
     R: A Regressao Logistica teve o maior F1 (0,865), seguida pelo SVM (0,850).
         A diferenca e pequena e nao e confiavel com apenas 113 exemplos.

Ex.3 - No problema de emprestimo, qual erro e mais caro para o banco:
       aprovacao indevida (FP) ou negacao indevida (FN)? Com base nisso,
       o modelo escolhido no Exercicio 2c continua sendo o melhor para o
       banco? Que metrica voce olharia (precisao ou recall)?
     R: A aprovacao indevida (FP) costuma ser mais cara. Por isso, eu olharia
         principalmente a precisao; a Regressao Logistica pode nao ser a melhor
         escolha se o objetivo for minimizar FP.

Ex.4 - Qual modelo mais sofreu sem padronizacao e qual nao mudou nada?
       Explique o porque nos dois casos usando a ideia de DISTANCIA
       (k-NN, SVM) versus LIMIAR (arvore).
     R: O SVM foi o que mais sofreu, caindo de 0,850 para 0,593; o k-NN caiu
         para 0,761. Eles dependem de distancia, enquanto a arvore nao mudou
         (0,841), pois usa limiares.

Ex.5 - O campeao foi sempre o mesmo nos 5 sorteios? O que isso diz sobre
       confiar em uma unica divisao treino/teste para declarar "o melhor
       modelo"? O que voce faria para ter mais confianca na comparacao?
     R: Nao houve um campeao unico: SVM e Random Forest venceram 3 sorteios,
         com alguns empates. Isso mostra que uma divisao e instavel; eu usaria
         validacao cruzada estratificada para comparar os modelos.

=======================================================================
"""

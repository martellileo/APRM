"""
Aprendizado de Maquina - IFSP Presidente Epitacio
TRABALHO 1 - ATIVIDADE INTEGRADORA EM SALA (vale 1,5 ponto)

Cobre preparacao de dados e tipos de atributo (Aulas 4 e 5), os paradigmas
supervisionado e nao supervisionado (Aulas 3 e 5) e regressao linear
(Aulas 6 e 7.1). Sao 4 partes e 6 perguntas ao todo:
  Parte 1 - Supervisionado: k-NN e padronizacao ......... 0,5  (perguntas 1-2)
  Parte 2 - Nao supervisionado: agrupamento ............. 0,35 (pergunta 3)
  Parte 3 - Preparacao de dados e tipos de atributo ...... 0,2  (pergunta 4)
  Parte 4 - Regressao: previsao e residuo na mao ......... 0,45 (perguntas 5-6)

COMO USAR
  1. Preencha SEU PRONTUARIO na linha marcada abaixo (PRONTUARIO = "...").
  2. Rode:   python trabalho1_integrador.py
  3. Preencha a ficha_respostas_trabalho1 (impressa) com os numeros que
     aparecerem AQUI NA SUA TELA e com suas respostas de interpretacao.

ATENCAO: os numeros deste script sao UNICOS para o seu prontuario. Nao
adianta comparar com o resultado de um colega - vai ser diferente por
definicao. Atividade sem consulta a internet ou a colegas.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================ PREENCHA AQUI
PRONTUARIO = "PE3021742"
# ============================================================================

pd.set_option("display.width", 100)


def titulo(texto):
    print()
    print("=" * 74)
    print(texto)
    print("=" * 74)


def semente_do_prontuario(texto):
    digitos = "".join(c for c in str(texto) if c.isdigit())
    if len(digitos) < 3:
        raise ValueError(
            "PRONTUARIO invalido - preencha com o seu prontuario real "
            "(precisa ter pelo menos 3 digitos) antes de rodar o script."
        )
    return int(digitos) % 100000


SEMENTE = semente_do_prontuario(PRONTUARIO)

titulo(f"PRONTUARIO: {PRONTUARIO}   ->   SEMENTE PESSOAL: {SEMENTE}")
print("Confira que este numero de semente aparece igual na sua ficha de respostas.")
print("Se voce mudar o PRONTUARIO, TODOS os numeros abaixo mudam.")


# ---------------------------------------------------------------------------
# Dataset base: solicitacoes de emprestimo (mesmo gerador das Aulas 4 e 5)
# ---------------------------------------------------------------------------
def gerar_dados_emprestimo(n=450, semente=42):
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
        "idade": idade, "renda": renda, "tempo_emprego": tempo_emprego,
        "score_credito": score_credito, "cidade": cidade, "aprovado": aprovado,
    })
    mask_idade = rng.uniform(0, 1, n) < 0.08
    mask_renda = rng.uniform(0, 1, n) < 0.10
    mask_cidade = rng.uniform(0, 1, n) < 0.06
    df.loc[mask_idade, "idade"] = np.nan
    df.loc[mask_renda, "renda"] = np.nan
    df.loc[mask_cidade, "cidade"] = None
    return df


# gerador SEMPRE com a mesma semente=42 (a base inteira e igual pra turma
# toda) - o que muda por aluno e QUAL FATIA dessa base cada um recebe
_df = gerar_dados_emprestimo(n=450, semente=42)
completo = _df.dropna().reset_index(drop=True)
colunas_num = ["idade", "renda", "tempo_emprego", "score_credito"]

OFFSET = SEMENTE % (len(completo) - 60)
grupo8 = completo.loc[OFFSET:OFFSET + 7, colunas_num + ["aprovado"]].reset_index(drop=True)
grupo8.index = [f"S{i+1}" for i in range(len(grupo8))]

idx_candidato = OFFSET + 45
candidato = completo.loc[idx_candidato, colunas_num]
rotulo_verdadeiro_candidato = int(completo.loc[idx_candidato, "aprovado"])  # nao imprimir agora


# ============================================================ PARTE 1 (0,5)
titulo("PARTE 1 - SUPERVISIONADO: CLASSIFICAR UM NOVO CANDIDATO (k-NN)")

print("Seu grupo de 8 solicitantes JA APROVADOS/NEGADOS no passado (rotulo conhecido):")
print(grupo8)

print("\nNovo candidato, rotulo DESCONHECIDO (e o que vamos classificar):")
print(candidato.to_frame().T.to_string(index=False))

X8 = grupo8[colunas_num].values

# distancia SEM padronizar (bruta) - so para comparacao na Pergunta 2
dist_bruta = pairwise_distances(candidato.to_frame().T.values, X8)[0]

escalador = StandardScaler().fit(X8)
X8_pad = escalador.transform(X8)
cand_pad = escalador.transform(candidato.to_frame().T.values)

dist_candidato = pairwise_distances(cand_pad, X8_pad)[0]
tabela_dist = pd.DataFrame({
    "distancia_SEM_padronizar": np.round(dist_bruta, 1),
    "distancia_PADRONIZADA": np.round(dist_candidato, 3),
    "aprovado_no_passado": grupo8["aprovado"].values,
}, index=grupo8.index).sort_values("distancia_PADRONIZADA")

print("\nDistancias do candidato a cada um dos 8, sem padronizar e padronizadas")
print("(ordenadas pela distancia PADRONIZADA, do mais perto pro mais longe):")
print(tabela_dist)

vizinho_mais_perto_bruta = pd.Series(dist_bruta, index=grupo8.index).idxmin()
vizinho_mais_perto_pad = tabela_dist.index[0]
mudou = "MUDOU" if vizinho_mais_perto_bruta != vizinho_mais_perto_pad else "continua o mesmo"

print(f"""
PERGUNTA 1 (0,3 - para a ficha de respostas): usando k=3 (os 3 solicitantes
MAIS PROXIMOS do candidato pela distancia PADRONIZADA), qual e o voto da
maioria entre "aprovado" (1) e "nao aprovado" (0)? Qual classificacao voce
daria ao candidato? Anote os 3 mais proximos e o resultado da votacao.

PERGUNTA 2 (0,2 - para a ficha de respostas): compare as duas colunas de
distancia da tabela acima. Sem padronizar, o solicitante mais proximo do
candidato e {vizinho_mais_perto_bruta}; padronizando, o mais proximo
{mudou} ({vizinho_mais_perto_pad}). Por que isso acontece? Qual atributo
(idade, renda, tempo_emprego ou score_credito) e o principal responsavel
por essa diferenca, e por que usamos StandardScaler antes de calcular
distancia no k-NN?
""")

# ============================================================ PARTE 2 (0,35)
titulo("PARTE 2 - NAO SUPERVISIONADO: AGRUPANDO OS 8 SOLICITANTES POR SIMILARIDADE")

D8 = pairwise_distances(X8_pad)
print("Matriz de distancia padronizada ENTRE os 8 solicitantes do seu grupo")
print("(ignore a coluna 'aprovado' agora - a ideia e agrupar sem usar o rotulo):")
print(pd.DataFrame(np.round(D8, 2), index=grupo8.index, columns=grupo8.index))

print("""
PERGUNTA 3 (0,35 - para a ficha de respostas): observando a matriz acima
(ignore a coluna "aprovado"), separe os 8 solicitantes em DOIS grupos de 4,
juntando quem tem as MENORES distancias entre si. Nao existe uma unica
resposta certa. Escreva os dois grupos (ex.: Grupo A = {S1, S3, S6, S7}) e
cite pelo menos 2 distancias da matriz que justificam por que voce separou
assim.
""")

# ============================================================ PARTE 3 (0,2)
titulo("PARTE 3 - PREPARACAO DE DADOS E TIPOS DE ATRIBUTO")

print("""
O dataset completo de solicitacoes de emprestimo (Aulas 4 e 5) tem os
atributos: idade, renda, tempo_emprego e score_credito (numericos) e
cidade (SP, RJ, MG ou Outra). Usamos so os 4 numericos nas Partes 1 e 2
porque distancia (Euclidiana) precisa de valores numericos - cidade
entraria por outro caminho (one-hot + Jaccard, como na Aula 5.2).

PERGUNTA 4 (0,2 - para a ficha de respostas): classifique os atributos
CIDADE e SCORE_CREDITO quanto ao tipo (nominal, ordinal, intervalar ou
racional - vocabulario da Aula 5.1), justificando cada um em 1-2 linhas.
Dica: pense se existe uma ORDEM natural entre os valores, e se um "zero
absoluto" faz sentido (ou seja, se dividir dois valores do atributo e
comparar a razao tem significado pratico).
""")

# ============================================================ PARTE 4 (0,45)
titulo("PARTE 4 - REGRESSAO: AJUSTE, PREVISAO E RESIDUO")

aluguel = pd.read_csv("alugueis_tratado.csv")
X = aluguel[["area_m2"]].values
y = aluguel["valor_aluguel"].values
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.30, random_state=SEMENTE
)
modelo = LinearRegression().fit(X_treino, y_treino)
b0, b1 = modelo.intercept_, modelo.coef_[0]

y_previsto = modelo.predict(X_teste)
residuos = y_teste - y_previsto
mae = mean_absolute_error(y_teste, y_previsto)
rmse = mean_squared_error(y_teste, y_previsto) ** 0.5
r2 = r2_score(y_teste, y_previsto)

print(f"Sua reta ajustada: y_previsto = {b0:.2f} + {b1:.2f} * area_m2")
print(f"MAE  = R$ {mae:.2f}")
print(f"RMSE = R$ {rmse:.2f}")
print(f"R^2  = {r2:.4f}")

caso = X_teste[0, 0]
real_caso = y_teste[0]
previsto_caso = y_previsto[0]
residuo_caso = residuos[0]
print(f"""
Um imovel do SEU conjunto de teste, para conferencia manual:
  area_m2 = {caso:.1f}
  y_real  = R$ {real_caso:.2f}
""")

print(f"""
PERGUNTA 5 (0,25 - para a ficha de respostas): usando SUA formula acima
(y_previsto = {b0:.2f} + {b1:.2f} * area_m2) e uma calculadora, calcule
y_previsto para o imovel com area_m2 = {caso:.1f}. Depois calcule o
residuo (y_real - y_previsto) desse imovel usando y_real = R$ {real_caso:.2f}.
Confira: seu resultado deve bater com y_previsto = R$ {previsto_caso:.2f} e
residuo = R$ {residuo_caso:.2f} (arredondamentos de centavos sao normais).
Se nao bateu, revise a conta antes de preencher a ficha.

PERGUNTA 6 (0,2 - para a ficha de respostas): seu R^2 foi {r2:.4f}. Isso
significa que area_m2 explica aproximadamente {r2*100:.0f}% da variacao do
valor do aluguel no SEU conjunto de teste. Isso e parecido, maior ou menor
do que o R^2 = 0,7707 obtido na demonstracao da Aula 7.1 (com uma divisao
treino/teste diferente, semente=42 fixa para a turma toda)? Na sua
opiniao, o fato desse numero mudar um pouco de aluno para aluno significa
que a regressao linear "nao funciona direito", ou e um efeito esperado de
usar recortes diferentes dos mesmos 1000 anuncios? Justifique.
""")

titulo("FIM - confira se PRONTUARIO, semente e todos os numeros acima estao na ficha de respostas")

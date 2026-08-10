import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. Dados de exemplo com escalas muito diferentes
# ---------------------------------------------------------
dados = pd.DataFrame({
    "idade":  [18, 25, 32, 40, 55],
    "altura": [1.60, 1.68, 1.75, 1.82, 1.90],
    "renda":  [1500, 3000, 5000, 8000, 15000]
})

print("=" * 60)
print("DADOS ANTES DO STANDARDSCALER")
print("=" * 60)
print(dados.to_string(index=False))

print("\nMédia antes da padronização:")
print(dados.mean().round(2))

print("\nDesvio-padrão antes da padronização:")
# ddof=0 usa o mesmo cálculo adotado pelo StandardScaler
print(dados.std(ddof=0).round(2))


# ---------------------------------------------------------
# 2. Criando e aplicando o StandardScaler
# ---------------------------------------------------------
scaler = StandardScaler()

dados_padronizados_array = scaler.fit_transform(dados)

# fit_transform devolve um array NumPy.
# Aqui transformamos novamente em DataFrame para facilitar a leitura.
dados_padronizados = pd.DataFrame(
    dados_padronizados_array,
    columns=dados.columns
)


# ---------------------------------------------------------
# 3. Dados depois da padronização
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("DADOS DEPOIS DO STANDARDSCALER")
print("=" * 60)
print(dados_padronizados.round(3).to_string(index=False))

print("\nMédia depois da padronização:")
print(dados_padronizados.mean().round(3))

print("\nDesvio-padrão depois da padronização:")
print(dados_padronizados.std(ddof=0).round(3))


# ---------------------------------------------------------
# 4. Comparação lado a lado
# ---------------------------------------------------------
comparacao = pd.concat(
    [
        dados.add_suffix("_antes"),
        dados_padronizados.round(3).add_suffix("_depois")
    ],
    axis=1
)

print("\n" + "=" * 60)
print("COMPARAÇÃO: ANTES E DEPOIS")
print("=" * 60)
print(comparacao.to_string(index=False))
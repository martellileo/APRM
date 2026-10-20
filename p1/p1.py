"""Calculos para os itens 1c, 1d, 1e e 2c, 2d, 2e, 2f."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
SOJA_CSV = BASE_DIR / "talhoes_soja.csv"
ONIBUS_CSV = BASE_DIR / "linhas_onibus.csv"


def build_soja_model() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), ["chuva_mm", "adubo_kg_ha"]),
        ],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", LinearRegression()),
        ]
    )


def run_soja_analysis() -> None:
    df = pd.read_csv(SOJA_CSV)
    x = df[["area_ha", "chuva_mm", "ph_solo", "adubo_kg_ha"]]
    y = df["produtividade_sacas_ha"]

    print("Exercicio 1")
    print("c)")
    print("StandardScaler deve ser ajustado dentro de cada treino da validacao cruzada, nunca antes da divisao, para evitar vazamento de informacao.")
    print()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        build_soja_model(),
        x,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
    )

    metrics = pd.DataFrame(
        {
            "MAE": -scores["test_mae"],
            "RMSE": -scores["test_rmse"],
            "R2": scores["test_r2"],
        },
        index=[f"Particao {i}" for i in range(1, 6)],
    )
    metrics = metrics.round(4)

    print("d)")
    print(metrics.to_string())
    print(f"Media: MAE={metrics['MAE'].mean():.4f}, RMSE={metrics['RMSE'].mean():.4f}, R2={metrics['R2'].mean():.4f}")
    print(f"Desvio-padrao: MAE={metrics['MAE'].std(ddof=1):.4f}, RMSE={metrics['RMSE'].std(ddof=1):.4f}, R2={metrics['R2'].std(ddof=1):.4f}")
    print()

    final_model = build_soja_model()
    final_model.fit(x, y)
    novos_talhoes = pd.DataFrame(
        [
            {"area_ha": 100, "chuva_mm": 1000, "ph_solo": 5.5, "adubo_kg_ha": 200},
            {"area_ha": 300, "chuva_mm": 1050, "ph_solo": 6.0, "adubo_kg_ha": 220},
            {"area_ha": 500, "chuva_mm": 1100, "ph_solo": 6.5, "adubo_kg_ha": 240},
        ],
        index=["A", "B", "C"],
    )
    previsoes = pd.Series(final_model.predict(novos_talhoes), index=novos_talhoes.index).round(2)
    print("e)")
    print(previsoes.to_string())
    print()


def evaluate_kmeans(x_scaled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(x_scaled)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(x_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(x_scaled, labels),
                "davies_bouldin": davies_bouldin_score(x_scaled, labels),
            }
        )
    return pd.DataFrame(rows)


def choose_best_k(results: pd.DataFrame) -> int:
    ordered = results.sort_values(
        by=["silhouette", "calinski_harabasz", "davies_bouldin"],
        ascending=[False, False, True],
    )
    return int(ordered.iloc[0]["k"])


def run_onibus_analysis() -> None:
    df = pd.read_csv(ONIBUS_CSV)
    x = df[["passageiros_dia", "atraso_medio_min"]]

    print("Exercicio 2")
    print("c)")
    print(
        "As variaveis devem ser padronizadas porque passageiros_dia esta na casa dos milhares, "
        "enquanto atraso_medio_min esta na casa de poucos minutos; sem padronizacao, a variavel de maior escala dominaria as distancias."
    )
    print()

    scaler = StandardScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    results = evaluate_kmeans(x_scaled).round(4)
    best_k = choose_best_k(results)

    print("d)")
    print(results.to_string(index=False))
    print(f"k escolhido: {best_k}")
    print()

    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_model.fit_predict(x_scaled)
    df = df.copy()
    df["grupo"] = labels

    group_summary = (
        df.groupby("grupo")[["passageiros_dia", "atraso_medio_min"]]
        .agg(["count", "mean"])
        .round(2)
    )
    group_summary.columns = [f"{col[0]}_{col[1]}" for col in group_summary.columns]

    centers_original = pd.DataFrame(
        scaler.inverse_transform(final_model.cluster_centers_),
        columns=["passageiros_dia", "atraso_medio_min"],
    ).round(2)

    print("e)")
    print(group_summary.to_string())
    print()
    print(centers_original.to_string(index=False))
    print()

    novas_linhas = pd.DataFrame(
        {
            "passageiros_dia": [700, 1100, 950],
            "atraso_medio_min": [2.5, 4.0, 3.0],
        },
        index=["X", "Y", "Z"],
    )
    novas_linhas_scaled = pd.DataFrame(
        scaler.transform(novas_linhas),
        columns=novas_linhas.columns,
        index=novas_linhas.index,
    )
    novas_labels = final_model.predict(novas_linhas_scaled)

    print("f)")
    for linha, grupo in zip(novas_linhas.index, novas_labels, strict=True):
        print(f"Linha {linha}: grupo {int(grupo)}")


def main() -> None:
    run_soja_analysis()
    run_onibus_analysis()


if __name__ == "__main__":
    main()
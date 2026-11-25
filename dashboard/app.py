"""Dashboard: segmentación de videojuegos con K-Means.

Ejecutar desde la carpeta raíz del proyecto:
    streamlit run dashboard/app.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "data" / "processed" / "vgsales_clusters.csv"
DIR_TAB = RAIZ / "outputs" / "tables"

COLORES = {
    "Éxitos occidentales": "#E4572E",
    "Nicho norteamericano": "#2E86AB",
    "Nicho europeo": "#76B041",
    "Nicho japonés": "#F2A541",
}
COLORES_REGION = ["#2E86AB", "#76B041", "#F2A541", "#999999"]
DESCRIPCIONES = {
    "Éxitos occidentales": "Juegos de ventas altas que se reparten sobre todo entre Norteamérica y Europa. "
    "Son poco más de un tercio de los juegos y concentran la mayor parte de las unidades vendidas.",
    "Nicho norteamericano": "Juegos de ventas modestas que se venden casi por completo en Norteamérica.",
    "Nicho europeo": "Juegos de ventas bajas con fuerte peso en Europa; destacan los títulos de PC.",
    "Nicho japonés": "Juegos que se venden casi exclusivamente en Japón, con mayor peso de los géneros de rol y aventura.",
}

st.set_page_config(page_title="Segmentación de videojuegos con K-Means", page_icon="🎮", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data
def cargar_datos():
    df = pd.read_csv(RUTA_DATOS)
    with open(DIR_TAB / "resumen.json", encoding="utf-8") as f:
        resumen = json.load(f)
    perfil = pd.read_csv(DIR_TAB / "perfil_clusters.csv", index_col=0)
    centroides = pd.read_csv(DIR_TAB / "centroides.csv", index_col=0)
    metricas_k = pd.read_csv(DIR_TAB / "metricas_k.csv")
    return df, resumen, perfil, centroides, metricas_k


def mostrar(fig):
    st.pyplot(fig)
    plt.close(fig)


def color_de(nombre):
    return COLORES.get(nombre, "#999999")


try:
    df, resumen, perfil, centroides, metricas_k = cargar_datos()
except FileNotFoundError:
    st.error(
        "No se encontraron los archivos procesados. Ejecuta primero `python src/analisis_kmeans.py` "
        "desde la carpeta raíz del proyecto."
    )
    st.stop()

orden_clusters = list(perfil.index)

# ----------------------------------------------------------------------------
# Barra lateral: navegación y filtros
# ----------------------------------------------------------------------------
st.sidebar.title("🎮 Navegación")
pagina = st.sidebar.radio("Página", ["1. Análisis general", "2. Segmentación K-Means"])

st.sidebar.header("Filtros")
generos_op = sorted(df["Genre"].unique())
plataformas_op = sorted(df["Platform"].unique())
editores_op = ["Todos"] + sorted(df["Publisher"].unique())
anio_min, anio_max = int(df["Year"].min()), int(df["Year"].max())

generos = st.sidebar.multiselect("Género", generos_op, default=generos_op)
plataformas = st.sidebar.multiselect("Plataforma", plataformas_op, default=plataformas_op)
rango_anios = st.sidebar.slider("Año de lanzamiento", anio_min, anio_max, (anio_min, anio_max))
editor = st.sidebar.selectbox("Editor (por ejemplo, Nintendo)", editores_op, index=0)
if pagina.startswith("2"):
    clusters_sel = st.sidebar.multiselect("Clúster", orden_clusters, default=orden_clusters)
else:
    clusters_sel = orden_clusters

mascara = (
    df["Genre"].isin(generos)
    & df["Platform"].isin(plataformas)
    & df["Year"].between(rango_anios[0], rango_anios[1])
    & df["Cluster_Nombre"].isin(clusters_sel)
)
if editor != "Todos":
    mascara &= df["Publisher"] == editor
datos = df[mascara]

st.title("🎮 Segmentación de videojuegos según sus ventas regionales (K-Means)")
st.caption(
    f"Dataset: Video Game Sales · {len(df):,} juegos · modelo K-Means con K = {resumen['k_final']} "
    f"sobre ventas (log10) y proporción de ventas en Norteamérica, Europa y Japón."
)

if datos.empty:
    st.warning("Los filtros seleccionados no devuelven ningún juego. Ajusta los filtros de la barra lateral.")
    st.stop()

# ----------------------------------------------------------------------------
# Página 1: análisis general
# ----------------------------------------------------------------------------
if pagina.startswith("1"):
    st.header("Análisis general del dataset")

    ventas_genero = datos.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Juegos analizados", f"{len(datos):,}")
    k2.metric("Ventas globales (M de unidades)", f"{datos['Global_Sales'].sum():,.1f}")
    k3.metric("Ventas promedio por juego (M)", f"{datos['Global_Sales'].mean():.2f}")
    k4.metric("Género con más ventas", ventas_genero.index[0])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Ventas globales por año de lanzamiento")
        por_anio = datos.groupby("Year")["Global_Sales"].sum()
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(por_anio.index, por_anio.values, marker="o", color="#2E86AB")
        ax.set_xlabel("Año")
        ax.set_ylabel("Millones de unidades")
        mostrar(fig)
    with c2:
        st.subheader("Ventas globales por género")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ventas_genero.sort_values().plot.barh(ax=ax, color="#76B041")
        ax.set_xlabel("Millones de unidades")
        ax.set_ylabel("")
        mostrar(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Ventas por región")
        regiones = datos[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum()
        regiones.index = ["Norteamérica", "Europa", "Japón", "Otras"]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.bar(regiones.index, regiones.values, color=COLORES_REGION)
        ax.set_ylabel("Millones de unidades")
        mostrar(fig)
    with c4:
        st.subheader("Top 10 plataformas por ventas")
        top_plat = datos.groupby("Platform")["Global_Sales"].sum().sort_values().tail(10)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        top_plat.plot.barh(ax=ax, color="#F2A541")
        ax.set_xlabel("Millones de unidades")
        ax.set_ylabel("")
        mostrar(fig)

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Distribución de ventas por juego (log10)")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.histplot(datos["log_ventas"], bins=40, color="#E4572E", ax=ax)
        ax.set_xlabel("log10(ventas globales en millones)")
        ax.set_ylabel("Juegos")
        mostrar(fig)
    with c6:
        st.subheader("Top 10 editores por ventas")
        top_ed = datos.groupby("Publisher")["Global_Sales"].sum().sort_values().tail(10)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        top_ed.plot.barh(ax=ax, color="#2E86AB")
        ax.set_xlabel("Millones de unidades")
        ax.set_ylabel("")
        mostrar(fig)

    st.subheader("Juegos con más ventas (según los filtros)")
    st.dataframe(
        datos.sort_values("Global_Sales", ascending=False)[
            ["Name", "Platform", "Year", "Genre", "Publisher", "Global_Sales", "Cluster_Nombre"]
        ].head(20)
    )
    st.download_button(
        "Descargar datos filtrados (CSV)",
        data=datos.to_csv(index=False).encode("utf-8"),
        file_name="videojuegos_filtrados.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------------
# Página 2: segmentación K-Means
# ----------------------------------------------------------------------------
else:
    st.header("Segmentación mediante K-Means")

    ventas_cluster = datos.groupby("Cluster_Nombre")["Global_Sales"].sum().sort_values(ascending=False)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Juegos analizados", f"{len(datos):,}")
    k2.metric("Clústeres presentes", f"{datos['Cluster_Nombre'].nunique()} de {resumen['k_final']}")
    k3.metric("Clúster con más ventas", ventas_cluster.index[0])
    k4.metric("Coeficiente de silueta del modelo", f"{resumen['silueta']:.2f}")

    presentes = [n for n in orden_clusters if n in set(datos["Cluster_Nombre"])]
    paleta = {n: color_de(n) for n in presentes}

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribución de juegos por clúster")
        conteo = datos["Cluster_Nombre"].value_counts().reindex(presentes)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(conteo.index, conteo.values, color=[paleta[n] for n in conteo.index])
        for i, v in enumerate(conteo.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom")
        ax.set_ylabel("Juegos")
        plt.setp(ax.get_xticklabels(), rotation=15)
        mostrar(fig)
    with c2:
        st.subheader("Agrupamientos (proyección PCA)")
        muestra = datos if len(datos) <= 8000 else datos.sample(8000, random_state=42)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        sns.scatterplot(
            data=muestra, x="PC1", y="PC2", hue="Cluster_Nombre", hue_order=presentes,
            palette=paleta, s=12, alpha=0.55, linewidth=0, ax=ax,
        )
        var = resumen["varianza_pca"]
        ax.set_xlabel(f"Componente 1 ({var[0]:.0%})")
        ax.set_ylabel(f"Componente 2 ({var[1]:.0%})")
        ax.legend(title="Clúster", markerscale=2, fontsize=8)
        mostrar(fig)
        if len(datos) > 8000:
            st.caption("Se muestran 8,000 puntos aleatorios para agilizar la gráfica.")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Composición regional promedio por clúster")
        comp = datos.groupby("Cluster_Nombre")[["pct_NA", "pct_EU", "pct_JP"]].mean().reindex(presentes)
        comp["Otras"] = (1 - comp.sum(axis=1)).clip(lower=0)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        comp.plot.bar(stacked=True, ax=ax, color=COLORES_REGION)
        ax.set_ylabel("Proporción de ventas")
        ax.set_xlabel("")
        ax.legend(["Norteamérica", "Europa", "Japón", "Otras"], fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.setp(ax.get_xticklabels(), rotation=15)
        mostrar(fig)
    with c4:
        st.subheader("Ventas globales por clúster (escala log)")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        sns.boxplot(
            data=datos, x="Cluster_Nombre", y="Global_Sales", order=presentes, hue="Cluster_Nombre",
            palette=paleta, legend=False, fliersize=2, ax=ax,
        )
        ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel("Millones de unidades")
        plt.setp(ax.get_xticklabels(), rotation=15)
        mostrar(fig)

    st.subheader("Estadísticas descriptivas por clúster (según los filtros)")
    estad = datos.groupby("Cluster_Nombre").agg(
        Juegos=("Name", "size"),
        Ventas_totales_M=("Global_Sales", "sum"),
        Ventas_media_M=("Global_Sales", "mean"),
        Ventas_mediana_M=("Global_Sales", "median"),
        Ventas_desv_M=("Global_Sales", "std"),
        Ventas_max_M=("Global_Sales", "max"),
        pct_NA=("pct_NA", "mean"),
        pct_EU=("pct_EU", "mean"),
        pct_JP=("pct_JP", "mean"),
        Anio_medio=("Year", "mean"),
    ).reindex(presentes).round(3)
    st.dataframe(estad)

    with st.expander("Centroides del modelo (calculados con todo el dataset)"):
        st.caption(
            "Valores en unidades originales: log_ventas es log10 de las ventas globales en millones y "
            "ventas_tipicas_M = 10^log_ventas."
        )
        columnas_cent = ["Cluster_Nombre", "log_ventas_orig", "ventas_tipicas_M_orig", "pct_NA_orig", "pct_EU_orig", "pct_JP_orig"]
        st.dataframe(centroides[columnas_cent].round(3))

    st.subheader("Descripción de los grupos identificados")
    st.caption("Cifras del modelo completo, sin filtros.")
    for nombre in orden_clusters:
        fila = perfil.loc[nombre]
        st.markdown(
            f"**{nombre}** — {int(fila['juegos']):,} juegos ({fila['%_juegos']:.1f}% del total) que generan "
            f"{fila['%_ventas']:.1f}% de las ventas. Mediana de {fila['ventas_mediana_M']:.2f} M de unidades por juego; "
            f"ventas promedio: Norteamérica {fila['pct_NA']:.0%}, Europa {fila['pct_EU']:.0%}, Japón {fila['pct_JP']:.0%}.  \n"
            f"{DESCRIPCIONES.get(nombre, '')}  \n"
            f"Géneros frecuentes: {fila['generos_top']}. Plataformas: {fila['plataformas_top']}."
        )

    st.subheader("Principales hallazgos")
    for h in resumen["hallazgos"]:
        st.markdown(f"- {h}")
    st.subheader("Recomendaciones")
    for r in resumen["recomendaciones"]:
        st.markdown(f"- {r}")
    with st.expander("Limitaciones del modelo"):
        for lim in resumen["limitaciones"]:
            st.markdown(f"- {lim}")

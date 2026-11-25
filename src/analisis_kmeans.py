# %% [markdown]
# # Segmentación de videojuegos con K-Means
# Proyecto integrador – Minería de datos (tercera unidad)
#
# Este archivo funciona de dos formas:
# - como script: `python src/analisis_kmeans.py` (genera todas las tablas, gráficas y el dataset procesado)
# - como notebook: se convirtió a `notebooks/analisis_kmeans.ipynb` con las mismas celdas.

# %% [markdown]
# ## 0. Configuración

# %%
import json
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

EN_NOTEBOOK = "ipykernel" in sys.modules
if not EN_NOTEBOOK:
    matplotlib.use("Agg")  # sin ventanas: solo guarda las figuras

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

SEMILLA = 42
REGIONES = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
ANIO_REFERENCIA = 2016  # último año con cobertura completa en el dataset


def encontrar_raiz():
    """Busca la carpeta del proyecto (la que contiene data/raw/vgsales.csv)."""
    inicio = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for carpeta in [inicio, *inicio.parents]:
        if (carpeta / "data" / "raw" / "vgsales.csv").exists():
            return carpeta
    raise FileNotFoundError("No se encontró data/raw/vgsales.csv. Ejecuta desde la carpeta del proyecto.")


RAIZ = encontrar_raiz()
RUTA_RAW = RAIZ / "data" / "raw" / "vgsales.csv"
RUTA_PROC = RAIZ / "data" / "processed" / "vgsales_clusters.csv"
DIR_FIG = RAIZ / "outputs" / "figures"
DIR_TAB = RAIZ / "outputs" / "tables"
for carpeta in (RUTA_PROC.parent, DIR_FIG, DIR_TAB):
    carpeta.mkdir(parents=True, exist_ok=True)


class Tee:
    """Duplica lo que se imprime en pantalla hacia un archivo de reporte."""

    def __init__(self, *flujos):
        self.flujos = flujos

    def write(self, texto):
        for f in self.flujos:
            f.write(texto)

    def flush(self):
        for f in self.flujos:
            f.flush()


_archivo_reporte = None
if not EN_NOTEBOOK:
    _archivo_reporte = open(RAIZ / "outputs" / "reporte_consola.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, _archivo_reporte)


def guardar_fig(fig, nombre):
    fig.savefig(DIR_FIG / nombre, dpi=150, bbox_inches="tight")
    if EN_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)


def titulo(texto):
    print("\n" + "=" * 80)
    print(texto)
    print("=" * 80)


# %% [markdown]
# ## Parte 2. Obtención y exploración del dataset
# **Fuente:** *Video Game Sales* (Kaggle, `vgsales.csv`), ventas en millones de unidades por región,
# obtenidas de VGChartz. Cada registro es un juego en una plataforma.

# %%
titulo("PARTE 2. EXPLORACIÓN DEL DATASET")
df_raw = pd.read_csv(RUTA_RAW)

print("Primeros 10 registros:")
print(df_raw.head(10).to_string())

print(f"\nDimensiones: {df_raw.shape[0]} filas x {df_raw.shape[1]} columnas")
print("\nColumnas y tipos de datos:")
print(df_raw.dtypes.to_string())

print("\nEstadísticas descriptivas (numéricas):")
print(df_raw.describe().round(3).to_string())

print("\nEstadísticas descriptivas (categóricas):")
print(df_raw[["Name", "Platform", "Genre", "Publisher"]].describe().to_string())

print("\nValores nulos por columna:")
print(df_raw.isna().sum().to_string())
print(f"\nRegistros duplicados exactos: {df_raw.duplicated().sum()}")

# %% [markdown]
# ### Gráficas exploratorias

# %%
# Gráfica 1: distribución de las ventas globales (original vs. logarítmica)
fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_raw["Global_Sales"].clip(upper=5), bins=50, ax=ejes[0], color="#2E86AB")
ejes[0].set_title("Ventas globales (recortadas a 5 M para visualizar)")
ejes[0].set_xlabel("Millones de unidades")
ejes[0].set_ylabel("Juegos")
sns.histplot(np.log10(df_raw["Global_Sales"]), bins=50, ax=ejes[1], color="#E4572E")
ejes[1].set_title("Ventas globales en escala logarítmica (log10)")
ejes[1].set_xlabel("log10(millones de unidades)")
ejes[1].set_ylabel("Juegos")
fig.suptitle("Las ventas están muy sesgadas: pocos juegos concentran gran parte de las unidades", y=1.03)
guardar_fig(fig, "01_distribucion_ventas.png")

# Gráfica 2: ventas globales por año de lanzamiento
ventas_anio = df_raw.dropna(subset=["Year"]).groupby("Year")["Global_Sales"].sum()
ventas_anio = ventas_anio[ventas_anio.index <= ANIO_REFERENCIA]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ventas_anio.index, ventas_anio.values, marker="o", color="#2E86AB")
ax.set_title("Ventas globales por año de lanzamiento (hasta 2016)")
ax.set_xlabel("Año")
ax.set_ylabel("Millones de unidades")
guardar_fig(fig, "02_ventas_por_anio.png")

# Gráfica 3: ventas globales por género
ventas_genero = df_raw.groupby("Genre")["Global_Sales"].sum().sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
ventas_genero.plot.barh(ax=ax, color="#76B041")
ax.set_title("Ventas globales por género")
ax.set_xlabel("Millones de unidades")
ax.set_ylabel("")
guardar_fig(fig, "03_ventas_por_genero.png")

# Gráfica 4: correlación entre ventas (muestra que Global_Sales es la suma de las regiones)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df_raw[REGIONES + ["Global_Sales"]].corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
ax.set_title("Correlación entre ventas por región")
guardar_fig(fig, "04_correlacion_ventas.png")

# Gráfica 5: composición regional de las ventas en las plataformas principales
top_plat = df_raw.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).head(10).index
comp = df_raw[df_raw["Platform"].isin(top_plat)].groupby("Platform")[REGIONES].sum()
comp = comp.div(comp.sum(axis=1), axis=0).loc[top_plat]
fig, ax = plt.subplots(figsize=(10, 4.5))
comp.plot.bar(stacked=True, ax=ax, color=["#2E86AB", "#76B041", "#F2A541", "#999999"])
ax.set_title("Composición regional de las ventas por plataforma (top 10)")
ax.set_ylabel("Proporción de ventas")
ax.set_xlabel("")
ax.legend(title="Región", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xticks(rotation=0)
guardar_fig(fig, "05_composicion_regional_plataformas.png")

print("Gráficas exploratorias guardadas en outputs/figures (01 a 05).")

# %% [markdown]
# ### Selección de variables para el agrupamiento
# - **Volumen de ventas** (`log_ventas`): indica qué tan grande fue el éxito del juego.
# - **Proporción de ventas en Norteamérica, Europa y Japón** (`pct_NA`, `pct_EU`, `pct_JP`): indica *dónde* vende el juego.
#
# No se usa `Global_Sales` junto con las ventas regionales porque es su suma (correlación casi perfecta, información repetida).
# Tampoco se usa `Other_Sales` como proporción, porque las cuatro proporciones suman 1 y la cuarta es redundante.
# Género, plataforma y editor se dejan fuera del modelo y se usan después para **interpretar** los grupos.

# %% [markdown]
# ## Parte 3. Limpieza y preparación de datos

# %%
titulo("PARTE 3. LIMPIEZA Y PREPARACIÓN")
df = df_raw.copy()
registro = {}  # bitácora de la limpieza
registro["registros_originales"] = len(df)

# 3.1 Inconsistencias de formato: espacios sobrantes en texto
for col in ["Name", "Platform", "Genre", "Publisher"]:
    df[col] = df[col].str.strip()

# 3.2 Valores nulos
registro["year_nulos"] = int(df["Year"].isna().sum())
registro["publisher_nulos"] = int(df["Publisher"].isna().sum())
# Año: se imputa con la mediana de su plataforma (cada consola tiene una época de lanzamientos)
mediana_plataforma = df.groupby("Platform")["Year"].transform("median")
df["Year"] = df["Year"].fillna(mediana_plataforma).fillna(df["Year"].median()).round().astype(int)
# Editor: categoría explícita en lugar de eliminar registros
df["Publisher"] = df["Publisher"].fillna("Unknown")

# 3.3 Duplicados
registro["duplicados_exactos"] = int(df.duplicated().sum())
mascara_dup = df.duplicated(["Name", "Platform", "Year"], keep=False)
print("Registros con mismo Name + Platform + Year (duplicados lógicos):")
print(df.loc[mascara_dup, ["Rank", "Name", "Platform", "Year", "Publisher", "Global_Sales"]].to_string())
antes = len(df)
df = df.sort_values("Global_Sales", ascending=False).drop_duplicates(["Name", "Platform", "Year"], keep="first")
df = df.sort_values("Rank").reset_index(drop=True)
registro["duplicados_logicos_eliminados"] = antes - len(df)

# 3.4 Consistencia: Global_Sales debe ser la suma de las regiones (con redondeo de 0.01 por región)
dif = (df[REGIONES].sum(axis=1) - df["Global_Sales"]).abs()
registro["max_diferencia_suma_regional"] = round(float(dif.max()), 3)
print(f"\nDiferencia máxima entre Global_Sales y la suma regional: {dif.max():.3f} (por redondeo)")

# Registros sin ventas regionales no permiten calcular proporciones
sin_ventas_reg = df[REGIONES].sum(axis=1) == 0
registro["sin_ventas_regionales_eliminados"] = int(sin_ventas_reg.sum())
df = df.loc[~sin_ventas_reg].reset_index(drop=True)

# 3.5 Tipos de datos
df["Year"] = df["Year"].astype(int)
for col in ["Platform", "Genre"]:
    df[col] = df[col].astype("category")

# 3.6 Valores atípicos (regla del rango intercuartílico)
q1, q3 = df["Global_Sales"].quantile([0.25, 0.75])
limite = q3 + 1.5 * (q3 - q1)
registro["atipicos_iqr_ventas"] = int((df["Global_Sales"] > limite).sum())
log_ventas_tmp = np.log10(df["Global_Sales"])
lq1, lq3 = log_ventas_tmp.quantile([0.25, 0.75])
lim_inf, lim_sup = lq1 - 1.5 * (lq3 - lq1), lq3 + 1.5 * (lq3 - lq1)
registro["atipicos_iqr_log"] = int(((log_ventas_tmp < lim_inf) | (log_ventas_tmp > lim_sup)).sum())
print(
    f"\nAtípicos por IQR en Global_Sales: {registro['atipicos_iqr_ventas']} "
    f"({registro['atipicos_iqr_ventas'] / len(df):.1%}) | tras log10: {registro['atipicos_iqr_log']}"
)

fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df["Global_Sales"], ax=ejes[0], color="#2E86AB")
ejes[0].set_title("Ventas globales (escala original)")
ejes[0].set_ylabel("Millones de unidades")
sns.boxplot(y=log_ventas_tmp, ax=ejes[1], color="#E4572E")
ejes[1].set_title("Ventas globales (log10)")
ejes[1].set_ylabel("log10(millones)")
guardar_fig(fig, "06_atipicos_ventas.png")
print(
    "Decisión: NO se eliminan los atípicos. Son éxitos reales (Wii Sports, Mario Kart...) y justamente "
    "son el segmento que interesa identificar. En su lugar se aplica una transformación logarítmica."
)

# 3.7 Transformaciones: ventas en log10 y proporciones regionales
df["Ventas_Regionales"] = df[REGIONES].sum(axis=1)
df["log_ventas"] = np.log10(df["Global_Sales"])
df["pct_NA"] = df["NA_Sales"] / df["Ventas_Regionales"]
df["pct_EU"] = df["EU_Sales"] / df["Ventas_Regionales"]
df["pct_JP"] = df["JP_Sales"] / df["Ventas_Regionales"]

# 3.8 Escalamiento con StandardScaler
FEATURES = ["log_ventas", "pct_NA", "pct_EU", "pct_JP"]
escalador = StandardScaler()
X = escalador.fit_transform(df[FEATURES])

registro["registros_finales"] = len(df)
registro["nulos_finales"] = int(df[["Year", "Publisher"]].isna().sum().sum())
print("\nResumen de la limpieza:")
for clave, valor in registro.items():
    print(f"  {clave}: {valor}")

print("\nVariables del modelo antes del escalamiento:")
print(df[FEATURES].describe().round(3).to_string())
print("\nVariables del modelo después de StandardScaler (media ~0, desviación ~1):")
print(pd.DataFrame(X, columns=FEATURES).describe().round(3).loc[["mean", "std", "min", "max"]].to_string())

# %% [markdown]
# **¿Por qué estas transformaciones?**
# - El logaritmo evita que unos cuantos superventas dominen la distancia euclidiana que usa K-Means.
# - Las proporciones separan *cuánto* vende un juego de *dónde* lo vende.
# - `StandardScaler` pone todas las variables en la misma escala; sin él, la variable con mayor rango pesaría más.

# %% [markdown]
# ## Parte 4. Implementación de K-Means

# %%
titulo("PARTE 4. K-MEANS")
print(f"Variables usadas en el entrenamiento ({len(FEATURES)}): {FEATURES}")

# 4.1 Evaluación de K = 2 ... 10
filas = []
for k in range(2, 11):
    modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
    etiquetas = modelo.fit_predict(X)
    filas.append(
        {
            "K": k,
            "inercia": modelo.inertia_,
            "silueta": silhouette_score(X, etiquetas, sample_size=8000, random_state=SEMILLA),
            "davies_bouldin": davies_bouldin_score(X, etiquetas),
            "calinski_harabasz": calinski_harabasz_score(X, etiquetas),
        }
    )
metricas_k = pd.DataFrame(filas)
metricas_k["caida_inercia_pct"] = (-metricas_k["inercia"].pct_change() * 100).round(1)
print("\nMétricas por número de clústeres:")
print(metricas_k.round(3).to_string(index=False))
metricas_k.round(4).to_csv(DIR_TAB / "metricas_k.csv", index=False)

K_FINAL = 4
fila_k = metricas_k.set_index("K")
mejor_sil_k = int(fila_k["silueta"].idxmax())
print(
    f"\nJustificación de K = {K_FINAL}:"
    f"\n- La silueta es máxima en K = {mejor_sil_k} ({fila_k.loc[mejor_sil_k, 'silueta']:.3f}), "
    f"pero K = {K_FINAL} conserva un valor aceptable ({fila_k.loc[K_FINAL, 'silueta']:.3f})."
    f"\n- Método del codo: pasar de K = 3 a K = {K_FINAL} reduce la inercia {fila_k.loc[K_FINAL, 'caida_inercia_pct']:.1f}%, "
    f"mientras que pasar a K = {K_FINAL + 1} solo la reduce {fila_k.loc[K_FINAL + 1, 'caida_inercia_pct']:.1f}%: el codo está en K = {K_FINAL}."
    f"\n- K = 2 solo separa los juegos de mercado japonés del resto, lo que aporta poco para tomar decisiones."
    f"\n- Con K = 3 los juegos de ventas altas quedan mezclados con los de nicho norteamericano en un solo grupo de unos 10,900 juegos; "
    f"K = {K_FINAL} los separa en un grupo propio."
)

fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
for ax, columna, etiqueta in zip(
    ejes,
    ["inercia", "silueta", "davies_bouldin"],
    ["Inercia (método del codo)", "Coeficiente de silueta (mayor es mejor)", "Davies-Bouldin (menor es mejor)"],
):
    ax.plot(metricas_k["K"], metricas_k[columna], marker="o", color="#2E86AB")
    ax.axvline(K_FINAL, color="#E4572E", linestyle="--", label=f"K = {K_FINAL}")
    ax.set_title(etiqueta)
    ax.set_xlabel("Número de clústeres (K)")
    ax.legend()
guardar_fig(fig, "07_seleccion_k.png")

# 4.2 Modelo final
km_final = KMeans(n_clusters=K_FINAL, n_init=10, random_state=SEMILLA)
df["Cluster"] = km_final.fit_predict(X)

centroides_esc = pd.DataFrame(km_final.cluster_centers_, columns=FEATURES)
centroides_orig = pd.DataFrame(escalador.inverse_transform(km_final.cluster_centers_), columns=FEATURES)
centroides_orig["ventas_tipicas_M"] = 10 ** centroides_orig["log_ventas"]

print("\nCentroides (variables escaladas):")
print(centroides_esc.round(3).to_string())
print("\nCentroides (unidades originales; ventas_tipicas_M = 10^log_ventas):")
print(centroides_orig.round(3).to_string())
print("\nRegistros por clúster:")
print(df["Cluster"].value_counts().sort_index().to_string())

# %% [markdown]
# ### Nombres de los clústeres
# Se asignan a partir de los centroides: el grupo con más ventas (repartidas sobre todo entre Norteamérica y Europa) es "Éxitos occidentales" y el resto se nombra
# por la región que domina sus ventas.

# %%
NOMBRE_REGION = {"pct_NA": "Nicho norteamericano", "pct_EU": "Nicho europeo", "pct_JP": "Nicho japonés"}


def nombrar_clusters(centroides):
    nombres = {}
    lider = centroides["log_ventas"].idxmax()
    nombres[lider] = "Éxitos occidentales"
    for cid, fila in centroides.drop(index=lider).iterrows():
        base = NOMBRE_REGION[fila[["pct_NA", "pct_EU", "pct_JP"]].idxmax()]
        nombre, n = base, 2
        while nombre in nombres.values():
            nombre, n = f"{base} {n}", n + 1
        nombres[cid] = nombre
    return nombres


NOMBRES = nombrar_clusters(centroides_orig)
COLORES = {
    "Éxitos occidentales": "#E4572E",
    "Nicho norteamericano": "#2E86AB",
    "Nicho europeo": "#76B041",
    "Nicho japonés": "#F2A541",
}
df["Cluster_Nombre"] = df["Cluster"].map(NOMBRES)
orden_nombres = [NOMBRES[i] for i in sorted(NOMBRES)]
paleta = {n: COLORES.get(n, "#999999") for n in orden_nombres}
print("Nombres asignados:", NOMBRES)

# %% [markdown]
# ### Visualización de los agrupamientos

# %%
pca = PCA(n_components=2, random_state=SEMILLA)
componentes = pca.fit_transform(X)
df["PC1"], df["PC2"] = componentes[:, 0], componentes[:, 1]
var_exp = pca.explained_variance_ratio_
print(f"Varianza explicada por PCA: PC1 = {var_exp[0]:.1%}, PC2 = {var_exp[1]:.1%} (total {var_exp.sum():.1%})")

centros_pca = pca.transform(km_final.cluster_centers_)
fig, ax = plt.subplots(figsize=(9, 6.5))
sns.scatterplot(
    data=df, x="PC1", y="PC2", hue="Cluster_Nombre", hue_order=orden_nombres,
    palette=paleta, s=10, alpha=0.5, linewidth=0, ax=ax,
)
for cid, (x0, y0) in enumerate(centros_pca):
    ax.scatter(x0, y0, marker="X", s=220, c="black", edgecolor="white", linewidth=1.2, zorder=5)
ax.set_title("Agrupamientos K-Means proyectados con PCA (X = centroides)")
ax.set_xlabel(f"Componente 1 ({var_exp[0]:.1%} de la varianza)")
ax.set_ylabel(f"Componente 2 ({var_exp[1]:.1%} de la varianza)")
ax.legend(title="Clúster", markerscale=2)
guardar_fig(fig, "08_clusters_pca.png")

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.scatterplot(
    data=df, x="pct_JP", y="log_ventas", hue="Cluster_Nombre", hue_order=orden_nombres,
    palette=paleta, s=10, alpha=0.5, linewidth=0, ax=ax,
)
ax.set_title("Ventas globales (log10) vs. proporción de ventas en Japón")
ax.set_xlabel("Proporción de ventas en Japón")
ax.set_ylabel("log10(ventas globales en millones)")
ax.legend(title="Clúster", markerscale=2)
guardar_fig(fig, "09_ventas_vs_japon.png")

fig, ax = plt.subplots(figsize=(7, 3.8))
tabla_cent = centroides_esc.copy()
tabla_cent.index = [NOMBRES[i] for i in tabla_cent.index]
sns.heatmap(tabla_cent, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Centroides (variables estandarizadas)")
guardar_fig(fig, "10_centroides_heatmap.png")

conteo = df["Cluster_Nombre"].value_counts().reindex(orden_nombres)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(conteo.index, conteo.values, color=[paleta[n] for n in conteo.index])
for i, v in enumerate(conteo.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
ax.set_title("Juegos por clúster")
ax.set_ylabel("Juegos")
plt.xticks(rotation=15)
guardar_fig(fig, "11_registros_por_cluster.png")

# %% [markdown]
# ### Comparación de dos configuraciones del modelo
# - **Configuración A (final):** ventas en escala logarítmica + proporciones regionales.
# - **Configuración B:** las mismas variables, pero con las ventas **sin** transformación logarítmica.

# %%
titulo("COMPARACIÓN DE CONFIGURACIONES")
FEATURES_B = ["Global_Sales", "pct_NA", "pct_EU", "pct_JP"]
XB = StandardScaler().fit_transform(df[FEATURES_B])
km_b = KMeans(n_clusters=K_FINAL, n_init=10, random_state=SEMILLA)
etiquetas_b = km_b.fit_predict(XB)
df["_cluster_B"] = etiquetas_b

filas_cfg = []
for nombre, cols, matriz, etiquetas in [
    ("A: log10(ventas) + proporciones", FEATURES, X, df["Cluster"].to_numpy()),
    ("B: ventas sin log + proporciones", FEATURES_B, XB, etiquetas_b),
]:
    tamanos = np.bincount(etiquetas)
    filas_cfg.append(
        {
            "configuracion": nombre,
            "silueta": round(silhouette_score(matriz, etiquetas, sample_size=8000, random_state=SEMILLA), 3),
            "davies_bouldin": round(davies_bouldin_score(matriz, etiquetas), 3),
            "calinski_harabasz": round(calinski_harabasz_score(matriz, etiquetas), 1),
            "tamanos_clusters": sorted(tamanos.tolist(), reverse=True),
            "cluster_mas_pequeno_pct": round(tamanos.min() / tamanos.sum() * 100, 2),
        }
    )
comparacion = pd.DataFrame(filas_cfg)
print(comparacion.to_string(index=False))
comparacion.to_csv(DIR_TAB / "comparacion_configuraciones.csv", index=False)

print("\nPerfil de la configuración B (ventas medias por clúster):")
perfil_b = df.groupby("_cluster_B").agg(
    juegos=("Name", "size"), ventas_media=("Global_Sales", "mean"),
    pct_NA=("pct_NA", "mean"), pct_EU=("pct_EU", "mean"), pct_JP=("pct_JP", "mean"),
)
print(perfil_b.round(3).to_string())
print(f"Concordancia entre A y B (Índice de Rand ajustado): {adjusted_rand_score(df['Cluster'], etiquetas_b):.3f}")
sil_a, sil_b = comparacion["silueta"]
n_pequeno_b = int(np.bincount(etiquetas_b).min())
print(
    f"\nInterpretación: la configuración B obtiene una silueta más alta ({sil_b:.3f} vs. {sil_a:.3f}), pero no es mejor: "
    f"sin el logaritmo, un puñado de superventas domina la escala y K-Means gasta un clúster completo en solo {n_pequeno_b} juegos. "
    "Los otros tres grupos se separan casi solo por región y dejan de distinguir entre juegos de ventas altas y bajas. "
    "Además, la silueta no es comparable entre espacios de variables distintos. Se conserva la configuración A."
)

fig, ejes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, etiquetas, nombre in zip(
    ejes, [df["Cluster"].to_numpy(), etiquetas_b], ["Configuración A (con log)", "Configuración B (sin log)"]
):
    tam = np.bincount(etiquetas)
    ax.bar(range(len(tam)), tam, color="#2E86AB")
    for i, v in enumerate(tam):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_yscale("log")
    ax.set_title(nombre)
    ax.set_xlabel("Clúster")
    ax.set_xticks(range(len(tam)))
ejes[0].set_ylabel("Juegos (escala log)")
guardar_fig(fig, "12_comparacion_configuraciones.png")

df = df.drop(columns="_cluster_B")

# Estabilidad del modelo A con diez semillas distintas
referencia = df["Cluster"].to_numpy()
aris = []
for semilla in range(10):
    otras = KMeans(n_clusters=K_FINAL, n_init=10, random_state=semilla).fit_predict(X)
    aris.append(adjusted_rand_score(referencia, otras))
print(f"\nEstabilidad (Rand ajustado vs. 10 semillas): media = {np.mean(aris):.3f}, mínimo = {np.min(aris):.3f}")

# %% [markdown]
# ## Parte 5. Análisis e interpretación de resultados

# %%
titulo("PARTE 5. ANÁLISIS DE RESULTADOS")


def top_categoria(serie, n=3):
    frec = serie.value_counts(normalize=True).head(n)
    return ", ".join(f"{i} ({v:.0%})" for i, v in frec.items())


total_ventas = df["Global_Sales"].sum()
perfil = df.groupby("Cluster_Nombre").agg(
    juegos=("Name", "size"),
    ventas_totales_M=("Global_Sales", "sum"),
    ventas_media_M=("Global_Sales", "mean"),
    ventas_mediana_M=("Global_Sales", "median"),
    pct_NA=("pct_NA", "mean"),
    pct_EU=("pct_EU", "mean"),
    pct_JP=("pct_JP", "mean"),
    anio_medio=("Year", "mean"),
).reindex(orden_nombres)
perfil["%_juegos"] = perfil["juegos"] / len(df) * 100
perfil["%_ventas"] = perfil["ventas_totales_M"] / total_ventas * 100
perfil["generos_top"] = [top_categoria(df.loc[df["Cluster_Nombre"] == n, "Genre"], 2) for n in perfil.index]
perfil["plataformas_top"] = [top_categoria(df.loc[df["Cluster_Nombre"] == n, "Platform"], 3) for n in perfil.index]
perfil["editores_top"] = [top_categoria(df.loc[df["Cluster_Nombre"] == n, "Publisher"], 2) for n in perfil.index]
perfil = perfil.round(3)
perfil.to_csv(DIR_TAB / "perfil_clusters.csv")
print("Tabla comparativa de los grupos:")
print(perfil.drop(columns=["generos_top", "plataformas_top", "editores_top"]).to_string())
for nombre, fila in perfil.iterrows():
    print(f"\n{nombre}\n  Géneros: {fila['generos_top']}\n  Plataformas: {fila['plataformas_top']}\n  Editores: {fila['editores_top']}")

# Estadísticas descriptivas por grupo
estad = df.groupby("Cluster_Nombre")[["Global_Sales", "pct_NA", "pct_EU", "pct_JP", "Year"]].describe().round(3)
estad.columns = [f"{a}_{b}" for a, b in estad.columns]
estad = estad.reindex(orden_nombres)
estad.to_csv(DIR_TAB / "estadisticas_clusters.csv")
print("\nEstadísticas descriptivas por grupo (ventas globales):")
print(df.groupby("Cluster_Nombre")["Global_Sales"].describe().reindex(orden_nombres).round(3).to_string())

# Centroides con nombre
cent_tabla = pd.concat(
    [centroides_orig.round(4).add_suffix("_orig"), centroides_esc.round(4).add_suffix("_esc")], axis=1
)
cent_tabla.insert(0, "Cluster_Nombre", [NOMBRES[i] for i in cent_tabla.index])
cent_tabla.index.name = "Cluster"
cent_tabla.to_csv(DIR_TAB / "centroides.csv")

# Gráficas de comparación entre grupos
fig, ax = plt.subplots(figsize=(8, 4.5))
perfil[["pct_NA", "pct_EU", "pct_JP"]].assign(Otras=lambda t: 1 - t.sum(axis=1)).plot.bar(
    stacked=True, ax=ax, color=["#2E86AB", "#76B041", "#F2A541", "#999999"]
)
ax.set_title("Distribución regional promedio de las ventas por clúster")
ax.set_ylabel("Proporción")
ax.set_xlabel("")
ax.legend(["Norteamérica", "Europa", "Japón", "Otras"], bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xticks(rotation=15)
guardar_fig(fig, "13_composicion_regional_clusters.png")

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(
    data=df, x="Cluster_Nombre", y="Global_Sales", order=orden_nombres, palette=paleta, hue="Cluster_Nombre",
    legend=False, ax=ax, fliersize=2,
)
ax.set_yscale("log")
ax.set_title("Ventas globales por clúster (escala logarítmica)")
ax.set_xlabel("")
ax.set_ylabel("Millones de unidades")
plt.xticks(rotation=15)
guardar_fig(fig, "14_ventas_por_cluster.png")

# Hallazgos calculados con los datos
exitos = perfil.loc["Éxitos occidentales"]
nintendo = df[df["Publisher"] == "Nintendo"]
nin_exitos = (nintendo["Cluster_Nombre"] == "Éxitos occidentales").mean()
exitos_gen_total = (df["Cluster_Nombre"] == "Éxitos occidentales").mean()
japon = perfil.loc["Nicho japonés"]
europa = perfil.loc["Nicho europeo"]
pc_europa = (df.loc[df["Cluster_Nombre"] == "Nicho europeo", "Platform"] == "PC").mean()
pc_total = (df["Platform"] == "PC").mean()
rpg_aventura_jp = df.loc[df["Cluster_Nombre"] == "Nicho japonés", "Genre"].isin(["Role-Playing", "Adventure"]).mean()
rpg_aventura_tot = df["Genre"].isin(["Role-Playing", "Adventure"]).mean()

hallazgos = [
    f"Concentración: los 'Éxitos occidentales' son solo {exitos['%_juegos']:.1f}% de los juegos, pero generan "
    f"{exitos['%_ventas']:.1f}% de las unidades vendidas (mediana de {exitos['ventas_mediana_M']:.2f} M por juego).",
    f"Mercado japonés separado: {int(japon['juegos']):,} juegos venden en promedio {japon['pct_JP']:.0%} en Japón; "
    f"los géneros Role-Playing y Adventure representan {rpg_aventura_jp:.0%} de este grupo frente a {rpg_aventura_tot:.0%} en todo el catálogo.",
    f"Europa y PC: el 'Nicho europeo' ({int(europa['juegos']):,} juegos) concentra {europa['pct_EU']:.0%} de sus ventas en Europa y "
    f"{pc_europa:.0%} de sus juegos son de PC, frente a {pc_total:.0%} de PC en el catálogo completo.",
    f"Nintendo: {nin_exitos:.0%} de los juegos de Nintendo cae en 'Éxitos occidentales', contra {exitos_gen_total:.0%} del catálogo total.",
]
recomendaciones = [
    "Priorizar el presupuesto de lanzamiento en los títulos con perfil de 'Éxitos occidentales' (ventas altas repartidas entre Norteamérica y Europa), "
    f"porque poco más de un tercio de los juegos ({exitos['%_juegos']:.0f}%) explica cerca de {exitos['%_ventas']:.0f}% de las unidades vendidas.",
    "Para el mercado japonés, planear lanzamientos específicos (géneros de rol y aventura; PSP y DS destacan entre las plataformas del grupo) en lugar de esperar que un título occidental se venda igual allá.",
    "Para Europa, considerar PC como plataforma prioritaria en juegos de menor escala, porque casi la mitad del grupo europeo son juegos de PC.",
    "Usar el clúster de un juego similar como referencia al estimar expectativas de ventas por región antes de un lanzamiento.",
]
limitaciones = [
    "Son estimaciones de ventas que llegan hasta 2016: no incluyen juegos recientes y el dataset no distingue entre ventas físicas y digitales.",
    "Las ventas están redondeadas a 0.01 M: en juegos con ventas muy bajas, las proporciones regionales son burdas (un juego con 0.02 M puede aparecer con 100% en una sola región), lo que refuerza los 'nichos'.",
    f"La silueta del modelo final es moderada ({fila_k.loc[K_FINAL, 'silueta']:.2f}); los grupos se traslapan en sus fronteras, sobre todo entre 'Éxitos occidentales' y los nichos norteamericano y europeo.",
    "K-Means asume grupos compactos y de forma aproximadamente esférica, y asigna cada juego a un solo grupo sin indicar qué tan claro es el caso.",
    "No hay variables de calidad, presupuesto, marketing o calificaciones; solo ventas.",
]
print("\nHALLAZGOS")
for i, h in enumerate(hallazgos, 1):
    print(f"{i}. {h}")
print("\nRECOMENDACIONES")
for i, r in enumerate(recomendaciones, 1):
    print(f"{i}. {r}")
print("\nLIMITACIONES")
for i, lim in enumerate(limitaciones, 1):
    print(f"{i}. {lim}")

# %% [markdown]
# ## Exportación de resultados

# %%
titulo("EXPORTACIÓN")
columnas_salida = [
    "Rank", "Name", "Platform", "Year", "Genre", "Publisher",
    "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales",
    "log_ventas", "pct_NA", "pct_EU", "pct_JP", "Cluster", "Cluster_Nombre", "PC1", "PC2",
]
df[columnas_salida].round(5).to_csv(RUTA_PROC, index=False)

resumen = {
    "registros_originales": registro["registros_originales"],
    "registros_finales": registro["registros_finales"],
    "k_final": K_FINAL,
    "variables_modelo": FEATURES,
    "silueta": round(float(fila_k.loc[K_FINAL, "silueta"]), 4),
    "davies_bouldin": round(float(fila_k.loc[K_FINAL, "davies_bouldin"]), 4),
    "calinski_harabasz": round(float(fila_k.loc[K_FINAL, "calinski_harabasz"]), 1),
    "inercia": round(float(km_final.inertia_), 1),
    "varianza_pca": [round(float(v), 4) for v in var_exp],
    "estabilidad_ari_media": round(float(np.mean(aris)), 4),
    "nombres_clusters": {str(k): v for k, v in NOMBRES.items()},
    "limpieza": registro,
    "hallazgos": hallazgos,
    "recomendaciones": recomendaciones,
    "limitaciones": limitaciones,
}
with open(DIR_TAB / "resumen.json", "w", encoding="utf-8") as f:
    json.dump(resumen, f, ensure_ascii=False, indent=2)

print(f"Dataset procesado: {RUTA_PROC.relative_to(RAIZ)} ({len(df):,} filas)")
print(f"Tablas en: {DIR_TAB.relative_to(RAIZ)}  |  Figuras en: {DIR_FIG.relative_to(RAIZ)}")

if _archivo_reporte is not None:
    sys.stdout = sys.__stdout__
    _archivo_reporte.close()

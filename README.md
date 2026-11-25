# Segmentación de videojuegos con K-Means y dashboard interactivo

Proyecto integrador de minería de datos (tercera unidad). Agrupa más de 16,500 videojuegos según su **volumen de ventas** y **dónde se venden** (Norteamérica, Europa y Japón) mediante el algoritmo no supervisado K-Means, y presenta los resultados en un dashboard de Streamlit.

## Problema

La industria de videojuegos es muy diversa en sus mercados: no todos los títulos venden igual ni en las mismas regiones. El proyecto identifica perfiles de juegos para apoyar decisiones de lanzamiento y distribución.

## Resultados principales

| Clúster | Juegos | % de juegos | % de ventas | Perfil |
|---|---|---|---|---|
| Éxitos occidentales | 5,681 | 34.2% | 79.9% | Ventas altas, repartidas entre Norteamérica (50%) y Europa (33%) |
| Nicho norteamericano | 5,694 | 34.3% | 10.9% | Ventas modestas, 82% en Norteamérica |
| Nicho japonés | 3,689 | 22.2% | 7.3% | 97% de las ventas en Japón; peso de rol y aventura |
| Nicho europeo | 1,531 | 9.2% | 1.9% | 81% en Europa; 47% son juegos de PC |

- **K = 4**, elegido con el método del codo, silueta y Davies-Bouldin (K = 2 a 10).
- Coeficiente de silueta del modelo final: **0.44** (moderado: los grupos se traslapan en sus fronteras).
- Estabilidad del modelo: Índice de Rand ajustado medio de 0.999 con 10 semillas distintas.
- Se comparan dos configuraciones: con y sin transformación logarítmica de las ventas (ver `outputs/tables/comparacion_configuraciones.csv`).

## Estructura del repositorio

```
├── README.md
├── requirements.txt
├── data/
│   ├── raw/vgsales.csv                    # dataset original
│   └── processed/vgsales_clusters.csv     # dataset limpio con clúster asignado y PCA
├── src/
│   └── analisis_kmeans.py                 # limpieza, K-Means, análisis y exportación
├── notebooks/
│   └── analisis_kmeans.ipynb              # mismo análisis en formato notebook
├── dashboard/
│   └── app.py                             # dashboard en Streamlit (2 páginas)
├── outputs/
│   ├── figures/                           # gráficas generadas (PNG)
│   ├── tables/                            # métricas, centroides, perfiles (CSV y JSON)
│   └── reporte_consola.txt                # salida completa del análisis
└── docs/                                  # documentación en PDF
```

## Herramientas

Python 3.10 o superior, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn y Streamlit.

## Cómo ejecutar

1. Clonar el repositorio e instalar dependencias (se recomienda un entorno virtual):

   ```bash
   pip install -r requirements.txt
   ```

2. Ejecutar el análisis (opcional: los resultados ya vienen incluidos). Regenera el dataset procesado, las tablas y las gráficas:

   ```bash
   python src/analisis_kmeans.py
   ```

   También se puede abrir `notebooks/analisis_kmeans.ipynb` en Jupyter o Google Colab y ejecutar las celdas en orden (en Colab hay que subir primero la carpeta `data/raw`).

3. Abrir el dashboard desde la carpeta raíz del proyecto:

   ```bash
   streamlit run dashboard/app.py
   ```

   Se abre en `http://localhost:8501`.

## Dashboard

- **Página 1, Análisis general:** 4 KPI, ventas por año, género, región, plataforma y editor, distribución de ventas y tabla de los juegos con más ventas (con descarga en CSV).
- **Página 2, Segmentación K-Means:** 4 KPI, distribución de juegos por clúster, dispersión con PCA, composición regional y ventas por clúster, tabla de estadísticas descriptivas, centroides, descripción de los grupos, hallazgos y recomendaciones.
- **Filtros (barra lateral):** género, plataforma, rango de años, editor (por ejemplo, Nintendo) y clúster (página 2).

## Dataset

*Video Game Sales* (Kaggle, `vgsales.csv`): 16,598 juegos con ventas en millones de unidades por región (Norteamérica, Europa, Japón, otras y global), plataforma, género, editor y año. Tras la limpieza quedan 16,595 registros.

Limpieza aplicada: imputación del año faltante (271 registros) con la mediana de su plataforma, editor faltante (58) como `Unknown`, eliminación de 2 duplicados lógicos (mismo nombre, plataforma y año) y de 1 registro sin ventas regionales. Los valores atípicos de ventas no se eliminan: se aplica una transformación logarítmica.

## Metodología

Variables del modelo: `log_ventas` (log10 de ventas globales) y `pct_NA`, `pct_EU`, `pct_JP` (proporción de ventas por región), estandarizadas con `StandardScaler`. Género, plataforma y editor se usan solo para interpretar los grupos y como filtros.

## Limitaciones

- El dataset llega hasta 2016 y no distingue entre ventas físicas y digitales.
- Las ventas están redondeadas a 0.01 millones, por lo que las proporciones regionales de juegos con ventas muy bajas son burdas y refuerzan los grupos de "nicho".
- K-Means asume grupos compactos y asigna cada juego a un solo clúster.
- No hay variables de calidad, presupuesto ni marketing.

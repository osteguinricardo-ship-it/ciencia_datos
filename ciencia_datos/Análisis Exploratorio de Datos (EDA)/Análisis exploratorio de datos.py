import pandas as pd
import numpy as np           # Añadido del artículo para operaciones numéricas
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as wr        # Añadido para omitir advertencias visuales
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Omitir advertencias como sugiere el artículo
wr.filterwarnings('ignore') 

# --- VENTANA PARA SELECCIONAR EL ARCHIVO ---
def seleccionar_archivo():
    root = tk.Tk()
    root.withdraw() # Ocultar la ventana principal de tkinter
    ruta_archivo = filedialog.askopenfilename(
        title="Selecciona el archivo wine-quality.csv",
        filetypes=[("Archivos CSV", "*.csv")]
    )
    return ruta_archivo

# 1. DEFINIR EL PROBLEMA Y CARGAR DATOS
print("Esperando selección de archivo...")
ruta = seleccionar_archivo()

if not ruta:
    print("No se seleccionó ningún archivo. Saliendo...")
else:
    # 2. RECOLECCIÓN (Carga)
    df = pd.read_csv(ruta)
    print("Archivo cargado con éxito.\n")

    # 3. LIMPIEZA DE DATOS (Tu código original)
    # Verificamos nulos y rellenamos con la mediana
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # =================================================================
    # NUEVOS PASOS: EXPLORATORY DATA ANALYSIS (Basado en GeeksforGeeks)
    # =================================================================
    
    print("\n--- Paso 3: Analizando los Datos ---")
    print("Forma del dataset (filas, columnas):", df.shape)
    print("\nInformación del dataset:")
    df.info()
    print("\nResumen estadístico transpuesto (describe.T):")
    print(df.describe().T)
    print("\nLista de columnas:")
    print(df.columns.tolist())

    print("\n--- Paso 4: Comprobación de Valores Nulos ---")
    print(df.isnull().sum())

    print("\n--- Paso 5: Valores Únicos (nunique) ---")
    print(df.nunique())

    print("\n--- Paso 6: Análisis Univariante ---")
    # 1. Gráfico de barras para evaluar el conteo de calidad del vino
    quality_counts = df['quality'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.bar(quality_counts.index, quality_counts, color='deeppink')
    plt.title('Count Plot of Quality (Gráfico de Barras)')
    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.show()

    # 2. Gráfico de densidad (Kernel Density Plot) para entender la varianza y asimetría (Skewness)
    sns.set_style("darkgrid")
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    plt.figure(figsize=(14, len(numerical_columns) * 3))
    for idx, feature in enumerate(numerical_columns, 1):
        plt.subplot(len(numerical_columns), 2, idx)
        sns.histplot(df[feature], kde=True)
        plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")
    plt.tight_layout()
    plt.show()

    # 3. Gráfico de Enjambre (Swarm Plot) para mostrar valores atípicos (outliers)
    plt.figure(figsize=(10, 8))
    sns.swarmplot(x="quality", y="alcohol", data=df, palette='viridis')
    plt.title('Swarm Plot for Quality and Alcohol')
    plt.xlabel('Quality')
    plt.ylabel('Alcohol')
    plt.show()

    print("\n--- Paso 7: Análisis Bivariante ---")
    # 1. Gráfico de Pares (Pair Plot) para distribución de variables individuales e interacciones
    sns.set_palette("Pastel1")
    plt.figure(figsize=(10, 6))
    sns.pairplot(df)
    plt.suptitle('Pair Plot for DataFrame', y=1.02)
    plt.show()

    # 2. Gráfico de Violín (Violin Plot) para examinar la relación entre alcohol y calidad
    # Nota: Usamos una copia del df para convertir la calidad a string solo para el gráfico, 
    # y así no afectar tu modelo predictivo posterior.
    df_plot = df.copy()
    df_plot['quality'] = df_plot['quality'].astype(str)
    plt.figure(figsize=(10, 8))
    sns.violinplot(x="quality", y="alcohol", data=df_plot, palette={
        '3': 'lightcoral', '4': 'lightblue', '5': 'lightgreen', 
        '6': 'gold', '7': 'lightskyblue', '8': 'lightpink'}, alpha=0.7)
    plt.title('Violin Plot for Quality and Alcohol')
    plt.xlabel('Quality')
    plt.ylabel('Alcohol')
    plt.show()

    # 3. Gráfico de Caja (Box Plot)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='quality', y='alcohol', data=df)
    plt.title('Box Plot for Quality and Alcohol')
    plt.show()

    print("\n--- Paso 8: Análisis Multivariante ---")
    # Mapa de calor de correlación (Correlation Heatmap) con el estilo de GFG
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Pastel2', linewidths=2)
    plt.title('Correlation Heatmap')
    plt.show()

    # =================================================================
    # CONTINUACIÓN DE TU CÓDIGO ORIGINAL (Modelo Predictivo)
    # =================================================================
    print("\n--- INTERPRETAR (Modelo Predictivo) ---")
    # Queremos predecir la 'quality'
    X = df.drop('quality', axis=1)
    y = df['quality']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predicciones = model.predict(X_val)
    precision = accuracy_score(y_val, predicciones)

    print("-" * 30)
    print(f"ANÁLISIS COMPLETADO")
    print(f"Precisión del modelo predictivo: {precision:.2%}")
    print("-" * 30)

    # Paso extra: Guardar un resumen de los resultados
    df.describe().to_csv("resumen_estadistico_vino.csv")
    print("Se ha generado 'resumen_estadistico_vino.csv' con las estadísticas del archivo.")
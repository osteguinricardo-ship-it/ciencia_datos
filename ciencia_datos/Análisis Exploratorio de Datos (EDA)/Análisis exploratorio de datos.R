# =================================================================
# ANÁLISIS EXPLORATORIO DE DATOS (EDA) Y MODELADO EN R
# =================================================================

# Omitir advertencias de visualización/cargas
options(warn = -1)

# --- INSTALACIÓN Y CARGA DE PAQUETES ---
instalar_y_cargar <- function(paquetes) {
  nuevos <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
  if (length(nuevos) > 0) {
    message("Instalando paquetes requeridos: ", paste(nuevos, collapse = ", "))
    install.packages(nuevos, dependencies = TRUE, repos = "https://cloud.r-project.org")
  }
  for (pkg in paquetes) {
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }
}

# Definimos los paquetes necesarios para emular el flujo de Python
paquetes_requeridos <- c("ggplot2", "GGally", "corrplot", "ggbeeswarm", "randomForest", "gridExtra")
instalar_y_cargar(paquetes_requeridos)

# --- VENTANA PARA SELECCIONAR EL ARCHIVO ---
seleccionar_archivo <- function() {
  ruta_archivo <- NULL
  # Intentar usar choose.files para una interfaz nativa en Windows con filtros
  tryCatch({
    ruta_archivo <- choose.files(
      caption = "Selecciona el archivo wine-quality.csv o WineQT.csv",
      filters = matrix(c("Archivos CSV (*.csv)", "*.csv"), 1, 2, byrow = TRUE),
      multi = FALSE
    )
  }, error = function(e) {
    # Alternativa multiplataforma si choose.files no está disponible o falla
    tryCatch({
      ruta_archivo <- file.choose()
    }, error = function(e2) {
      ruta_archivo <- NULL
    })
  })
  return(ruta_archivo)
}

# 1. DEFINIR EL PROBLEMA Y CARGAR DATOS
message("Esperando selección de archivo...")
ruta <- seleccionar_archivo()

if (is.null(ruta) || length(ruta) == 0 || ruta == "") {
  message("No se seleccionó ningún archivo. Saliendo...")
} else {
  # 2. RECOLECCIÓN (Carga)
  df <- read.csv(ruta)
  message("Archivo cargado con éxito.\n")

  # 3. LIMPIEZA DE DATOS (Código equivalente a pandas .fillna con mediana)
  imputar_mediana <- function(data) {
    for (col in names(data)) {
      if (is.numeric(data[[col]])) {
        mediana <- median(data[[col]], na.rm = TRUE)
        data[[col]][is.na(data[[col]])] <- mediana
      }
    }
    return(data)
  }
  df <- imputar_mediana(df)

  # =================================================================
  # PASOS: EXPLORATORY DATA ANALYSIS (EDA)
  # =================================================================

  cat("\n--- Paso 3: Analizando los Datos ---\n")
  cat("Forma del dataset (filas, columnas):", paste(dim(df), collapse = ", "), "\n")
  
  cat("\nInformación del dataset:\n")
  str(df)

  # Resumen estadístico transpuesto (Equivalente a df.describe().T)
  cat("\nResumen estadístico transpuesto (describe.T):\n")
  describir_T <- function(data) {
    num_cols <- data[sapply(data, is.numeric)]
    resumen <- do.call(rbind, lapply(num_cols, function(x) {
      c(
        count = sum(!is.na(x)),
        mean = mean(x, na.rm = TRUE),
        std = sd(x, na.rm = TRUE),
        min = min(x, na.rm = TRUE),
        `25%` = as.numeric(quantile(x, 0.25, na.rm = TRUE)),
        `50%` = as.numeric(quantile(x, 0.50, na.rm = TRUE)),
        `75%` = as.numeric(quantile(x, 0.75, na.rm = TRUE)),
        max = max(x, na.rm = TRUE)
      )
    }))
    return(as.data.frame(resumen))
  }
  print(describir_T(df))

  cat("\nLista de columnas:\n")
  print(names(df))

  cat("\n--- Paso 4: Comprobación de Valores Nulos ---\n")
  print(colSums(is.na(df)))

  cat("\n--- Paso 5: Valores Únicos (nunique) ---\n")
  print(sapply(df, function(x) length(unique(x))))

  cat("\n--- Paso 6: Análisis Univariante ---\n")
  
  # 1. Gráfico de barras para evaluar el conteo de calidad del vino (deeppink)
  p1 <- ggplot(df, aes(x = factor(quality))) +
    geom_bar(fill = "deeppink") +
    labs(title = "Count Plot of Quality (Gráfico de Barras)", x = "Quality", y = "Count") +
    theme_minimal()
  print(p1)

  # Función para calcular asimetría (skewness) compatible con pandas.skew()
  skewness <- function(x) {
    n <- length(x)
    if (n < 3) return(0)
    mean_val <- mean(x)
    sd_val <- sd(x)
    if (sd_val == 0) return(0)
    factor <- n / ((n - 1) * (n - 2))
    sum(((x - mean_val) / sd_val)^3) * factor
  }

  # 2. Gráfico de densidad (Kernel Density Plot) con asimetría (Skewness) anotada
  numerical_columns <- names(df)[sapply(df, is.numeric)]
  plots <- list()
  for (feature in numerical_columns) {
    skew_val <- round(skewness(df[[feature]]), 2)
    p_dens <- ggplot(df, aes(x = .data[[feature]])) +
      geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "#5DADE2", color = "white", alpha = 0.7) +
      geom_density(color = "#E74C3C", linewidth = 1) +
      labs(title = paste(feature, "| Skewness:", skew_val), x = feature, y = "Density") +
      theme_minimal() +
      theme(plot.title = element_text(size = 9))
    plots[[feature]] <- p_dens
  }
  # Organizar plots en 2 columnas
  grid.arrange(grobs = plots, ncol = 2)

  # 3. Gráfico de Enjambre (Swarm Plot)
  # Usamos tryCatch para usar ggbeeswarm o caer elegantemente en geom_jitter
  p3 <- tryCatch({
    ggplot(df, aes(x = factor(quality), y = alcohol, color = factor(quality))) +
      geom_beeswarm(alpha = 0.7, size = 1.2) +
      scale_color_viridis_d(option = "D") +
      labs(title = "Swarm Plot for Quality and Alcohol", x = "Quality", y = "Alcohol") +
      theme_minimal() +
      theme(legend.position = "none")
  }, error = function(e) {
    # Fallback si ggbeeswarm tiene algún inconveniente
    ggplot(df, aes(x = factor(quality), y = alcohol, color = factor(quality))) +
      geom_jitter(width = 0.2, alpha = 0.7, size = 1.2) +
      scale_color_viridis_d(option = "D") +
      labs(title = "Jitter Plot for Quality and Alcohol (Fallback)", x = "Quality", y = "Alcohol") +
      theme_minimal() +
      theme(legend.position = "none")
  })
  print(p3)

  cat("\n--- Paso 7: Análisis Bivariante ---\n")
  
  # 1. Gráfico de Pares (Pair Plot)
  p4 <- ggpairs(df, progress = FALSE) +
    theme_minimal() +
    labs(title = "Pair Plot for DataFrame")
  print(p4)

  # 2. Gráfico de Violín (Violin Plot) con paleta personalizada
  df_plot <- df
  df_plot$quality <- as.character(df_plot$quality)
  custom_palette <- c(
    "3" = "lightcoral", "4" = "lightblue", "5" = "lightgreen", 
    "6" = "gold", "7" = "lightskyblue", "8" = "lightpink"
  )
  p5 <- ggplot(df_plot, aes(x = quality, y = alcohol, fill = quality)) +
    geom_violin(alpha = 0.7) +
    scale_fill_manual(values = custom_palette) +
    labs(title = "Violin Plot for Quality and Alcohol", x = "Quality", y = "Alcohol") +
    theme_minimal()
  print(p5)

  # 3. Gráfico de Caja (Box Plot)
  p6 <- ggplot(df, aes(x = factor(quality), y = alcohol)) +
    geom_boxplot(fill = "lightblue") +
    labs(title = "Box Plot for Quality and Alcohol", x = "Quality", y = "Alcohol") +
    theme_minimal()
  print(p6)

  cat("\n--- Paso 8: Análisis Multivariante ---\n")
  
  # Mapa de calor de correlación (Correlation Heatmap)
  cor_matrix <- cor(df[sapply(df, is.numeric)])
  corrplot(cor_matrix, method = "color", type = "upper", 
           addCoef.col = "black", number.cex = 0.8,
           col = colorRampPalette(c("#E8F8F5", "#FFFFFF", "#FADBD8"))(200),
           tl.col = "black", tl.srt = 45,
           mar = c(0, 0, 2, 0),
           title = "Correlation Heatmap")

  # =================================================================
  # MODELADO PREDICTIVO (Random Forest)
  # =================================================================
  cat("\n--- INTERPRETAR (Modelo Predictivo) ---\n")
  
  # Preparar datos (convertir la variable objetivo a factor para Clasificación)
  df_model <- df
  df_model$quality <- as.factor(df_model$quality)

  # División entrenamiento y validación (80% / 20%)
  set.seed(42)
  train_idx <- sample(seq_len(nrow(df_model)), size = round(0.8 * nrow(df_model)))
  train_data <- df_model[train_idx, ]
  val_data <- df_model[-train_idx, ]

  # Ajustar el modelo Random Forest (100 árboles)
  model <- randomForest(quality ~ ., data = train_data, ntree = 100)

  # Predicciones y cálculo de la precisión (Accuracy)
  predicciones <- predict(model, newdata = val_data)
  precision <- mean(predicciones == val_data$quality)

  cat(rep("-", 30), collapse = "", "\n")
  cat("ANÁLISIS COMPLETADO\n")
  cat(sprintf("Precisión del modelo predictivo: %.2f%%\n", precision * 100))
  cat(rep("-", 30), collapse = "", "\n")

  # Paso extra: Guardar un resumen de los resultados
  resumen_csv <- describir_T(df)
  write.csv(resumen_csv, "resumen_estadistico_vino.csv", row.names = TRUE)
  cat("Se ha generado 'resumen_estadistico_vino.csv' con las estadísticas del archivo.\n")
}

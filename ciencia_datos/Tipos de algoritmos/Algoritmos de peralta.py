import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")
 
# Scikit-learn imports
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
 
np.random.seed(42)
 
# ─────────────────────────────────────────────
# Paleta de colores unificada
# ─────────────────────────────────────────────
C1, C2, C3 = "#4C72B0", "#DD8452", "#55A868"
CMAP2 = ListedColormap(["#AED6F1", "#FFDAB9"])
CMAP3 = ListedColormap(["#AED6F1", "#FFDAB9", "#A9DFBF"])
 
# ─────────────────────────────────────────────
# 1. REGRESIÓN LINEAL
# ─────────────────────────────────────────────
def plot_linear_regression(ax):
    tamaño = np.array([50,60,75,80,90,100,110,120,130,150,160,175,200], dtype=float)
    precio = tamaño * 1.8 + np.random.normal(0, 15, len(tamaño)) + 20
 
    model = LinearRegression()
    model.fit(tamaño.reshape(-1,1), precio)
    x_line = np.linspace(40, 210, 200)
    y_line = model.predict(x_line.reshape(-1,1))
 
    ax.scatter(tamaño, precio, color=C1, s=70, zorder=5, label="Datos reales")
    ax.plot(x_line, y_line, color=C2, lw=2.5, label=f"Recta: y={model.coef_[0]:.2f}x+{model.intercept_:.1f}")
    ax.set_title("1. Regresión Lineal\nPrecio de casas vs Tamaño", fontweight="bold")
    ax.set_xlabel("Tamaño (m²)"); ax.set_ylabel("Precio (miles $)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
 
# ─────────────────────────────────────────────
# 2. REGRESIÓN LOGÍSTICA
# ─────────────────────────────────────────────
def plot_logistic_regression(ax):
    horas = np.array([1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10], dtype=float)
    aprobado = np.array([0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1])
 
    model = LogisticRegression()
    model.fit(horas.reshape(-1,1), aprobado)
    x_line = np.linspace(0, 11, 300)
    prob = model.predict_proba(x_line.reshape(-1,1))[:,1]
 
    ax.scatter(horas, aprobado, color=[C2 if y==1 else C1 for y in aprobado], s=70, zorder=5)
    ax.plot(x_line, prob, color=C3, lw=2.5, label="Probabilidad de aprobar")
    ax.axhline(0.5, color="gray", linestyle="--", lw=1.2, label="Umbral 50%")
    ax.set_title("2. Regresión Logística\n¿Aprobará el examen?", fontweight="bold")
    ax.set_xlabel("Horas de estudio"); ax.set_ylabel("Probabilidad")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    custom = [plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=C1,ms=8,label='Reprobó'),
              plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=C2,ms=8,label='Aprobó')]
    ax.legend(handles=custom+[ax.lines[0], ax.lines[1]], fontsize=7); ax.grid(alpha=0.3)
 
# ─────────────────────────────────────────────
# 3. ÁRBOL DE DECISIÓN
# ─────────────────────────────────────────────
def plot_decision_tree(ax):
    # Dataset: préstamos bancarios
    X = np.array([[80,0],[90,0],[60,1],[50,1],[70,0],[40,2],[30,2],[55,1],[95,0],[25,2]])
    y = np.array([1,1,1,0,1,0,0,0,1,0])
 
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    plot_tree(model, feature_names=["Sueldo","Deudas"], class_names=["Rechazado","Aprobado"],
              filled=True, rounded=True, ax=ax, fontsize=8,
              impurity=False, proportion=False)
    ax.set_title("3. Árbol de Decisión\nAprobación de préstamos bancarios", fontweight="bold")
 
# ─────────────────────────────────────────────
# 4. BOSQUE ALEATORIO
# ─────────────────────────────────────────────
def plot_random_forest(ax):
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                n_clusters_per_class=1, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
 
    h = 0.03
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h),
                          np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
 
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=CMAP2)
    ax.scatter(X[y==0,0], X[y==0,1], color=C1, s=30, label="Sin fraude")
    ax.scatter(X[y==1,0], X[y==1,1], color=C2, s=30, label="Fraude")
    acc = accuracy_score(y, model.predict(X))
    ax.set_title(f"4. Bosque Aleatorio\nDetección de fraude · Acc={acc:.2%}", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
 
# ─────────────────────────────────────────────
# 5. SVM
# ─────────────────────────────────────────────
def plot_svm(ax):
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                n_clusters_per_class=1, class_sep=1.5, random_state=5)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = SVC(kernel="linear", C=1)
    model.fit(X, y)
 
    h = 0.02
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h),
                          np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
 
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=CMAP2)
    ax.scatter(X[y==0,0], X[y==0,1], color=C1, s=40, label="Clase 0")
    ax.scatter(X[y==1,0], X[y==1,1], color=C2, s=40, label="Clase 1")
 
    # Hiperplano y márgenes
    w = model.coef_[0]; b = model.intercept_[0]
    x_hp = np.linspace(X[:,0].min(), X[:,0].max(), 200)
    ax.plot(x_hp, -(w[0]*x_hp + b)/w[1], color="black", lw=2, label="Hiperplano")
    ax.plot(x_hp, -(w[0]*x_hp + b - 1)/w[1], color="gray", lw=1, linestyle="--")
    ax.plot(x_hp, -(w[0]*x_hp + b + 1)/w[1], color="gray", lw=1, linestyle="--")
    ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1],
               s=120, facecolors="none", edgecolors="black", lw=1.5, label="Vectores soporte")
 
    ax.set_title("5. SVM — Máquinas de Vectores de Soporte\nClasificación con margen máximo", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
 
# ─────────────────────────────────────────────
# 6. K-NN
# ─────────────────────────────────────────────
def plot_knn(ax):
    X, y = make_classification(n_samples=150, n_features=2, n_redundant=0,
                                n_clusters_per_class=1, random_state=10)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
 
    h = 0.03
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h),
                          np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
 
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=CMAP2)
    ax.scatter(X[y==0,0], X[y==0,1], color=C1, s=35, label="Clase A")
    ax.scatter(X[y==1,0], X[y==1,1], color=C2, s=35, label="Clase B")
 
    # Mostrar un punto nuevo y sus 5 vecinos
    nuevo = np.array([[0.5, 0.5]])
    distancias, indices = model.kneighbors(nuevo)
    ax.scatter(*nuevo[0], color="black", s=150, zorder=10, marker="*", label="Nuevo punto")
    for idx in indices[0]:
        ax.plot([nuevo[0,0], X[idx,0]], [nuevo[0,1], X[idx,1]],
                color="black", lw=1, linestyle=":", alpha=0.7)
 
    ax.set_title("6. K-NN (K=5)\nSistema de recomendación / Clasificación", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
 
# ─────────────────────────────────────────────
# 7. NAIVE BAYES
# ─────────────────────────────────────────────
def plot_naive_bayes(ax):
    # Palabras en emails: simulación frecuencia de palabras
    palabras = ["oferta","gratis","dinero","ganador","banco","urgente","amigo","trabajo","reunión","proyecto"]
    freq_spam = np.array([0.85, 0.78, 0.72, 0.68, 0.55, 0.62, 0.10, 0.12, 0.08, 0.05])
    freq_ham  = np.array([0.10, 0.08, 0.15, 0.05, 0.20, 0.07, 0.45, 0.55, 0.60, 0.65])
 
    x = np.arange(len(palabras))
    w = 0.38
    bars1 = ax.bar(x - w/2, freq_spam, w, color=C2, alpha=0.85, label="SPAM")
    bars2 = ax.bar(x + w/2, freq_ham,  w, color=C1, alpha=0.85, label="NO SPAM")
 
    ax.set_title("7. Naive Bayes\nProbabilidad de palabras en emails (Filtro Spam)", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(palabras, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Frecuencia relativa"); ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
 
# ─────────────────────────────────────────────
# 8. GRADIENT BOOSTING
# ─────────────────────────────────────────────
def plot_gradient_boosting(ax):
    n_est_list = [1, 5, 10, 20, 50, 100, 150, 200]
    X, y = make_classification(n_samples=300, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
    train_scores, test_scores = [], []
    for n in n_est_list:
        m = GradientBoostingClassifier(n_estimators=n, random_state=42)
        m.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, m.predict(X_train)))
        test_scores.append(accuracy_score(y_test,  m.predict(X_test)))
 
    ax.plot(n_est_list, train_scores, color=C1, marker="o", ms=5, lw=2, label="Entrenamiento")
    ax.plot(n_est_list, test_scores,  color=C2, marker="s", ms=5, lw=2, label="Prueba")
    ax.fill_between(n_est_list, train_scores, test_scores, alpha=0.15, color="gray")
    ax.set_title("8. Gradient Boosting\nMejora progresiva con más árboles", fontweight="bold")
    ax.set_xlabel("Número de estimadores"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0.7, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
 
# ─────────────────────────────────────────────
# 9. RED NEURONAL ARTIFICIAL (ANN)
# ─────────────────────────────────────────────
def plot_ann(ax):
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                n_clusters_per_class=1, class_sep=0.8, random_state=7)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    model.fit(X, y)
 
    h = 0.03
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h),
                          np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
 
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=CMAP2)
    ax.scatter(X[y==0,0], X[y==0,1], color=C1, s=30, label="Clase 0")
    ax.scatter(X[y==1,0], X[y==1,1], color=C2, s=30, label="Clase 1")
    acc = accuracy_score(y, model.predict(X))
    ax.set_title(f"9. Red Neuronal Artificial (ANN)\nFrontera no lineal · Acc={acc:.2%}", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
 
# ─────────────────────────────────────────────
# 10. LDA — Análisis Discriminante Lineal
# ─────────────────────────────────────────────
def plot_lda(ax):
    iris = load_iris()
    X, y = iris.data, iris.target
 
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)
 
    colors = [C1, C2, C3]
    for cls, color, name in zip(range(3), colors, iris.target_names):
        ax.scatter(X_lda[y==cls, 0], X_lda[y==cls, 1],
                   color=color, s=45, alpha=0.8, label=name.capitalize())
 
    ax.set_title("10. LDA — Análisis Discriminante Lineal\nIris Dataset proyectado a 2D", fontweight="bold")
    ax.set_xlabel("Componente Discriminante 1")
    ax.set_ylabel("Componente Discriminante 2")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
 
# ─────────────────────────────────────────────
# FIGURA PRINCIPAL — 2×5 subgráficas
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor("#F8F9FA")
 
axes = []
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.set_facecolor("white")
    axes.append(ax)
 
plot_linear_regression(axes[0])
plot_logistic_regression(axes[1])
plot_decision_tree(axes[2])
plot_random_forest(axes[3])
plot_svm(axes[4])
plot_knn(axes[5])
plot_naive_bayes(axes[6])
plot_gradient_boosting(axes[7])
plot_ann(axes[8])
plot_lda(axes[9])
 
fig.suptitle(
    "10 Algoritmos de Aprendizaje Supervisado — Ejemplos con gráficas\n"
    "Docente: Peralta  |  Alumno: Cárdenas Infante Ricardo  |  ITCM 2026",
    fontsize=14, fontweight="bold", y=1.01
)
 
plt.tight_layout(pad=2.5)
plt.show()
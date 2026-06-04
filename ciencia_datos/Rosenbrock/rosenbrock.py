import numpy as np
import pandas as pd
import time
import plotly.express as px

# ============================================
# FUNCIÓN ROSENBROCK
# ============================================
def rosenbrock(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

# ============================================
# ALGORITMO GENÉTICO
# ============================================
def runGA(Pm, Pc, numPopulation, maxIter):
    
    population = np.random.uniform(-2, 2, (numPopulation, 2))
    bestFitness = float("inf")
    
    for _ in range(maxIter):
        
        fitness = np.apply_along_axis(rosenbrock, 1, population)
        
        # Selección (torneo)
        newPopulation = population.copy()
        for i in range(numPopulation):
            i1, i2 = np.random.randint(0, numPopulation, 2)
            winner = i1 if fitness[i1] < fitness[i2] else i2
            newPopulation[i] = population[winner]
        
        # Cruza
        for i in range(0, numPopulation, 2):
            if i+1 < numPopulation and np.random.rand() < Pc:
                alpha = np.random.rand()
                p1, p2 = newPopulation[i].copy(), newPopulation[i+1].copy()
                newPopulation[i] = alpha * p1 + (1 - alpha) * p2
                newPopulation[i+1] = alpha * p2 + (1 - alpha) * p1
        
        # Mutación
        for i in range(numPopulation):
            if np.random.rand() < Pm:
                newPopulation[i] += np.random.normal(0, 0.1, 2)
        
        population = newPopulation
        fitness = np.apply_along_axis(rosenbrock, 1, population)
        bestFitness = min(bestFitness, np.min(fitness))
    
    return bestFitness

# ============================================
# EXPERIMENTOS
# ============================================
experiments = [
    ("Exp1", 0.10, 0.80, 20, 100),
    ("Exp2", 0.10, 0.80, 30, 200),
    ("Exp3", 0.10, 0.90, 50, 300),
    ("Exp4", 0.15, 0.90, 50, 300),
    ("Exp5", 0.15, 0.90, 80, 500),
]

results = []

# ============================================
# EJECUCIÓN (30 semillas)
# ============================================
for name, Pm, Pc, pop, iters in experiments:
    
    errores = []
    tiempos = []
    
    for seed in range(1, 31):
        np.random.seed(seed)
        
        start = time.time()
        best = runGA(Pm, Pc, pop, iters)
        end = time.time()
        
        errores.append(best)
        tiempos.append(end - start)
    
    results.append({
        "Configuracion": name,
        "Mediana_Error": np.median(errores),
        "IQR_Error": np.percentile(errores, 75) - np.percentile(errores, 25),
        "Mediana_Tiempo": np.median(tiempos),
        "IQR_Tiempo": np.percentile(tiempos, 75) - np.percentile(tiempos, 25),
    })

df = pd.DataFrame(results)
print(df)

# Guardar CSV
df.to_csv("resultados_rosenbrock.csv", index=False)

# ============================================
# DASHBOARD (PLOTLY)
# ============================================

fig1 = px.bar(df, x="Configuracion", y="Mediana_Error",
              title="Mediana del Error por Experimento")

fig2 = px.bar(df, x="Configuracion", y="Mediana_Tiempo",
              title="Mediana del Tiempo por Experimento")

fig3 = px.bar(df, x="Configuracion", y="IQR_Error",
              title="IQR del Error (Estabilidad)")

fig4 = px.bar(df, x="Configuracion", y="IQR_Tiempo",
              title="IQR del Tiempo (Estabilidad)")

fig1.show()
fig2.show()
fig3.show()
fig4.show()
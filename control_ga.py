# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 21:20:25 2023

@author: Emanuel G. Muñoz
"""
import time
import numpy as np
from mochila_ga_EMAX import generate_population, valor_real, binario_a_decimal,genDecimal, cal_fitness,selection, crossover, mutation,optimize, plot_fitness_hist,optimize1,experimento
import pandas as pd

population=np.array(generate_population(300,120))

tm_gen=[0,6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120]

np.array(population[0])

print(genDecimal(population[0],tm_gen))



#weight = np.random.randint(3, 7, size = 20)
#value = np.random.randint(15, 35, size = 20)
#knapsack_threshold = 110    #Maximum weight that the bag of thief can hold 

weight=np.array([3,5,7,6,4,7,5,3,3,7,6,6,4,7,5,7,7,6,7,6])
value=np.array([15,17,21,19,16,18,20,16,15,35,20,29,20,35,20,25,36,23,22,30])
knapsack_threshold = 110  

crossover_rate=0.9
mutation_rate=0.09
pop_size= (population.shape[0], population.shape[1])
num_generations = 100


start = time.perf_counter()
parameters, fitness_history,max_fitness,population1,fitness= optimize(weight, value, population, pop_size, num_generations, knapsack_threshold,mutation_rate,crossover_rate,tm_gen)
end = time.perf_counter()

print(f"Time taken is {end - start}")

fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
fitness_history_max = [np.max(fitness) for fitness in fitness_history]
#np.mean(fitness_history_mean)
np.max(fitness_history_mean)


print(genDecimal(parameters[0],tm_gen))


plot_fitness_hist(num_generations,fitness_history)



X=[[50,200,0.01,0.5],[60,300,0.05,0.7],[70,350,0.09,0.9]] 
X=[[10,10,0.01,0.5],[20,20,0.05,0.7],[30,30,0.09,0.9]]            
#GENERACIÓN, POBLACIÓN, MUTACIÓN, CRUCE

bists=120

start = time.perf_counter()
df_1=experimento(weight, value,bists,X,knapsack_threshold,tm_gen)
end = time.perf_counter()

print(f"Time taken is {end - start}")

start = time.perf_counter()
df_2=experimento(weight, value,bists,X,knapsack_threshold,tm_gen)
end = time.perf_counter()

print(f"Time taken is {end - start}")


start = time.perf_counter()
df_3=experimento(weight, value,bists,X,knapsack_threshold,tm_gen)
end = time.perf_counter()

print(f"Time taken is {end - start}")

start = time.perf_counter()
df_4=experimento(weight, value,bists,X,knapsack_threshold,tm_gen)
end = time.perf_counter()

print(f"Time taken is {end - start}")

df_concat = pd.concat([df_1, df_2[["max fit", "mean fit"]],df_3[["max fit", "mean fit"]],df_4[["max fit", "mean fit"]]], axis=1)
df_concat.shape[1]

print(df_concat)

# Guardar el DataFrame en un archivo CSV
df_concat.to_csv("datos.csv", index=False, decimal=".")

# Leer el archivo CSV y cargarlo en un nuevo DataFrame
df_nuevo = pd.read_csv("datos.csv")

# Imprimir el nuevo DataFrame
print(df_nuevo)
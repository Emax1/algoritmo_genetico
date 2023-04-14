# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 21:06:04 2023

@author: Emanuel G. Muñoz
"""

import time
import numpy as np
import pandas as pd
import random as rd
from random import randint
import random
import matplotlib.pyplot as plt
import pandas as pd



def generate_population(individuos,bits):
	population = []
	for _ in range(individuos):
		genes = [0, 1]
		chromosome = []
		for _ in range(bits):
			chromosome.append(random.choice(genes))
		population.append(chromosome)
	#print("Generated a random population of size", individuos)
	return population


def valor_real(v_decimal):
    real=round(v_decimal*37/64)
    return real


def binario_a_decimal(binario):
    decimal = 0
    for i in range(len(binario)):
        decimal += binario[i] * 2**(len(binario)-i-1)
    return decimal


def genDecimal(cromosoma,tm_gen):
    valores_r=[]
    #len(indice)
    for i in range(20):
        bina=cromosoma[tm_gen[i]:tm_gen[i+1]]
        decimal = binario_a_decimal(bina)
        vreal= valor_real(decimal)
        valores_r.append(vreal)
    return valores_r


def cal_fitness(weight, value, population, threshold,tm_gen):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        population1=np.array(genDecimal(population[i],tm_gen))
        S1 = np.sum(population1 * value)
        S2 = np.sum(population1 * weight)
        if S2 <= threshold:
            fitness[i] = S1
        else :
            fitness[i] = 0 
    return fitness.astype(int) 


def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i,:] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents


def crossover(parents, num_offsprings,crossover_rate):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    #crossover_rate = 0.8
    i=0
    while (parents.shape[0] < num_offsprings):
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        offsprings[i,0:crossover_point] = parents[parent1_index,0:crossover_point]
        offsprings[i,crossover_point:] = parents[parent2_index,crossover_point:]
        i=+1
    return offsprings   


def mutation(offsprings,mutation_rate):
    mutants = np.empty((offsprings.shape))
    #mutation_rate = 0.4
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i,:] = offsprings[i,:]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0,offsprings.shape[1]-1)    
        if mutants[i,int_random_value] == 0 :
            mutants[i,int_random_value] = 1
        else :
            mutants[i,int_random_value] = 0
    return mutants 

def optimize(weight, value, population, pop_size, num_generations, threshold,mutation_rate,crossover_rate,tm_gen):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents 
    for i in range(num_generations):
        fitness = cal_fitness(weight, value, population, threshold,tm_gen)
        fitness_history.append(fitness)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings,crossover_rate)
        mutants = mutation(offsprings,mutation_rate)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
        
    print('Last generation: \n{}\n'.format(population)) 
    fitness_last_gen = cal_fitness(weight, value, population, threshold,tm_gen)      
    print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0],:])
    return parameters, fitness_history,max_fitness,population,fitness

def optimize1(weight, value, population, pop_size, num_generations, threshold,mutation_rate,crossover_rate,tm_gen):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents 
    for i in range(num_generations):
        fitness = cal_fitness(weight, value, population, threshold,tm_gen)
        fitness_history.append(fitness)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings,crossover_rate)
        mutants = mutation(offsprings,mutation_rate)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
        
    #print('Last generation: \n{}\n'.format(population)) 
    fitness_last_gen = cal_fitness(weight, value, population, threshold,tm_gen)      
    #print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0],:])
    fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
    fitness_history_max = [np.max(fitness) for fitness in fitness_history]
    max_fit=np.max(fitness_history_max)
    mean_fit=np.mean(fitness_history_mean)
    return max_fit, mean_fit

def experimento(weight, value,bists,X,knapsack_threshold,tm_gen):
    r=0
    df = pd.DataFrame(columns=["GENERACION", "POBLACION", "MUTACION", "CRUCE", "max fit","mean fit"])
    for i in range(3):
        for j in range(3):
            for k in range(3): 
                for l in range(3):  
                    max_fit, mean_fit= optimize1(weight, value, np.array(generate_population(X[j][1],bists)), (X[j][1],bists), X[i][0], knapsack_threshold,X[k][2],X[l][3],tm_gen)
                    print(r+1,X[i][0],X[j][1],X[k][2],X[l][3],max_fit,mean_fit)
                    df.loc[r]=[X[i][0],X[j][1],X[k][2],X[l][3],max_fit,mean_fit]
                    r=r+1
    return df
    
    
def plot_fitness_hist(num_generations,fitness_history):
    fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
    fitness_history_max = [np.max(fitness) for fitness in fitness_history]
    plt.plot(list(range(num_generations)), fitness_history_mean, label = 'Mean Fitness')
    plt.plot(list(range(num_generations)), fitness_history_max, label = 'Max Fitness')
    plt.legend()
    plt.title('Fitness through the generations')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()
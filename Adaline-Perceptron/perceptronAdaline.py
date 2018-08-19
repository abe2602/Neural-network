"""
Bruno Bacelar Abe  9292858
Exercício 1 - Perceptron
"""

import numpy
import os
import glob

RATE_LEARNING = 0.5 
NUM_MAX_INTERACTIONS = 100
MATRIX_SIZE = 5
NUM_EXAMPLES_TEST = 12

#Encontra os resultados esperados
def find_right_results():
	results = numpy.loadtxt("correctResult.txt")
	return results

def find_example_training(path):
	training_examples = []

	#Lê os arquivos de teste e transforma-os em vetor
	for filename in glob.glob(path):
		training_examples.append(numpy.reshape(numpy.loadtxt(filename), newshape = (1, MATRIX_SIZE*MATRIX_SIZE)))

	#print(training_examples[0])
	return training_examples

def activation_function(x):
    if x >= 0:
        return 1
    else:
        return -1

def train(results, weights_array, training_examples):
	i = 0
	changed = False	

	#Treina o neurônio
	while (i < NUM_MAX_INTERACTIONS and changed == False):

		changed = False

		for j in range(0, NUM_EXAMPLES_TEST):
			sumNumbers = numpy.multiply(training_examples[j], weights_array[1::])
			aux = weights_array[0] + numpy.sum(sumNumbers) #Faz o somatório para colocar na função de ativação
			aux = activation_function(aux) #Coloca o resultado na função de ativação
			error = results[j] - aux #Compara o resultado obtido com o esperado

			if error != 0: #Se houve algum erro em algum dos casos do teste
				changed = True 

			#Atualiza os pesos
			weights_array[0] += (0.5*error)
			weights_array[1::] = weights_array[1::] + (0.5*error*training_examples[j])

		i+=1

	numpy.savetxt("weights.txt", weights_array)

	return 0;

def initialize_weights():
	weights_array = [] 
	cont = 0
	
	#Lê os pesos do arquivo de pesos, caso esteja vazio, ainda não houve treinamento e inicia tudo com zero
	if os.stat("weights.txt").st_size == 0:
		print("Arquivo vazio! Iniciando com zero todos os pesos")
		weights_array = numpy.zeros((MATRIX_SIZE*MATRIX_SIZE) + 1) #vetor dos pesos inicializado com zero
	else:
		print("Pesos encontrados! Atribuindo valores")
		weights_array = numpy.loadtxt("weights.txt") #lê os pesos do arquivo de pesos
		cont = 1

	return weights_array, cont

def neuronion_training():
	results = [] #vetor dos resultados esperados
	training_examples = [] #vetor com os dados do treino
	weights_array = [] #vetor de pesos
	cont = 0

	#Encontra/Inicializa os pesos
	weights_array, cont = initialize_weights()
	
    #Encontra o valor correto dos testes
	results = find_right_results()

	#Coloca os testes em memória
	training_examples = find_example_training('treino/*.txt')	

	#Começa a treinar
	train(results, weights_array, training_examples)

def neuronion_classifier():
    weights_array = numpy.zeros((MATRIX_SIZE*MATRIX_SIZE+1))
    weights_array = numpy.loadtxt("weights.txt")
    results_array = numpy.loadtxt("correctResultTeste.txt")
    test = []
    i = 0
    totalAcertos = 0
    totalErros = 0

    #Classifica os testes
    for filename in glob.glob(os.path.join('teste/', '*.txt')):
        with open(filename) as f:
        	test.append(numpy.reshape(numpy.loadtxt(filename), newshape = (1, MATRIX_SIZE*MATRIX_SIZE))) #transforma a matrix em vetor
        	sumNumbers = numpy.multiply(test, weights_array[1::])
        	aux = weights_array[0] + numpy.sum(sumNumbers) #Faz o somatório para colocar na função de ativação
        	aux = activation_function(aux)

        	error = results_array[i] - aux
        	test = []

        	if error == 0:
        		totalAcertos+=1
        	else:
        		totalErros+=1

        i+=1
    
    print("Foram classificados ", i, "arquivos")
    print("Acertos: ", totalAcertos, "Erros: ", totalErros)

neuronion_training()
neuronion_classifier()
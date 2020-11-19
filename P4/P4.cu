/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* PRACTICA 2: "Suma De Matrices Paralela"
* >> Arreglar for en __global__
* >> Pasar numElem como argumento
*
* AUTOR: Ivanes
*/
///////////////////////////////////////////////////////////////////////////
// Includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Defines
#define RAN_MIN 1
#define RAN_MAX 9
// Bloques - Hilos
#define MAX_HILOS 10
#define MAX_BLOQUES 512

// Declaracion de funciones
int numHilos()
{
	int numHilos;

	// Saca num hilos, funcion CUDA
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	// deviceProp.maxThreadsPerBlock;
	// deviceProp.maxGridSize[0];
	int maxValores = MAX_HILOS*MAX_BLOQUES;


	//
	printf("\n***********************************************************************\n\n");
	printf("> Nombre Dispositivos: %s\n", deviceProp.name);
	printf("> Capacidad de Computo: %d.%d\n", deviceProp.major, deviceProp.minor);
	printf("> Numero de MultiProcesadores: %d \n", deviceProp.multiProcessorCount);
	printf("> Numero de Nucleos (Arq. PASCAL): %d \n", 64);
	printf("> Maximo de hilos por eje en bloque\n");
	printf(" \t[x -> %d]\n \t[y -> %d]\n \t[z -> %d]\n",deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("> Maximo de bloques por eje\n");
	printf(" \t[x -> %d]\n \t[y -> %d]\n \t[z -> %d]\n",deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("\n***********************************************************************\n");

	printf("\nEl numero maximo de elementos del array es: %d valores\n", maxValores);
	do {
		printf("\n\nCuantos elementos quieres que tenga los vectores: ");
		scanf("%d", &numHilos);
		getchar();
	} while ((numHilos > maxValores) || (numHilos <= 0));

	return numHilos;
}

__global__
void reverseMatriz(int *dev_matriz, int *dev_matriz_reverse, int *dev_matriz_resultado, int numElem)
{
	// Crea la matriz inversa
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	dev_matriz_reverse[id] = dev_matriz[numElem - 1 - id];

	// Suma las matrices
	dev_matriz_resultado[id] = dev_matriz[id] + dev_matriz_reverse[id];
}

// MAIN: Rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// Declaracion
	int *hst_matriz;
	int *hst_matriz_reverse;
	int *hst_matriz_resultado;
	int *dev_matriz;
	int *dev_matriz_reverse;
	int *dev_matriz_resultado;

	// Saca numero de hilos y pregunta cuantos elementos quiere en el array. Pone el número de bloques a usar, 1 en este caso
	int numElem = numHilos();
	int numBlock = numElem/MAX_HILOS;
	
	// Obtencion del numero del bloques
	if(numElem%MAX_HILOS != 0)
		numBlock++;
	printf("Lanzamos %d valores en %d bloques de %d hilos", numElem, numBlock, MAX_HILOS);


	// Reserva en el host
	hst_matriz = (int*)malloc(numElem * sizeof(int));
	hst_matriz_reverse = (int*)malloc(numElem * sizeof(int));
	hst_matriz_resultado = (int*)malloc(numElem * sizeof(int));

	// Reserva en el device
	cudaMalloc( &dev_matriz, numElem * sizeof(int));
	cudaMalloc( &dev_matriz_reverse, numElem * sizeof(int));
	cudaMalloc( &dev_matriz_resultado, numElem * sizeof(int));

	// Insertamos valores random en la matriz
	srand((int)time(NULL));
	for (int i = 0; i < numElem; i++)
	{
		hst_matriz[i] = RAN_MIN + rand() % RAN_MAX;
	}

	// Pasamos el array al device y le damos la vuelta
	cudaMemcpy(dev_matriz, hst_matriz, numElem * sizeof(int), cudaMemcpyHostToDevice);
	reverseMatriz <<< numBlock, MAX_HILOS>>>(dev_matriz, dev_matriz_reverse, dev_matriz_resultado, numElem);

	// Check de errores
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	// Pasamos el array inverso a la cpu
	cudaMemcpy(hst_matriz_reverse, dev_matriz_reverse, numElem * sizeof(int), cudaMemcpyDeviceToHost);

	// Pasamos el resultado a la cpu
	cudaMemcpy(hst_matriz_resultado, dev_matriz_resultado, numElem * sizeof(int), cudaMemcpyDeviceToHost);

	// Muestra contenido de arrays y resultado
	printf("\n\nMatriz: \n");
	for (int i = 0; i < numElem; i++)
		printf("%d ", hst_matriz[i]);

	printf("\n\nMatriz Inversa: \n");
	for (int i = 0; i < numElem; i++)
		printf("%d ", hst_matriz_reverse[i]);

	printf("\n\nMatriz Resultado: \n");
	for (int i = 0; i < numElem; i++)
		printf("%d ", hst_matriz_resultado[i]);

	free(hst_matriz);
	free(hst_matriz_reverse);
	free(hst_matriz_resultado);

	cudaFree(dev_matriz);
	cudaFree(dev_matriz_reverse);
	cudaFree(dev_matriz_resultado);

	// salida
	time_t fecha;
	time(&fecha);
	printf("\n\n***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}

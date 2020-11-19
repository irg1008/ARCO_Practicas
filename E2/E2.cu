/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* PRACTICA 2: "Ordenación de Array De Menor a Mayor".
* >> TODO => Finalizado.
*
* AUTOR: Iván Ruiz Gázquez e Iván Maeso Adrián.
*/
///////////////////////////////////////////////////////////////////////////
// Includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Defines
#define RAN_MIN 1
#define RAN_MAX 50

// Declaracion de funciones
void cudaDev()
{
	// Saca num hilos, funcion CUDA
	int dev;
	cudaGetDevice(&dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

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
}

__global__
void ordenarArray(int *dev_desordenado, int *dev_ordenado, int elem)
{
	int myID = threadIdx.x;
	int rango = 0;

	for(int i=0; i<elem; i++) {
		if((dev_desordenado[myID] > dev_desordenado[i]) && (myID != i))
			rango++;
		if(dev_desordenado[myID] == dev_desordenado[i] && myID > i)
			rango++;
	}

	dev_ordenado[rango] = dev_desordenado[myID];
}

// MAIN: Rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	//Eventos
	cudaEvent_t start;
	cudaEvent_t stop;

	// Creacion de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Marca de inicio
	cudaEventRecord(start, 0);

	// Declaracion
	int *hst_desordenado;
	int *hst_ordenado;
	int *dev_desordenado;
	int *dev_ordenado;

	// Elementos
	int elem;

	// Llama a la función Cuda que devuelve info
	cudaDev();

	// Pregunta número de elemetos
	do {
		printf("\n\nNumero de elementos (MAX=1024): ");
		scanf("%d", &elem);
		getchar();
	} while (elem<=0 || elem>1024);

	// Dimensiones del kernel
	dim3 Nbloques(1);
	dim3 hilosB(elem);

	// Reserva en el host
	hst_ordenado = (int*)malloc(elem * sizeof(int));
	hst_desordenado = (int*)malloc(elem * sizeof(int));

	// Reserva en el device
	cudaMalloc( &dev_ordenado, elem * sizeof(int));
	cudaMalloc( &dev_desordenado, elem * sizeof(int));

	
	// Insertamos valores random en la matriz
	srand((int)time(NULL));
	for (int i = 0; i < elem; i++)
	{
		hst_desordenado[i] = RAN_MIN + rand() % RAN_MAX;
	}

	// Pasamos el array al device y le damos la vuelta
	cudaMemcpy(dev_desordenado, hst_desordenado, elem * sizeof(int), cudaMemcpyHostToDevice);
	ordenarArray <<<Nbloques, hilosB>>>(dev_desordenado, dev_ordenado, elem);

	// Check de errores
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	// Pasamos el resultado a la cpu
	cudaMemcpy(hst_ordenado, dev_ordenado, elem * sizeof(int), cudaMemcpyDeviceToHost);

	// Muestra contenido de arrays y resultado
	printf("\n\nMatriz Desordenada: \n");
	printf("*********************\n");
	for (int i = 0; i < elem; i++) {
		printf("%d ", hst_desordenado[i]);
	}

	printf("\n\nMatriz Ordenada: \n");
	printf("*********************\n");
	for (int i = 0; i < elem; i++) {
		printf("%d ", hst_ordenado[i]);
	}
	
	// Marca de final
	cudaEventRecord(stop, 0);

	// Sincronizacion CPU-GPU
	cudaEventSynchronize(stop);

	// Calculo del tiempo
	float tiempoTrans;
	cudaEventElapsedTime(&tiempoTrans, start, stop);
	printf("\n\n\n> Tiempo de ejecuccion: %f ms\n", tiempoTrans);

	// Liberacion de recursos
	free(hst_desordenado);
	free(hst_ordenado);
	cudaFree(dev_desordenado);
	cudaFree(dev_ordenado);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Salida
	time_t fecha;
	time(&fecha);
	printf("\n\n***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}

﻿/*
* ARQUITECTURA DE COMPUTADORES
* 2� Grado en Ingenieria Informatica
*
* PRACTICA 2: "Ordenaci�n de Array De Menor a Mayor".
* >> TODO => Finalizado
*
* AUTOR: Iv�n Ruiz G�zquez e Iv�n Maeso Adri�n.
*/

///////////////////////////////////////////////////////////////////////////
// Includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#ifdef __linux__
#include <sys/time.h>
typedef struct timeval event;
#else
#include <windows.h>
typedef LARGE_INTEGER event;
#endif

// Defines
#define RAN_MIN 1
#define RAN_MAX 50

// Func
void cudaDev();
void calcularTiempos(int valorElem);
float calcularCPU(int valorElem);
float calcularGPU(int valorElem);
__global__ void ordenarArray(int *dev_desordenado, int *dev_ordenado, int elem);

// Funciones Temporizador CPU
__host__ void setEvent(event *ev)
	/* Descripcion: Genera un evento de tiempo */
{
#ifdef __linux__
	gettimeofday(ev, NULL);
#else
	QueryPerformanceCounter(ev);
#endif
}
__host__ double eventDiff(event *first, event *last)
	/* Descripcion: Devuelve la diferencia de tiempo (en ms) entre dos eventos */
{
#ifdef __linux__
	return
		((double)(last->tv_sec + (double)last->tv_usec/1000000)-
		(double)(first->tv_sec + (double)first->tv_usec/1000000))*1000.0;
#else
	event freq;
	QueryPerformanceFrequency(&freq);
	return ((double)(last->QuadPart - first->QuadPart) / (double)freq.QuadPart) * 1000.0;
#endif
}

// MAIN: Rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// Variables
	int valorElem = 32;
	int valorMax = 4096;

	// Llama a la función Cuda que devuelve info
	cudaDev();
	// Llama tantas veces como número de columnas queremos.
	for(valorElem; valorElem<=valorMax; valorElem *=2)
		calcularTiempos(valorElem);

	// Salida
	time_t fecha;
	time(&fecha);
	printf("\n\n***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	fflush(stdin);
	getchar();

	return 0;
}

void cudaDev() {
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

void calcularTiempos(int valorElem) {
	// Variables
	float tiempoGPU = 0, tiempoCPU = 0, ganancia = 0;
	int numHilos = 32, numBloques = valorElem/32; // Optimizar
	int numIteracciones = 5;

	// Salida por pantalla
	printf("\n\n***************************************************");
	printf("\n> Numero de elementos: %d", valorElem);
	printf("\n> Lanzamiento con bloques de %d hilos y %d bloques", numHilos, numBloques);
	printf("\n***************************************************");

	printf("\n\nEjecución GPU");
	for(int i=0; i<numIteracciones; i++) {
		float temp = calcularGPU(valorElem);
		printf("\n Iteracción numero %d. Tiempo = %.6f ms", i+1, temp);
		tiempoGPU += temp;
	}
	tiempoGPU /= numIteracciones;
	printf("\n> Tiempo medio de ejecución GPU: %.6f ms",tiempoGPU);

	printf("\n\nEjecución CPU");
	for(int i=0; i<numIteracciones; i++) {
		float temp = calcularCPU(valorElem);
		printf("\n Iteracción numero %d. Tiempo = %.6f ms", i+1, temp);
		tiempoCPU += temp;
	}
	tiempoCPU /= numIteracciones;
	printf("\n> Tiempo medio de ejecución CPU: %.6f ms",tiempoCPU);

	ganancia = tiempoCPU/tiempoGPU;
	printf("\n\n> Ganancia GPU/CPU: %.6f",ganancia);
	printf("\n\n***************************************************");
}

float calcularCPU(int elem) {
	int *hst_desordenado;
	int *hst_ordenado;
	event start; // variable para almacenar el evento de tiempo inicial.
	event stop; // variable para almacenar el evento de tiempo final.
	double tiempo_ms;

	// Reserva en el host
	hst_desordenado = (int*)malloc(elem * sizeof(int)); // Ver si se puede cambiar por void
	hst_ordenado = (int*)malloc(elem * sizeof(int));

	// Insertamos valores random en la matriz
	srand((int)time(NULL));
	for (int i = 0; i < elem; i++)
		hst_desordenado[i] = RAN_MIN + rand() % RAN_MAX;

	/****************************/
	// Iniciamos el contador
	setEvent(&start); // marca de evento inicial
	// Ordenamos el array
	for(int i=0; i<elem; i++) {
		int rango = 0;
		for(int j=0; j<elem; j++)
			if(hst_desordenado[j] >= hst_desordenado[i] && j > i)
				rango++;
		hst_ordenado[rango] = hst_desordenado[i];
	}
	 // Paramos el contador
	 setEvent(&stop);// marca de evento final
	 /****************************/

	 // Intervalos de tiempo
	 tiempo_ms = eventDiff(&start, &stop); // diferencia de tiempo en ms
	
	// Liberacion de recursos
	free(hst_desordenado);
	free(hst_ordenado);

	return tiempo_ms;
}

float calcularGPU(int elem) {

	// Declaracion
	int *hst_desordenado;
	int *hst_ordenado;
	int *dev_desordenado;
	int *dev_ordenado;

	//Eventos
	cudaEvent_t start;
	cudaEvent_t stop;

	// Creacion de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Reserva en el host
	hst_ordenado = (int*)malloc(elem * sizeof(int));
	hst_desordenado = (int*)malloc(elem * sizeof(int));

	// Reserva en el device
	cudaMalloc(&dev_ordenado, elem * sizeof(int));
	cudaMalloc(&dev_desordenado, elem * sizeof(int));

	// Insertamos valores random en la matriz
	srand((int)time(NULL));
	for (int i = 0; i < elem; i++)
		hst_desordenado[i] = RAN_MIN + rand() % RAN_MAX;

	// Pasamos del host al device
	cudaMemcpy(dev_desordenado, hst_desordenado, elem * sizeof(int), cudaMemcpyHostToDevice);

	/****************************/
	// Marca de inicio
	cudaEventRecord(start, 0);
	//Ejecutamos
	ordenarArray <<<elem/32, 32>>>(dev_desordenado, dev_ordenado, elem);
	// Marca de final
	cudaEventRecord(stop, 0);
	/****************************/

	// Pasamos el resultado a la cpu
	cudaMemcpy(hst_ordenado, dev_ordenado, elem * sizeof(int), cudaMemcpyDeviceToHost);

	// Sincronizacion CPU-GPU
	cudaEventSynchronize(stop);

	// Calculo del tiempo
	float tiempoTrans;
	cudaEventElapsedTime(&tiempoTrans, start, stop);

	// Liberacion de recursos
	free(hst_desordenado);
	free(hst_ordenado);
	cudaFree(dev_desordenado);
	cudaFree(dev_ordenado);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return tiempoTrans;
}

__global__ void ordenarArray(int *dev_desordenado, int *dev_ordenado, int elem)
{
	int myID = threadIdx.x + blockIdx.x * blockDim.x;
	int rango = 0;

	for(int i=0; i<elem; i++)
		if(dev_desordenado[myID] >= dev_desordenado[i] && myID > i)
			rango++;

	dev_ordenado[rango] = dev_desordenado[myID];
}

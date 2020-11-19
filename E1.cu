/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* PRACTICA 2: "Reducción Paralela"
* >> TODO => Añadir comprobacion de potencia de 2
*
* AUTOR: Iván Ruiz Gázquez
*/
///////////////////////////////////////////////////////////////////////////
// Includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Declaracion de funciones
void cudaDev()
{
	// Saca num hilos, funcion CUDA
	int dev = 0;
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
void logNep(float *dev_datos, float *dev_suma, int elem)
{
	int myID = threadIdx.x;

	float denominador = myID + 1;

	int j = 1;
	if(myID%2 != 0)
		j = -1;

	// Insertamos valores
	dev_datos[myID] = (1/denominador)*j;

	// Sincronizamos antes de seguir
	__syncthreads();
	
	// REDUCCION PARALELA
	int salto = elem / 2;
	
	// Realizamos log2(N) iteraciones
	while (salto > 0)
	{
		// En cada paso solo trabajan la mitad de los hilos
		if (myID < salto)
		{
			dev_datos[myID] = dev_datos[myID] + dev_datos[myID + salto];
		}
		// Sincronizamos los hilos
		__syncthreads();
		salto = salto / 2;
	}
	// El hilo no.'0' escribe el resultado final
	if (myID == 0)
	{
		dev_suma[0] = dev_datos[0];
	}
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
	float *hst_suma;
	float *dev_suma;
	float *dev_datos;

	// Elementos
	int elem;
	float logDos = 0.6931472;

	// Llama a la función Cuda que devuelve info
	cudaDev();

	// Pregunta número de elemetos
	printf("Numero de hilos (potencia de dos y menor o igual que 1024)");
	do {
		printf("\n\nNumero de hilos: ");
		scanf("%d", &elem);
		getchar();
	} while (elem%2 != 0 && elem<=1024); 

	// Dimensiones del kernel
	dim3 Nbloques(1);
	dim3 hilosB(elem);

	// Reserva en el host
	hst_suma = (float*)malloc(1 * sizeof(float));

	// Reserva en el device
	cudaMalloc( &dev_suma, 1 * sizeof(float));
	cudaMalloc( &dev_datos, elem * sizeof(float));

	// Pasamos el array al device y le damos la vuelta
	logNep <<<Nbloques, hilosB>>>(dev_datos, dev_suma, elem);

	// Check de errores
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	// Pasamos el resultado a la cpu
	cudaMemcpy(hst_suma, dev_suma, 1 * sizeof(float), cudaMemcpyDeviceToHost);

	// Muestra contenido de arrays y resultado
	printf("\nValor Calculado: \n");
	printf("*****************\n");
	printf(">%.6f\n", hst_suma[0]);


	printf("\nValor Real: \n");
	printf("*********************\n");
	printf(">%f\n", logDos);

	printf("\nError Relativo: \n");
	printf("*********************\n");
	printf(">%.7f%%\n", (1-(hst_suma[0]/logDos))*100);
	
	// Marca de final
	cudaEventRecord(stop, 0);

	// Sincronizacion CPU-GPU
	cudaEventSynchronize(stop);

	// Calculo del tiempo
	float tiempoTrans;
	cudaEventElapsedTime(&tiempoTrans, start, stop);
	printf("\n> Tiempo de ejecuccion: %f ms\n", tiempoTrans);

	// Liberacion de recursos
	free(hst_suma);
	cudaFree(dev_datos);
	cudaFree(dev_suma);
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

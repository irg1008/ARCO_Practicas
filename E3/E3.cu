/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* PRACTICA 2: "Crea un mapa de bits en blanco y negro -> Ajedrez".
* >> TODO => Finalizado.
*
* AUTOR: Iván Ruiz Gázquez e Iván Maeso Adrián.
*/
///////////////////////////////////////////////////////////////////////////
// Includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "gpu_bitmap.h"

// Defines
#define ANCHO 256 // Dimension horizontal
#define ALTO 256 // Dimension vertical

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

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void board( unsigned char *imagen )
{
	// coordenada horizontal
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	// coordenada vertical
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// coordenada global de cada hilo
	int posicion = x + y * blockDim.x * gridDim.x;
	// cada hilo obtiene la posicion de un pixel
	int pixel = posicion * 4;

	if((blockIdx.x/2+blockIdx.y/2)%2 != 0) {
		imagen[pixel + 0] = 255; // canal R
		imagen[pixel + 1] = 255; // canal G
		imagen[pixel + 2] = 255; // canal B
		imagen[pixel + 3] = 0; // canal alfa
	} else {
		imagen[pixel + 0] = 0; // canal R
		imagen[pixel + 1] = 0; // canal G
		imagen[pixel + 2] = 0; // canal B
		imagen[pixel + 3] = 0; // canal alfa
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
	RenderGPU foto(ANCHO, ALTO);

	// Llama a la función Cuda que devuelve info
	cudaDev();

	// Tamaño en bytes
	size_t size = foto.image_size();

	// Asignacion a la memoria del host
	unsigned char *host_bitmap = foto.get_ptr();
	
	// Reserva en el device
	unsigned char *dev_bitmap;
	cudaMalloc( (void**)&dev_bitmap, size );

	// Lanzamos un kernel con bloques de 256 hilos (16x16)
	dim3 hilosB(16,16);

	// Calculamos el numero de bloques necesario
	dim3 Nbloques(ANCHO/16, ALTO/16);
	
	// Generamos el bitmap
	board<<<Nbloques,hilosB>>>( dev_bitmap );

	// Recogemos el bitmap desde la GPU para visualizarlo
	cudaMemcpy( host_bitmap, dev_bitmap, size, cudaMemcpyDeviceToHost );
	
	// Marca de final
	cudaEventRecord(stop, 0);

	// Sincronizacion CPU-GPU
	cudaEventSynchronize(stop);

	// Calculo del tiempo
	float tiempoTrans;
	cudaEventElapsedTime(&tiempoTrans, start, stop);
	printf("\n\n\n> Tiempo de ejecuccion: %f ms\n", tiempoTrans);

	// Salida
	time_t fecha;
	time(&fecha);
	printf("\n\n***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("\n...pulsa [ESC] para finalizar...");
	foto.display_and_exit();
	getchar();
	return 0;
}

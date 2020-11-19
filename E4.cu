/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* PRACTICA 2: Cambia de color a escala de grises.
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

// Declaracion de funciones
// HOST: funcion llamada desde el host y ejecutada en el host
__host__ void leerBMP_RGBA(const char* nombre, int* w, int* h, unsigned char **imagen);
// Funcion que lee un archivo de tipo BMP:
// -> entrada: nombre del archivo
// <- salida : ancho de la imagen en pixeles
// <- salida : alto de la imagen en pixeles
// <- salida : puntero al array de datos de la imagen en formato RGBA

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

	float color = imagen[pixel + 0]*0.299 + imagen[pixel + 1]*0.587 + imagen[pixel + 2]*0.114;

	imagen[pixel + 0] = color; // canal R
	imagen[pixel + 1] = color; // canal G
	imagen[pixel + 2] = color; // canal B
	imagen[pixel + 3] = 0; // canal alfa
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

	// Llama a la función Cuda que devuelve info
	cudaDev();

	/////////////////////////////////////////////////////////////////////////////
	// Leemos el archivo BMP
	unsigned char *host_color;
	int ancho, alto;
	leerBMP_RGBA("imagen.bmp", &ancho, &alto, &host_color);
	/////////////////////////////////////////////////////////////////////////////

	// Declaracion del bitmap RGBA
	RenderGPU foto(ancho, alto);

	// Tamaño en bytes de una imagen de tipo RGBA
	size_t img_size = foto.image_size();

	// puntero al framebuffer vinculado con la estructura RenderGPU
	unsigned char *host_bitmap = foto.get_ptr();

	// Copiamos los datos de la imagen al framebuffer:
	cudaMemcpy(host_bitmap, host_color, img_size, cudaMemcpyHostToHost);

	// Reserva en el device
	unsigned char *dev_bitmap;
	cudaMalloc((void**)&dev_bitmap, img_size);

	// Copiar imagen de cpu a gpu
	cudaMemcpy(dev_bitmap, host_bitmap, img_size, cudaMemcpyHostToDevice);

	// Lanzamos un kernel con bloques de 256 hilos (16x16)
	dim3 hilosB(25, 25);

	// Calculamos el numero de bloques necesario
	dim3 Nbloques(ancho/25, alto/25);

	// Llama kernel
	board<<<Nbloques,hilosB>>>(dev_bitmap);

	// Copiar imagen de gpu a cpu
	cudaMemcpy(host_bitmap, dev_bitmap, img_size, cudaMemcpyDeviceToHost);

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

	// Visualizacion
	foto.display_and_exit();

	getchar();

	return 0;

}

// Funcion que lee un archivo de tipo BMP:
__host__ void leerBMP_RGBA(const char* nombre, int* w, int* h, unsigned char **imagen)
{
	// Lectura del archivo .BMP
	FILE *archivo;

	// Abrimos el archivo en modo solo lectura binario
	if ((archivo = fopen(nombre, "rb")) == NULL)
	{
		printf("\nERROR ABRIENDO EL ARCHIVO %s...", nombre);
		// salida
		printf("\npulsa [INTRO] para finalizar");
		getchar();
		exit(1);
	}
	printf("> Archivo [%s] abierto:\n", nombre);

	// En Windows, la cabecera tiene un tamaño de 54 bytes:
	// 14 bytes (BMP header) + 40 bytes (DIB header)
	// BMP HEADER
	// Extraemos cada campo y lo almacenamos en una variable del tipo adecuado
	// posicion 0x00 -> Tipo de archivo: "BM" (leemos 2 bytes)
	unsigned char tipo[2];
	fread(tipo, 1, 2, archivo);

	// Comprobamos que es un archivo BMP
	if(tipo[0] != 'B' || tipo[1] != 'M' )
	{
		printf("\nERROR: EL ARCHIVO %s NO ES DE TIPO BMP...", nombre);
		// salida
		printf("\npulsa [INTRO] para finalizar");
		getchar();
		exit(1);
	}

	// posicion 0x02 -> Tamaño del archivo .bmp (leemos 4 bytes)
	unsigned int file_size;
	fread(&file_size, 4, 1, archivo);
	// posicion 0x06 -> Campo reservado (leemos 2 bytes)
	// posicion 0x08 -> Campo reservado (leemos 2 bytes)
	unsigned char buffer [4];
	fread(buffer, 1, 4, archivo);
	// posicion 0x0A -> Offset a los datos de imagen (leemos 4 bytes)
	unsigned int offset;
	fread(&offset, 4, 1, archivo);

	// imprimimos los datos
	printf(" \nDatos de la cabecera BMP\n");
	printf("> Tipo de archivo : %c%c\n", tipo[0], tipo[1]);
	printf("> Tamano del archivo : %u KiB\n", file_size / 1024);
	printf("> Offset de datos : %u bytes\n", offset);

	// DIB HEADER
	// Extraemos cada campo y lo almacenamos en una variable del tipo adecuado
	// posicion 0x0E -> Tamaño de la cabecera DIB (BITMAPINFOHEADER) (leemos 4 bytes)
	unsigned int header_size;
	fread(&header_size, 4, 1, archivo);

	// posicion 0x12 -> Ancho de la imagen (leemos 4 bytes)
	unsigned int ancho;
	fread(&ancho, 4, 1, archivo);

	// posicion 0x16 -> Alto de la imagen (leemos 4 bytes)
	unsigned int alto;
	fread(&alto, 4, 1, archivo);

	// posicion 0x1A -> Numero de planos de color (leemos 2 bytes)
	unsigned short int planos;
	fread(&planos, 2, 1, archivo);

	// posicion 0x1C -> Profundidad de color (leemos 2 bytes)
	unsigned short int color_depth;
	fread(&color_depth, 2, 1, archivo);

	// posicion 0x1E -> Tipo de compresion (leemos 4 bytes)
	unsigned int compresion;
	fread(&compresion, 4, 1, archivo);

	// imprimimos los datos
	printf(" \nDatos de la cabecera DIB\n");
	printf("> Tamano de la cabecera: %u bytes\n", header_size);
	printf("> Ancho de la imagen : %u pixeles\n", ancho);
	printf("> Alto de la imagen : %u pixeles\n", alto);
	printf("> Planos de color : %u\n", planos);
	printf("> Profundidad de color : %u bits/pixel\n", color_depth);
	printf("> Tipo de compresion : %s\n", (compresion == 0) ? "none" : "unknown");

	// LEEMOS LOS DATOS DEL ARCHIVO
	// Calculamos espacio para una imagen de tipo RGBA:
	size_t img_size = ancho * alto * 4;

	// Reserva para almacenar los datos del bitmap
	unsigned char *datos = (unsigned char*)malloc(img_size);;

	// Desplazzamos el puntero FILE hasta el comienzo de los datos de imagen: 0 + offset
	fseek(archivo, offset, SEEK_SET);

	// Leemos pixel a pixel, reordenamos (BGR -> RGB) e insertamos canal alfa
	unsigned int pixel_size = color_depth / 8;
	for (unsigned int i = 0; i < ancho * alto; i++)
	{
		fread(buffer, 1, pixel_size, archivo); // leemos un pixel
		datos[i * 4 + 0] = buffer[2]; // escribimos canal R
		datos[i * 4 + 1] = buffer[1]; // escribimos canal G
		datos[i * 4 + 2] = buffer[0]; // escribimos canal B
		datos[i * 4 + 3] = buffer[3]; // escribimos canal alfa (si lo hay)
	}

	// Cerramos el archivo
	fclose(archivo);

	// PARAMETROS DE SALIDA
	// Ancho de la imagen en pixeles
	*w = ancho;

	// Alto de la imagen en pixeles
	*h = alto;

	// Puntero al array de datos RGBA
	*imagen = datos;

	// Salida
	return;
}

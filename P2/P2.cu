// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#define N 8
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// declaracion
	float *hst_matriz;
	float *dev_matriz;
	float *hst1_matriz;
	float *dev1_matriz;
	// reserva en el host
	hst_matriz = (float*)malloc( N*sizeof(float) );
	hst1_matriz = (float*)malloc( N*sizeof(float) );
	// reserva en el device
	cudaMalloc( (void**)&dev_matriz, N*sizeof(float) );
	cudaMalloc( (void**)&dev1_matriz, N*sizeof(float) );
	// inicializacion de datos en el host
	srand ( (int)time(NULL) );
	for (int i=0; i<N; i++)
	{
		hst_matriz[i] = (float) rand() / RAND_MAX;
	}
	// visualizacion de datos en el host
	printf("DATOS:\n");
	for (int i=0; i<N; i++)
	{
		printf("A[%i] = %.2f\n", i, hst_matriz[i]);
	}
	// copia de datos CPU -> GPU
	cudaMemcpy(dev_matriz, hst_matriz, N*sizeof(float), cudaMemcpyHostToDevice);
	// copia de datos GPU -> GPU
	cudaMemcpy(dev1_matriz, dev_matriz, N*sizeof(float), cudaMemcpyDeviceToDevice);
	// copia de datos GPU -> CPU
	cudaMemcpy(hst1_matriz, dev1_matriz, N*sizeof(float), cudaMemcpyDeviceToHost);
	// visualizacion de datos en el host
	printf("\nDATOS:\n");
	for (int i=0; i<N; i++)
	{
		printf("A[%i] = %.2f\n", i, hst1_matriz[i]);
	}
	// salida
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}
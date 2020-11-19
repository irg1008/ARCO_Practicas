/*
 * ARQUITECTURA DE COMPUTADORES
 * 2º Grado en Ingenieria Informatica
 *
 * PRACTICA 0: "Hola Mundo"
 * >> Comprobacion de la instalacion de CUDA
 * 
 * AUTOR: APELLIDO APELLIDO Nombre
*/
///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////
// defines

///////////////////////////////////////////////////////////////////////////
// declaracion de funciones

// DEVICE: funcion llamada desde el device y ejecutada en el device

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)

// HOST: funcion llamada desde el host y ejecutada en el host
void suma(int *a, int *b, int *sumatorio);
///////////////////////////////////////////////////////////////////////////
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv) 
{
	// cuerpo del programa
	int *a;
	int *b;
	int *sumatorio;
		
	suma(a, b, sumatorio);

	// salida del programa
	time_t fecha;
	time(&fecha);
	printf("\n***************************************************\n");
	printf("Programa ejecutado el dia: %s\n", ctime(&fecha));
	printf("<pulsa INTRO para finalizar>");
	// Esto es necesario para que el IDE no cierre la consola de salida
	getchar();
	return 0;
}
 ///////////////////////////////////////////////////////////////////////////
void suma(int *a, int *b, int *sumatorio){
	int n;

	printf("Dime la longitud del array: ");
	scanf("%d", &n); 
	getchar();

	a=(int *)malloc(n*sizeof(int));
	b=(int *)malloc(n*sizeof(int));
	sumatorio=(int *)malloc(n*sizeof(int));


	for (int k=0 ; k < n; k++) 
	{	
		a[k] = k+1;
		b[k] = k+2;
	}

	for(int i = 0; i < n; i++){	
		sumatorio[i] = a[i] + b[i]; 
	}

	for (int i=0 ; i < n; i++) 
	{	
	printf("\n\nLa suma de %i y %i es: %i",a[i],b[i],sumatorio[i]);
	}
}
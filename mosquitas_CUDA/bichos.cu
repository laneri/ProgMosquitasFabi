#include<iostream>
#include <fstream>
#include<thrust/device_vector.h>
#include<thrust/sort.h>
#include<thrust/reduce.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/tuple.h>
#include<thrust/find.h>
#include <cstdlib>
#include <thrust/fill.h>

#include "bichos.h"
#include "gpu_timer.h"
#include <stdio.h>
#include <stdlib.h>


using namespace std;

int ND=400;			//número de dias 

/*--------------------  Códigos para calcular la población total de mosquitas ---------------------

Condición inicial para las mosquitas:
- estado si vive (0) o muere (1) 
- edad (dias).
- tacho en el que vive.

Archivo de salida "Poblaciones_GPU.dat" respectivamente
Para cambiar número de manzanas ir al archivo parametros.h

//*************************************************************************************************************
/*					código para GPU

Este código realiza el cálculo de las poblaciones de mosquitas directamente sobre las N manzanas durante el período de un año (400 días).

- Inicialmente considera un número fijo de mosquitas (Ninicial) distribuidas de forma random en los tachos (NUMEROTACHOS). 
- Las manzanas son independientes entre sí, es decir, un mosquito de una manzana no va a otra manzana a poner huevos.

Para que se puedan comparar resultados con el código serial, es necesario tener en cuenta las siguientes relaciones para modificar parámetros (ver parametros.h)

- (NUMERODEMAZANAS/NUMEROTACHOS)=5
- NUMEROTACHOS=Ninicial
- MAXIMONUMEROBICHOS = NUMEROMANZANAS*4000

*/
//*************************************************************************************************************

// Este es un test detallado de las funciones de la clase bichos
// Aqui esta desagregado lo que pasa con bichos en un dia
// Reproduccion, Muerte, Envejecimiento, Remocion de muertos 
/*int test1()
{
	bichos mosquitas(Ninicial);

	std::cout << "mosquitas iniciales" << std::endl;
	int dia=0;
	mosquitas.imprimir(dia);

	std::cout << "dia" << "\t" << "tovip" << "\t" << "tpupad" << std::endl;
	int tOVI=mosquitas.tiempo_entre_oviposiciones(dia);
	int TPUP=mosquitas.tiempo_pupas_adultas(dia);
	std::cout << dia << "\t" << tOVI << "\t" << TPUP << std::endl;

	std::cout << "mosquitas muertas con alguna probabilidad" << std::endl;
	mosquitas.matar(dia);
	mosquitas.imprimir(dia);

	std::cout << "mosquitas despues de reproducirse" << std::endl;
	mosquitas.reproducir(dia);
	mosquitas.imprimir(dia);

	std::cout << "mosquitas muertas por vejez" << std::endl;
	mosquitas.matar_viejos(dia);
	mosquitas.imprimir(dia);

	std::cout << "mosquitas muertas por descacharrado de los tachos" << std::endl;
	mosquitas.descacharrar_tacho(dia);
	mosquitas.imprimir(dia);

	std::cout << "mosquitas envejecidas un día" << std::endl;
	mosquitas.envejecer();
	mosquitas.imprimir(dia);

	// este paso se debe hacer despues de los nacimientos y las muertes
	std::cout << "removidas las muertas" << std::endl;
	mosquitas.recalcularN();
	mosquitas.imprimir(dia);

	//cuántos mosquitos hay por edad, por tacho y cuántos tachos hay por manzana
	std::cout << "estadisticas" << std::endl;
	mosquitas.imprimir_estadisticas();

	return 0;
};*/


// Cálculo de toda la población de mosquitas vivas para 200 manzanas
int testGPU(){

	ofstream outfile;
        outfile.open("Poblacion_total_GPU.dat");

	bichos mosquitas(Ninicial);
	
	for(int dia=0;dia<ND;dia++){
		mosquitas.avanza_dia(dia);
		int vivas=mosquitas.vivos(dia);
		if(vivas>0)
			outfile << dia << "\t" << vivas << endl;			
		else{
			outfile << "extincion de bichos"<< endl;
			exit(0);			
		}

	//Descomentar si desea conocer el detalle de la población de mosquitos en un día determinado. 
	/*if(dia==0){
	mosquitas.imprimir(dia);
    std::cout << "mosquitas despues de reproducirse" << std::endl;
	mosquitas.reproducir(dia);
	mosquitas.imprimir(dia);
	}*/
	
/*	if(dia==350){
	mosquitas.imprimir(dia);
        std::cout << "mosquitas despues de matarlas" << std::endl;
	mosquitas.matar(dia);
	mosquitas.imprimir(dia);
        std::cout << "mosquitas despues de reproducirse" << std::endl;
	mosquitas.reproducir(dia);
	mosquitas.imprimir(dia);
        std::cout << "mosquitas despues de morirse de viejas" << std::endl;	
	mosquitas.matar_viejos(dia);
	mosquitas.imprimir(dia);
	std::cout << "mosquitas despues de descacharrar tachos" << std::endl;
	mosquitas.descacharrar_tacho(dia);
	mosquitas.imprimir(dia);
        std::cout << "mosquitas despues de envejecerlas un día" << std::endl;	
	mosquitas.envejecer();
	mosquitas.imprimir(dia);
	std::cout << "mosquitas despues de remover las muertas" << std::endl;
	mosquitas.recalcularN();
	mosquitas.imprimir(dia);
	}*/


/* 	if(dia==108){
	mosquitas.imprimir(dia);
//    std::cout << "mosquitas despues de matarlas" << std::endl;
//	mosquitas.matar(dia);
//	mosquitas.imprimir(dia);
    std::cout << "mosquitas despues de reproducirse" << std::endl;
	mosquitas.reproducir(dia);
	mosquitas.imprimir(dia);
    std::cout << "mosquitas despues de morirse de viejas" << std::endl;	
	mosquitas.matar_viejos(dia);
	mosquitas.imprimir(dia);
	std::cout << "mosquitas despues de descacharrar tachos" << std::endl;
	mosquitas.descacharrar_tacho(dia);
	mosquitas.imprimir(dia);
    std::cout << "mosquitas despues de envejecerlas un día" << std::endl;	
	mosquitas.envejecer();
	mosquitas.imprimir(dia);
	std::cout << "mosquitas despues de remover las muertas" << std::endl;
	mosquitas.recalcularN();
	mosquitas.imprimir(dia);
	}*/
/*	if(dia==399){
	//mosquitas.imprimir(dia);
	cout << "\n Estadisticas" << endl;
	mosquitas.imprimir_estadisticas();}
*/
	}//cierro dias

	return 0;

   // close the opened file.
   outfile.close();
};



int main(int argc, char **argv)
{

	printf("Cálculo de la población de mosquitos para %d manzanas\n",NUMEROMANZANAS);

	//test1(argc,argv);
  	gpu_timer Reloj_GPU;
	Reloj_GPU.tic();
	testGPU();
	printf("Tiempo en GPU: %lf ms\n",Reloj_GPU.tac());

	return 0;
}



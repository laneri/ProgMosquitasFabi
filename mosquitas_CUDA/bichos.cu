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

#include "gpu_timer.h"
#include "bichos.h"

using namespace std;

// Este es un test detallado de las funciones de la clase bichos
// Aqui esta desagregado lo que pasa con bichos en un dia
// Reproduccion, Muerte, Envejecimiento, Remocion de muertos 
/*int test1(int argc, char **argv)
{
	bichos mosquitas(Ninicial);

	std::cout << "mosquitas iniciales" << std::endl;
	int dia=0;
	mosquitas.imprimir(dia);

	std::cout << "mosquitas despues de reproducirse" << std::endl;
	mosquitas.reproducir(dia);
	mosquitas.imprimir(dia);

	std::cout << "mosquitas despues de matar algunas" << std::endl;
	mosquitas.mortalidades_varias(dia);
	mosquitas.imprimir(dia);

	std::cout << "mosquitas envejecidas" << std::endl;
	mosquitas.envejecer();
	mosquitas.imprimir(dia);

	// este paso se debe hacer despues de los nacimientos y las muertes
	std::cout << "removidas las muertas" << std::endl;
	mosquitas.recalcularN();
	mosquitas.imprimir(dia);

	std::cout << "estadisticas" << std::endl;
	mosquitas.imprimir_estadisticas();

	return 0;
};
*/

// este es un test de loop de tiempo, monitoreando #vivas
int test2(int argc, char **argv){

	ofstream outfile;
        outfile.open("Poblacion_total_GPU.dat");

	bichos mosquitas(Ninicial);

	
 	for(int dia=0;dia<ND;dia++){
		mosquitas.avanza_dia(dia);
		int vivas=mosquitas.vivos();
		if(vivas>0)
			outfile << dia << "\t" << vivas << endl;
		else{
			outfile << "extincion de bichos"<< endl;
			exit(0);			
		}
	if(dia==15){
	mosquitas.imprimir(dia);
	mosquitas.imprimir_estadisticas();}

   outfile.close();

   return 0;

   // close the opened file.

};



int main(int argc, char **argv)
{
	//printf("Cálculo de la población de mosquitos para %d manzanas\n",NUMEROMANZANAS);

	//test1(argc,argv);

//  	gpu_timer Reloj_GPU;
//	Reloj_GPU.tic();
	testGPU();
//	printf("Tiempo en GPU: %lf ms\n",Reloj_GPU.tac());


	return 0;
}

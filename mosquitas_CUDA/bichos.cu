#include<iostream>
#include<thrust/device_vector.h>
#include<thrust/sort.h>
#include<thrust/reduce.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/tuple.h>
#include<thrust/find.h>
#include <cstdlib>
#include <thrust/fill.h>

#include "bichos.h"

// Este es un test detallado de las funciones de la clase bichos
// Aqui esta desagregado lo que pasa con bichos en un dia
// Reproduccion, Muerte, Envejecimiento, Remocion de muertos 
int test1(int argc, char **argv)
{
	int Ninicial=10;
	bichos mosquitas(Ninicial);

	std::cout << "mosquitas iniciales" << std::endl;
	int dia=0;
	mosquitas.imprimir(dia);

	std::cout << "mosquitas despues de reproducirse" << std::endl;
	mosquitas.tiempo_entre_oviposiciones(dia);
	mosquitas.tiempo_pupas_adultas(dia);
	mosquitas.reproducir(dia);
	mosquitas.imprimir(dia);

	std::cout << "mosquitas muertas con alguna probabilidad" << std::endl;
	mosquitas.tiempo_pupas_adultas(dia);
	mosquitas.matar(dia);
	mosquitas.imprimir(dia);

	std::cout << "mosquitas muertas por vejez" << std::endl;
	mosquitas.matar_viejos(dia);
	mosquitas.imprimir(dia);


	std::cout << "mosquitas envejecidas un día" << std::endl;
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


// este es un test de loop de tiempo, monitoreando #vivas
int test2(int argc, char **argv){
	int Ninicial=10;
	bichos mosquitas(Ninicial);

	int Ndias=400;
	for(int dia=0;dia<Ndias;dia++){
		mosquitas.avanza_dia(dia);
		int vivas=mosquitas.vivos();
		if(vivas>0)
			std::cout << vivas << std::endl;
		else{
			std::cout << "extincion de bichos"<< std::endl;
			exit(0);			
		}
	}
	return 0;
};



int main(int argc, char **argv)
{
	test1(argc,argv);
	//test2(argc,argv);
	return 0;
}


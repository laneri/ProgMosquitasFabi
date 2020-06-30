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

using namespace std;

//---------------------  Código para calcular la población de mosquitos en N manzanas  -----------------------

//************************************************************************************************************
//                                                   GPU
//*************************************************************************************************************

// Este es un test detallado de las funciones de la clase bichos
// Aqui esta desagregado lo que pasa con bichos en un dia
// Reproduccion, Muerte, Envejecimiento, Remocion de muertos 
int test1(int argc, char **argv)
{
	int Ninicial=5;
	bichos mosquitas(Ninicial);

	std::cout << "mosquitas iniciales" << std::endl;
	int dia=0;
	mosquitas.imprimir(dia);

	std::cout << "dia" << "\t" << "tovip" << "\t" << "tpupad" << std::endl;
	int tOVI=mosquitas.tiempo_entre_oviposiciones(dia);
	int TPUP=mosquitas.tiempo_pupas_adultas(dia);
	std::cout << dia << "\t" << tOVI << "\t" << TPUP << std::endl;

	std::cout << "mosquitas despues de reproducirse" << std::endl;
	mosquitas.reproducir(dia);
	mosquitas.imprimir(dia);

	std::cout << "mosquitas muertas con alguna probabilidad" << std::endl;
	mosquitas.matar(dia);
	mosquitas.imprimir(dia);

	std::cout << "mosquitas muertas por vejez" << std::endl;
	mosquitas.matar_viejos(dia);
	mosquitas.imprimir(dia);

	std::cout << "mosquitas muertas por descacharrado" << std::endl;
	mosquitas.descacharrar_tacho(dia);
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

	ofstream outfile;
        outfile.open("Suma_poblaciones_GPU.dat");

	int Ndias=400;
	//int Manzana[NUMEROMANZANAS][Ndias];

	//for(int i=0;i<NUMEROMANZANAS;i++){for(int j=0;j<Ndias;j++){Manzana[i][j]=0;}}

	//int k=0;
	//while(k < NUMEROMANZANAS){
	int Ninicial=10;					//nro. inicial de mosquitos 
	bichos mosquitas(Ninicial);	

		for(int dia=0;dia<Ndias;dia++){
			mosquitas.avanza_dia(dia);
			int vivas=mosquitas.vivos();
			if(vivas>0)
			outfile << dia << "\t" << vivas << endl;			
			else{
			outfile << "extincion de bichos"<< endl;
			exit(0);}
		//Manzana[k][dia]=vivas;
		}//cierro dias
	//k++;
	//}

	return 0;

   // close the opened file.
   outfile.close();
};

int main(int argc, char **argv)
{
//	test1(argc,argv);
//	printf("Cálculo de la población de mosquitos en %d manzanas\n",NUMEROTACHOS);	
  	gpu_timer Reloj_GPU;
	Reloj_GPU.tic();
	test2(argc,argv);
	printf("Tiempo en GPU: %lf ms\n",Reloj_GPU.tac());

	return 0;
}

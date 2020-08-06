#include <cstdlib>
#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ran2.h"
#include "parametros.h"
#include "main.cu"
#include "cpu_timer.h"

using namespace std;

struct evolucion_temporal
{
  const unsigned int ND;

  evolucion_temporal(unsigned int _ND) : ND(_ND) {}

  evolucion_temporal operator()(int ji,int dia,tachj T, mosquitos M,int indice,long *seed,int descach, int *poblacion){

	for(dia = 1; dia <= ND; dia++){

	int tovip=dia_entre_oviposiciones(dia);
	int tpupad=dia_pupas_adultas(dia);						
	conteo_huevosxT(dia,indice,tpupad,M,T);
 	mortalidades(indice,M,tpupad,seed);

		for(int i =1;i <= indice;i++){
		if (T.tach[M.tacho[i]] < sat){ 
			if(M.VoM[i] == 1 && M.DdV[i] > tpupad){
				  if(M.DdV[i]%tovip == 0){
 				  int iovip=ran2(seed)*4+7; 
   					for(int ik=1;ik <= iovip;ik++){ 
 					indice=indice+1; 
 					M.VoM[indice]=1;
 					M.DdV[indice]=1;   
 					M.tacho[indice]=M.tacho[i]; 
	         			M.DdM[indice]=ran2(seed)*3+28; 
					int j=M.tacho[indice];
 					T.tach[j]=T.tach[j]+1;
   					}
  				   }    
			}  
   		} 
	}


	muerte_x_vejez(indice,M);
	descacharrado(dia,seed,descach,M,indice,tpupad);
	poblacion_total(dia,indice,tpupad,M,poblacion);
	envejezco_poblacion(indice,M);
	}//cierro loop para dias
return 0;
	}
};


int main(int argc, char **argv){

 ofstream outfile,outfile1;
 outfile.open("poblaciones_t.dat");
 outfile1.open("suma_poblaciones.dat");


 int dia=0,indice=0;
 double producto=round(ntachito*prop);
 int  descach=producto;						//cantidad de tachos que vacío con la propaganda
 int ND=400;							//número de dias	

 long semilla = -975;  						//semilla del generador de numeros aleatorios


//estructuras para cada mosquita
 mosquitos M;							
//estructura para tacho donde pone huevos la mosquita
 tachj T;


 int *poblacion;
 poblacion= (int *)malloc((ttotal+1)*sizeof(int));

 for(int i=1;i<=ttotal;i++){poblacion[i]=0;}

 int *poblacion_suma;
 poblacion_suma= (int *)malloc((ttotal+1)*sizeof(int));
	
 for(int i=1;i<=ttotal;i++){poblacion_suma[i]=0;}


cpu_timer Reloj_CPU;
Reloj_CPU.tic();

//for(int ji=1;ji<=nmanzanas;ji++){
 int ji=1;
// long semilla = -975 + ji;
 inicializoMosquitos(M);
 indice=0;						//contador que debo inicializar en cero en cada manzana
 printf("Manzana:%d\n",ji);

	/*condiciones iniciales*/
 	for(int i=1;i<=inhem;i++){
	M.VoM[i] = 1; 
   	M.tacho[i] = i;
   	M.DdV[i] =ran2(&semilla)*5+12;
   	M.DdM[i] =ran2(&semilla)*3+28;
	indice=indice+1;
	printf("\t\t%d\t %d\t %d\t %d\t %d\n",i,M.VoM[i],M.tacho[i],M.DdV[i],M.DdM[i]);
	}


	evolucion_temporal operacion(ND);
	operacion(ji,dia,T,M,indice,&semilla,descach,poblacion);

	//guardo datos de la población de cada manzana en un archivo
	for(dia=1; dia<=ttotal ; dia++){    	
	outfile << ji << "\t" << dia << "\t" << poblacion[dia] << "\n"; 
	poblacion_suma[dia] = poblacion_suma[dia]+ poblacion[dia];
	}

//  }//cierro bucle para las manzanas
printf("\n");
	//milisegundos
	printf("Calcula poblaciones de mosquitas para un nro de manzanas %d en %lf ms en CPU\n",nmanzanas,Reloj_CPU.tac());

	for(int dia=1;dia<=ttotal;dia++){
	outfile1 << dia << "\t" << poblacion_suma[dia] << "\n";}
	outfile1 << endl;

//cierro archivos
outfile.close();
outfile1.close();

//libero memoria
free(poblacion); 
free(poblacion_suma); 
							
}// end for main


#include<stdlib.h>
#include<stdio.h>
#include "ran2.h"
#include "math.h"
#include "funciones.h"

int main()
{ 
	FILE *out;
	out=fopen("acuaticos_vivos_total_C.dat","w");
 
  int N=5; 									//dimensión == nro de hembras en una manzana (== inhem)
  int tiempo,tovip,indice,descach,ntach,mor,idia,ad,ac;
  int *tach;   									//vector para los tachos de nmax elementos 
  double producto;

//  int semilla = 20;
  long semilla; 
//  long* idum;
    semilla= -975;
//    idum=&semilla;
/*
mosquitas es un puntero a una estructura 'manzana' con campos {VoM[nmax], tacho[nmax], DdV[nmax], DdM[nmax]}
*/

//  manzana *mosquitas;								
//  mosquitas = (manzana *) malloc(sizeof(manzana)); 				//reservo memoria

  producto=round(ntachito*prop);
  descach=producto;								//cantidad de tachos que vacío con la propaganda
  indice=0;

/*
Reservo memoria ya que en la evolución temporal la población de mosquitas crece y si no lo hago me sale segmentation fault.
Cada uno de los 4 arrays son de dimensión nmax > N
*/
//	aloco_memoria(tach,mosquitas); 
//        inicializar(mosquitas); //inicializo los arrays en cero

struct mosquitos M();
/*
Para poder comparar con los datos extraídos del programa de Fabiana, este programa calcula la población de mosquitas correspondiente a una manzana con número de hembras N = 5 y condiciones iniciales dadas por 
*/
//	printf("nro de hembras\t VoM\t tacho\t DdV\t DdM\n");
/*for(int j=1;j<=nmanzanas;j++){
	for(int i=1;i<=N;i++){
        M.Numanzana[j]=j;    
		M.VoM[i] = 1;
		M.tacho[i] = i;
		M.DdV[i] = ran2(&semilla)*5+12;
		M.DdM[i] = ran2(&semilla)*6+27;
        indice=indice+1;
		printf("\t%d\t %d\t %d\t %d\t %d\n",i,M.VoM[i],M.tacho[i],M.DdV[i],M.DdM[i]);
        }
            
 }*/
	


	

}//cierro main




























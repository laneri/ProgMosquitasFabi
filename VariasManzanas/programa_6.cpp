#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ran2.h"
#include "funciones.h"


int main(){

/* guardo datos en los archivos */
	FILE *out1;
	out1=fopen("acuaticos_vivos_total_C.dat","w"); 


 int tiempo,tovip,descach,indice,ntach;
 int ad,ac;
 int *tach;   //vector para los tachos de nmax elementos 
 double producto;

 long semilla = -975;  		//semilla del generador de numeros aleatorios
 producto=round(ntachito*prop);
 descach=producto;		//cantidad de tachos que vacío con la propaganda

 indice=0;
 mosquitos M;

 tach = (int *)malloc(nmanzanas*nmax*sizeof(int)); //Aloco memoria para vector tach[nmax]

 printf("\tnro de hembras\t VoM\t tacho\t DdV\t DdM\n");
/* Bucle para las manzanas */

 for(int j=1;j<=nmanzanas;j++){

 printf("Manzana:%d\n",j);

/*condiciones iniciales*/
 	for(int i=1;i<=inhem;i++){
	M.VoM[i] = 1; 
   	M.tacho[i] = i;
   	M.DdV[i] =ran2(&semilla)*5+12;
   	M.DdM[i] =ran2(&semilla)*6+27;
	indice=indice+1;
	printf("\t\t%d\t %d\t %d\t %d\t %d\n",i,M.VoM[i],M.tacho[i],M.DdV[i],M.DdM[i]);
	}

/*Evolución temporal para mi sistema*/
 
//-------------------------------------------------------------------	
	for(tiempo=1; tiempo<=ttotal ;tiempo++){ 
//-------------------------------------------------------------------------
// defino la temperatura segun la estacion y el tiempo entre oviposiciones
//-------------------------------------------------------------------------

	tovip=tiempo_en_dias(tiempo,tovip,&semilla);

//---------------------------------------------------------------------
// cuento cuantos huevos hay en cada tacho
//---------------------------------------------------------------------- 

	incializar_vec(tach);
	huevos_x_tacho(indice,M,tach);

//--------------------------------------------------------------------------
// MORTALIDADADES VARIAS (morirse antes de ser vieja): mato acuaticos con prob moracu,
// a las que estan por volverse adultas picadoras con prob morpupad
// y a las adultas con prob morad (no es la muerte por vejez, ojo)
//--------------------------------------------------------------------------

	mortalidades(indice, M,&semilla);

// -------------------------------------------------------------------	    
// NACIMIENTOS CON SATURACION
//-------------------------------------------------------------------------- 

	for(int i =1;i <= indice;i++){

		if (tach[M.tacho[i]] < sat){ 

			if(M.VoM[i] == 1 && M.DdV[i] > tpupad){

				  if(M.DdV[i]%tovip == 0){

   					for(int ik=1;ik <= iovip;ik++){ 

 					indice=indice+1; 
 					M.VoM[indice]=1;
 					M.DdV[indice]=1;   
 					M.tacho[indice]=M.tacho[i]; 
	         			M.DdM[indice]=ran2(&semilla)*6+27; 
 					tach[M.tacho[indice]]=tach[M.tacho[indice]]+1;
   					}

  				   }    //del periodo entre oviposiciones y saturacion

			}      //de vivas y maduras

   		}   //de la saturac

	//printf("%d\n",tach[M.tacho[i]]);
	}  // de poblacion total (indice)

//--------------------------------------------------------------------------
//MUERTE POR VEJEZ: la mato cuando su edad alcanza el tiempo de vida asignado
//--------------------------------------------------------------------------

	muerte_x_vejez(indice,M);

//--------------------------------------------------------------------			  		 
//DESCACHARRADO: elimino los acuaticos de algunos tachos, 
//una vez por semana a partir del día 20 (esto se puede cambiar, obvio)
//-------------------------------------------------------------------- 

	descacharro(tiempo,descach,&semilla,indice,M,tach);

//--------------------------------------------------------------------			  		 
// cuento poblacion adulta y acuatica
//--------------------------------------------------------------------			  
  ad=0;  //adultas total
  ac=0;  //acuaticas vivas total

	ad=poblacion_adultos(ad,indice,M);
	ac=poblacion_acuaticos(ac,indice,M);

//--------------------------------------------------------------------	   
//envejezco a toda la población un día (aumento en 1)
//--------------------------------------------------------------------

	envejecer_poblacion(indice,M);

//--------------------------------------------------------------------
//guardo los datos 
//--------------------------------------------------------------------
 	fprintf(out1,"\t %d\t %d\t %d\t %d\n",tiempo,ad,ac,ad+ac);		

	}// cierro bucle para el tiempo
	fprintf(out1,"\n");//inserta una linea en blanco en el archivo.dat

  }//cierro bucle para las manzanas

	fclose(out1);

free(tach);
}// end for main

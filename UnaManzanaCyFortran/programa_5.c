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

  long semilla = -975; 

/*
mosquitas es un puntero a una estructura 'manzana' con campos {VoM[nmax], tacho[nmax], DdV[nmax], DdM[nmax]}
*/

  manzana *mosquitas;								
  mosquitas = (manzana *) malloc(sizeof(manzana)); 				//reservo memoria

  producto=round(ntachito*prop);
  descach=producto;								//cantidad de tachos que vacío con la propaganda
  indice=0;

/*
Reservo memoria ya que en la evolución temporal la población de mosquitas crece y si no lo hago me sale segmentation fault.
Cada uno de los 4 arrays son de dimensión nmax > N
*/
	aloco_memoria(tach,mosquitas); 
        inicializar(mosquitas); //inicializo los arrays en cero

/*
Para poder comparar con los datos extraídos del programa de Fabiana, este programa calcula la población de mosquitas correspondiente a una manzana con número de hembras N = 5 y condiciones iniciales dadas por 
*/
	printf("nro de hembras\t VoM\t tacho\t DdV\t DdM\n");

	for(int i=1;i<=N;i++){
		mosquitas -> VoM[i] = 1;
		mosquitas -> tacho[i] = i;
		mosquitas -> DdV[i] = ran2(&semilla)*5+12;
		mosquitas -> DdM[i] = ran2(&semilla)*6+27;
		indice=indice+1;
		printf("\t%d\t %d\t %d\t %d\t %d\n",i,mosquitas->VoM[i],mosquitas->tacho[i],mosquitas->DdV[i],mosquitas->DdM[i]);
	}

//---------------------------------------------------------------------------------------------------------------------------------------- Comienzo la evolución temporal para mi sistema. 
//-----------------------------------------------------------------------------------------------------------------------------------------	

	for(tiempo=1; tiempo<=ttotal ;tiempo++){ 
//-------------------------------------------------------------------------
// defino la temperatura según la estación y el tiempo entre oviposiciones
//-------------------------------------------------------------------------
//	tovip=tiempo_en_dias(tiempo,tovip,semilla); esto no es correcto porque lo paso por valor y luego toma la direccion de memoria de la copia en funciones.h
    tovip=tiempo_en_dias(tiempo,tovip,&semilla);
    /*		if(tiempo < 80 || tiempo > 320) tovip=tovip1;
  	
		if(tiempo > 160 && tiempo < 240) tovip=tovip3;

  		if(tiempo >= 80 && tiempo <= 160){
    			if(ran2(&semilla) < 0.5){
    	 		tovip=tovip2a;}
    			else{
   	 		tovip=tovip2b;
   		 	}
	  	}

	  	if(tiempo >= 240 && tiempo <= 320){
	    		if(ran2(&semilla) < 0.5){
		 	tovip=tovip2a;}
	    		else{
		 	tovip=tovip2b;
	    		}
	  	}
*/
//---------------------------------------------------------------------
// cuento cuantos huevos hay en cada tacho
//---------------------------------------------------------------------- 
	incializar_vec(N,tach);
	huevos_x_tacho(indice,tach,mosquitas);

//--------------------------------------------------------------------------
// MORTALIDADADES VARIAS (morirse antes de ser vieja)
//--------------------------------------------------------------------------
	for(int i =1;i <= indice;i++){
			if (mosquitas -> VoM[i] == 1 && mosquitas -> DdV[i] < tpupad){ 
			  	 if (ran2(&semilla) < moracu)mosquitas -> VoM[i]=0;  
			}

			if (mosquitas -> VoM[i] == 1 && mosquitas -> DdV[i] == tpupad){ 
		  	 	 if (ran2(&semilla) < morpupad)mosquitas -> VoM[i]=0;  
		  	}

			if (mosquitas -> VoM[i] == 1 && mosquitas -> DdV[i] > tpupad){ 
		  	 	 if(ran2(&semilla) < morad)mosquitas -> VoM[i]=0;  
			}

		//printf("%d\n",mosquitas -> DdV[i]);
		}  //del indice 
	
	

// -------------------------------------------------------------------	    
// NACIMIENTOS CON SATURACION manda fruta
//-------------------------------------------------------------------------- 
	for(int i =1;i <= indice; i++){

		if (tach[mosquitas -> tacho[i]] < sat){ 

			if(mosquitas -> VoM[i] == 1 && mosquitas -> DdV[i] > tpupad){

				  if(mosquitas -> DdV[i]%tovip == 0){

   					for(int ik=1;ik <= iovip;ik++){ 

 					indice=indice+1; 
 					mosquitas -> VoM[indice]=1;
 					mosquitas -> DdV[indice]=1;   
 					mosquitas -> tacho[indice]=mosquitas -> tacho[i];  
	         			mosquitas -> DdM[indice]=ran2(&semilla)*6+27;
 					tach[mosquitas->tacho[indice]]=tach[mosquitas -> tacho[indice]]+1;
   					}

  				   }    //del periodo entre oviposiciones y saturacion

			}      //de vivas y maduras

   		}   //de la saturac

	//printf("%d\n",tach[mosquitas -> tacho[i]]);
	}  // de poblacion total (indice)

//--------------------------------------------------------------------------
//MUERTE POR VEJEZ: la mato cuando su edad alcanza el tiempo de vida asignado
//--------------------------------------------------------------------------

	muerte_x_vejez(indice,mosquitas);

//--------------------------------------------------------------------			  		 
//DESCACHARRADO: elimino los acuaticos de algunos tachos, 
//una vez por semana a partir del día 20 (esto se puede cambiar, obvio)
//-------------------------------------------------------------------- 
   	if(tiempo%7 == 0 && tiempo > 20){
  		for(int itach=1;itach <= descach;itach++){
    		ntach=ran2(&semilla)*ntachito;

			for(int i =1;i <= indice;i++){

  				if (mosquitas -> VoM[i] == 1 && mosquitas -> DdV[i] < tpupad && mosquitas -> tacho[i] == ntach)mosquitas -> VoM[i]=0;
				//printf("%d\n",mosquitas -> VoM[i]);
			}

		}    
	
   	}
//--------------------------------------------------------------------			  		 
// cuento poblacion adulta y acuatica
//--------------------------------------------------------------------	
 
  	ad=0;  //adultas total
  	ac=0;  //acuaticas vivas total
 
	ad=poblacion_adulta(ad,indice,mosquitas);	
	ac=poblacion_acuatica(ac,indice,mosquitas);	 

//--------------------------------------------------------------------	   
//envejezco a toda la población un día (aumento en 1)
//--------------------------------------------------------------------

	envejecer_poblacion(indice,mosquitas);

//--------------------------------------------------------------------
//guardo los datos 
//--------------------------------------------------------------------
 	fprintf(out,"\t %d\t %d\t %d\t %d\n",tiempo,ad,ac,ad+ac);		

	}//cierro loop temporal 

}//cierro main




























#include <stdlib.h>
#include<stdio.h>
#include<math.h>
#include "ran2.h"

#define nmax 800000


int main(){   
	FILE *out1,*out2,*out3;
	out1=fopen("adultos_vivos_por_tacho_C.dat","w"); 
	out2=fopen("adultos_nuevo_por_tacho_C.dat","w"); 
	out3=fopen("acuaticos_vivos_total_C.dat","w"); 

 int inhem,ntachito,iovip,indice,idia,mor,ntach;
 int ad,ac,ad1,ad2,ad3,ad4,ad5,adn1,adn2,adn3,adn4,adn5;
 int **mosquita,*tach;   //mosquitas[nmax][4] y tach[nmax]
 int tiempo,ttotal,tpupad,tovip,tovip1,tovip2b,tovip2a,tovip3,sat,descach;  
 double morhue,morlar,morpup,moracu,morad,morpupad,prop,producto;

// Parámetros

 inhem=5;     
 ntachito=5;   	  
 iovip=32;    
 ttotal=400;  

 morhue=0.01;   
 morlar=0.01;   
 morpup=0.01;   

 moracu=morhue+morlar+morpup;   


 morad= 0.01; 
 morpupad=0.17;

 tpupad=17; 
 
 tovip1=2;
 tovip2a=3;  
 tovip2b=4;   
 tovip3=10;

 sat=800;
 long semilla = -975;  //semilla del generador de numeros aleatorios

 prop=0.6;    
 producto=round(ntachito*prop);
 descach=producto;
 //printf("%d\n",descach);
 
//-------------------------------------------- Aloco memoria para tach[nmax] y mosquita[nmax][4] -----------------------
 tach = (int *)malloc(nmax*sizeof(int));

 mosquita = (int **)malloc(nmax*sizeof(int *));
	 for (int i=1;i <= nmax;i++){
		mosquita[i] = (int *)malloc(4*sizeof(int));
		}

//-------------------------------------------
// inicializo la poblacion mosquítica
//-------------------------------------------
//--------------------------------------------- Inicializo mosquita[nmax][4]=0 ------------------------------------------
	 for(int i=1;i <= nmax;i++){
		for(int j=1;j <=4 ;j++){
		mosquita[i][j]=0;
		}
	}
 


   indice=0;      

        for(int kk=1;kk <= inhem;kk++){   
   		mosquita[kk][1]=1;
   		idia=ran2(&semilla)*5+12;
   		mosquita[kk][2]=kk;
   		mosquita[kk][3]=idia;
   		mor=ran2(&semilla)*6+27; 
   		mosquita[kk][4]=mor;
   		indice=indice+1;
   		printf("%d\t %d\t %d\t %d\t %d\n",kk,mosquita[kk][1],mosquita[kk][2],mosquita[kk][3],mosquita[kk][4]);
    	}


//-------------------------------------------------------------------	
	for(tiempo=1; tiempo<=ttotal ;tiempo++){ 
//-------------------------------------------------------------------------
// defino la temperatura segun la estacion y el tiempo entre oviposiciones
//-------------------------------------------------------------------------
		if(tiempo < 80 || tiempo > 320) tovip=tovip1;
  	
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
	//printf("%d\t %d\n",tiempo,tovip); 

//---------------------------------------------------------------------
// cuento cuantos huevos hay en cada tacho
//---------------------------------------------------------------------- 
	for(int i=1;i <= nmax;i++){   
	tach[i]=0;
	}
  
	int j=0;
		for(int i=1;i <= indice; i++){
   			if(mosquita[i][3] < tpupad && mosquita[i][1] == 1){
    			j=mosquita[i][2]; 
		    	tach[j]=tach[j]+1;
			}
		} 

	//printf("%d\t %d\n",tiempo,tach[j]);
//--------------------------------------------------------------------------
// MORTALIDADADES VARIAS (morirse antes de ser vieja): mato acuaticos con prob moracu,
// a las que estan por volverse adultas picadoras con prob morpupad
// y a las adultas con prob morad (no es la muerte por vejez, ojo)
//--------------------------------------------------------------------------
	for(int i =1;i <= indice;i++){
		if (mosquita[i][1] == 1 && mosquita[i][3] < tpupad){ 
		  	 if (ran2(&semilla) < moracu)mosquita[i][1]=0;  
		}

		if (mosquita[i][1] == 1 && mosquita[i][3] == tpupad){ 
	  	 	 if (ran2(&semilla) < morpupad)mosquita[i][1]=0;  
	  	}

		if (mosquita[i][1] == 1 && mosquita[i][3] > tpupad){ 
	  	 	 if(ran2(&semilla) < morad)mosquita[i][1]=0;  
		}

	//printf("%d\n",mosquita[i][1]);
	}  //del indice 

// -------------------------------------------------------------------	    
// NACIMIENTOS CON SATURACION
//-------------------------------------------------------------------------- 
	for(int i =1;i <= indice;i++){

		if (tach[mosquita[i][2]] < sat){ 

			if(mosquita[i][1] == 1 && mosquita[i][3] > tpupad){

				  if(mosquita[i][3]%tovip == 0){

   					for(int ik=1;ik <= iovip;ik++){ 

 					indice=indice+1; 
 					mosquita[indice][1]=1;
 					mosquita[indice][3]=1;   
 					mosquita[indice][2]=mosquita[i][2]; 
 					mor=ran2(&semilla)*6+27; 
	         			mosquita[indice][4]=mor;
 					tach[mosquita[indice][2]]=tach[mosquita[indice][2]]+1;
   					}

  				   }    //del periodo entre oviposiciones y saturacion

			}      //de vivas y maduras

   		}   //de la saturac

//	printf("%d\n",tach[mosquita[i][2]]);
	}  // de poblacion total (indice)
 
//--------------------------------------------------------------------------
//MUERTE POR VEJEZ: la mato cuando su edad alcanza el tiempo de vida asignado
//--------------------------------------------------------------------------
	for(int i =1;i <= indice;i++){

		if (mosquita[i][1] == 1 && mosquita[i][3] >= mosquita[i][4])mosquita[i][1]=0;
	//printf("%d\n",mosquita[i][1]);

	} 

//--------------------------------------------------------------------			  		 
//DESCACHARRADO: elimino los acuaticos de algunos tachos, 
//una vez por semana a partir del día 20 (esto se puede cambiar, obvio)
//-------------------------------------------------------------------- 
   	if(tiempo%7 == 0 && tiempo > 20){

  		for(int itach=1;itach <= descach;itach++){
    		ntach=ran2(&semilla)*ntachito;

			for(int i =1;i <= indice;i++){

  				if (mosquita[i][1] == 1 && mosquita[i][3] < tpupad && mosquita[i][2] == ntach)mosquita[i][1]=0;
				//printf("%d\n",mosquita[i][1]);
			}

		}    
	
   	}
//--------------------------------------------------------------------			  		 
// cuento poblacion adulta y acuatica
//--------------------------------------------------------------------			  
  ad=0;  
  ac=0;  
  ad1=0;
  ad2=0;
  ad3=0;
  ad4=0;
  ad5=0;
  adn1=0; 
  adn2=0;
  adn3=0;
  adn4=0;
  adn5=0;
  
  	for(int i =1;i <= indice; i++){
		if (mosquita[i][1] == 1 && mosquita[i][3] >= tpupad)ad=ad+1; //adultos tot
		if (mosquita[i][1] == 1 && mosquita[i][3] < tpupad)ac=ac+1; //acuaticos tot

		if (mosquita[i][1] == 1 && mosquita[i][3] >= tpupad && mosquita[i][2] == 1)ad1=ad1+1;
		if (mosquita[i][1] == 1 && mosquita[i][3] >= tpupad && mosquita[i][2] == 2)ad2=ad2+1;
		if (mosquita[i][1] == 1 && mosquita[i][3] >= tpupad && mosquita[i][2] == 3)ad3=ad3+1;
		if (mosquita[i][1] == 1 && mosquita[i][3] >= tpupad && mosquita[i][2] == 4)ad4=ad4+1;
		if (mosquita[i][1] == 1 && mosquita[i][3] >= tpupad && mosquita[i][2] == 5)ad5=ad5+1;

		if (mosquita[i][1] == 1 && mosquita[i][3] == tpupad && mosquita[i][2] == 1)adn1=adn1+1;
		if (mosquita[i][1] == 1 && mosquita[i][3] == tpupad && mosquita[i][2] == 2)adn2=adn2+1;
		if (mosquita[i][1] == 1 && mosquita[i][3] == tpupad && mosquita[i][2] == 3)adn3=adn3+1;
		if (mosquita[i][1] == 1 && mosquita[i][3] == tpupad && mosquita[i][2] == 4)adn4=adn4+1;
		if (mosquita[i][1] == 1 && mosquita[i][3] == tpupad && mosquita[i][2] == 5)adn5=adn5+1;
	}


//--------------------------------------------------------------------	   
//envejezco a toda la población un día (aumento en 1)
//--------------------------------------------------------------------
   	for(int i =1;i <= indice;i++){

	 	 if (mosquita[i][1] == 1)mosquita[i][3]=mosquita[i][3]+1;
	//printf("%d\n",mosquita[i][3]);
	} 
	
//--------------------------------------------------------------------
//guardo los datos 
//--------------------------------------------------------------------
 	fprintf(out3,"\t %d\t %d\t %d\t %d\n",tiempo,ad,ac,ad+ac);		
	fprintf(out1,"\t %d\t %d\t %d\t %d\t %d \t %d\n",tiempo,ad1,ad2,ad3,ad4,ad5);	
	fprintf(out2,"\t %d\t %d\t %d\t %d\t %d\t %d\n",tiempo,adn1,adn2,adn3,adn4,adn5);

	}// end for time



	fclose(out1);
	fclose(out2);
	fclose(out3);

}//cierro main

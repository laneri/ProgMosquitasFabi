#include <stdio.h>
#include <stdlib.h>
#include "ran2.h"
#include <math.h>

#define nmax 8000000	
#define inhem 5		//nro de hembras
#define ntachito 5   	//nro de tachos
#define iovip 32        //nro de oviposición
#define ttotal 200	//tiempo
#define morhue 0.01   	//mortalidad de huevos
#define morlar 0.01     //mortalidad de larvas 
#define morpup 0.01   	//mortalidad de pupas
#define morad 0.01 	//mortalidad diaria adultas
#define morpupad 0.17	//pupas que no se vuelven adultas
#define tpupad 17 	//pupas se vuelven adultas que pican a los 17 dias
#define tovip1 2	//tiempo entre dos oviposiciones (T=30)
#define tovip2a 3  	//tiempo entre dos oviposiciones (T=25)
#define tovip2b 4   	//tiempo entre dos oviposiciones (T=25)
#define tovip3 10	//tiempo ente dos oviposiciones (T=18)
#define sat 800		//numero maximo de huevos en cada tacho
#define prop 0.6    	//efectividad de la propaganda



/*creo una estructura que se llama elementos, donde guardo:

1) si vive o muere (1 o 0) 		--> VoM
2) tacho en el que se encuentra 	--> tacho
3) días de vida (entre 12 y 17 días) 	--> DdV
4) día que muere (entre 27 y 32) 	--> DdM
que son los valores representativos para el vector mosquita que defino luego
*/

typedef struct elementos_mosquita{
  int VoM; 
  int tacho; 
  int DdV; 
  int DdM;  
}elementos;



int main(){

/* guardo datos en los archivos */
	FILE *out1,*out2,*out3;
	out1=fopen("adultos_vivos_por_tacho_C.dat","w"); 
	out2=fopen("adultos_nuevo_por_tacho_C.dat","w"); 
	out3=fopen("acuaticos_vivos_total_C.dat","w"); 


 int tiempo,tovip,descach,indice,ntach,mor,idia;
 int ad,ac,ad1,ad2,ad3,ad4,ad5,adn1,adn2,adn3,adn4,adn5;
 int *tach;   //vector para los tachos de nmax elementos 
 double moracu,producto;

 moracu=morhue+morlar+morpup; 	//mortalidad total de las acuaticas  
 long semilla = -975;  		//semilla del generador de numeros aleatorios
 producto=round(ntachito*prop);
 descach=producto;		//cantidad de tachos que vacío con la propaganda


/*condiciones iniciales*/
	indice=0;
	elementos *mosquita=NULL; //variable "mosquita" donde voy a guardar {VoM,tacho,DdV,DdM} de la estructura elementos

	mosquita = (elementos*)malloc(nmax*sizeof(elementos)); //reservo memoria dinámica para esa variable por si el número de mosquitas tiene que ser muy grande en un futuro

	/*ahora quiero que el vector mosquita[nmax] donde nmax es la dimensión y representa el número de mosquitas adultas que puedo querer en mi programa, que en este caso son 5, tenga por elementos a {VoM,tacho,DdV,DdM}, entonces */

 	for(int i=1;i<=inhem;i++){
	mosquita[i].VoM = 1; //buscame en elementos (linea 35-40) a VoM, asignale el valor 1 y guardalo en mosquita[i]  
   	idia=ran2(&semilla)*5+12;
   	mosquita[i].tacho = i;//buscame en elementos (linea 35-40) a tacho, asignale el valor i y guardalo en mosquita[i] 
   	mosquita[i].DdV =idia;//buscame en elementos (linea 35-40) a DdV, asignale el valor idia y guardalo en mosquita[i] 
   	mor=ran2(&semilla)*6+27; 
   	mosquita[i].DdM =mor;//buscame en elementos (linea 35-40) a DdM, asignale el valor mor y guardalo en mosquita[i]
	indice=indice+1;
	}
/*cuando sale del bucle, para un i determinado, por ejemplo,i=1 tengo armado mosquita[1]={1,1,idia,mor}, y asi para cada i=1,...,5*/

/* Imprimo en la pantalla los vectores mosquitas para chequear que coincidan los valores con los valores que obtiene Fabiana en su programa*/
	for(int i=1;i<=inhem;i++){
	printf("\t%d\t %d\t %d\t %d\t %d\n",i,mosquita[i].VoM,mosquita[i].tacho,mosquita[i].DdV,mosquita[i].DdM);
	}

// Comienzo la evolución temporal para mi sistema. En cada una de las etapas que voy a calcular para la hembra adulta voy a tener que acceder a la variable mosquita[i] donde tengo almacenados los valores para {VoM,tacho,DdV,DdM} para cada i

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

	tach = (int *)malloc(nmax*sizeof(int)); //Aloco memoria para vector tach[nmax] por si nmax llega a ser muy grande en el futuro, sino salta un segmentation fault en el output

	for(int i=1;i <= nmax;i++){     //para ello inicializo el vector tacho[nmax] en 0
	tach[i]=0; 			
	}
  
	int j=0;
		for(int i=1;i <= indice; i++){ //ciclo que va desde i=1,...,5
   			if(mosquita[i].DdV < tpupad && mosquita[i].VoM == 1){ 
    			j=mosquita[i].tacho; 
		    	tach[j]=tach[j]+1;
			}
		} 

	//printf("%d\n",tach[j]);

//--------------------------------------------------------------------------
// MORTALIDADADES VARIAS (morirse antes de ser vieja): mato acuaticos con prob moracu,
// a las que estan por volverse adultas picadoras con prob morpupad
// y a las adultas con prob morad (no es la muerte por vejez, ojo)
//--------------------------------------------------------------------------

	for(int i =1;i <= indice;i++){
		if (mosquita[i].VoM == 1 && mosquita[i].DdV < tpupad){ 
		  	 if (ran2(&semilla) < moracu)mosquita[i].VoM=0;  
		}

		if (mosquita[i].VoM == 1 && mosquita[i].DdV == tpupad){ 
	  	 	 if (ran2(&semilla) < morpupad)mosquita[i].VoM=0;  
	  	}

		if (mosquita[i].VoM == 1 && mosquita[i].DdV > tpupad){ 
	  	 	 if(ran2(&semilla) < morad)mosquita[i].VoM=0;  
		}

	//printf("%d\n",mosquita[i].DdV);
	}  //del indice 
// -------------------------------------------------------------------	    
// NACIMIENTOS CON SATURACION
//-------------------------------------------------------------------------- 
	for(int i =1;i <= indice;i++){

		if (tach[mosquita[i].tacho] < sat){ 

			if(mosquita[i].VoM == 1 && mosquita[i].DdV > tpupad){

				  if(mosquita[i].DdV%tovip == 0){

   					for(int ik=1;ik <= iovip;ik++){ 

 					indice=indice+1; 
 					mosquita[indice].VoM=1;
 					mosquita[indice].DdV=1;   
 					mosquita[indice].tacho=mosquita[i].tacho; 
 					mor=ran2(&semilla)*6+27; 
	         			mosquita[indice].DdM=mor;
 					tach[mosquita[indice].tacho]=tach[mosquita[indice].tacho]+1;
   					}

  				   }    //del periodo entre oviposiciones y saturacion

			}      //de vivas y maduras

   		}   //de la saturac

	//printf("%d\n",tach[mosquita[i].tacho]);
	}  // de poblacion total (indice)

//--------------------------------------------------------------------------
//MUERTE POR VEJEZ: la mato cuando su edad alcanza el tiempo de vida asignado
//--------------------------------------------------------------------------
	for(int i =1;i <= indice;i++){

		if (mosquita[i].VoM == 1 && mosquita[i].DdV >= mosquita[i].DdM)mosquita[i].VoM=0;
	//printf("%d\n",mosquita[i].VoM);

	} 
//--------------------------------------------------------------------			  		 
//DESCACHARRADO: elimino los acuaticos de algunos tachos, 
//una vez por semana a partir del día 20 (esto se puede cambiar, obvio)
//-------------------------------------------------------------------- 
   	if(tiempo%7 == 0 && tiempo > 20){

  		for(int itach=1;itach <= descach;itach++){
    		ntach=ran2(&semilla)*ntachito;

			for(int i =1;i <= indice;i++){

  				if (mosquita[i].VoM == 1 && mosquita[i].DdV < tpupad && mosquita[i].tacho == ntach)mosquita[i].VoM=0;
				//printf("%d\n",mosquita[i].VoM);
			}

		}    
	
   	}

//--------------------------------------------------------------------			  		 
// cuento poblacion adulta y acuatica
//--------------------------------------------------------------------			  
  ad=0;  //adultas total
  ac=0;  //acuaticas vivas total
  ad1=0; //adultas vivas por tacho
  ad2=0;
  ad3=0;
  ad4=0;
  ad5=0;
  adn1=0; //adultas nuevas por tacho
  adn2=0;
  adn3=0;
  adn4=0;
  adn5=0;
  
  	for(int i =1;i <= indice; i++){
		if (mosquita[i].VoM == 1 && mosquita[i].DdV >= tpupad)ad=ad+1; //adultos tot
		if (mosquita[i].VoM == 1 && mosquita[i].DdV < tpupad)ac=ac+1; //acuaticos tot

		if (mosquita[i].VoM == 1 && mosquita[i].DdV >= tpupad && mosquita[i].tacho == 1)ad1=ad1+1;
		if (mosquita[i].VoM == 1 && mosquita[i].DdV >= tpupad && mosquita[i].tacho == 2)ad2=ad2+1;
		if (mosquita[i].VoM == 1 && mosquita[i].DdV >= tpupad && mosquita[i].tacho == 3)ad3=ad3+1;
		if (mosquita[i].VoM == 1 && mosquita[i].DdV >= tpupad && mosquita[i].tacho == 4)ad4=ad4+1;
		if (mosquita[i].VoM == 1 && mosquita[i].DdV >= tpupad && mosquita[i].tacho == 5)ad5=ad5+1;

		if (mosquita[i].VoM == 1 && mosquita[i].DdV == tpupad && mosquita[i].tacho == 1)adn1=adn1+1;
		if (mosquita[i].VoM == 1 && mosquita[i].DdV == tpupad && mosquita[i].tacho == 2)adn2=adn2+1;
		if (mosquita[i].VoM == 1 && mosquita[i].DdV == tpupad && mosquita[i].tacho == 3)adn3=adn3+1;
		if (mosquita[i].VoM == 1 && mosquita[i].DdV == tpupad && mosquita[i].tacho == 4)adn4=adn4+1;
		if (mosquita[i].VoM == 1 && mosquita[i].DdV == tpupad && mosquita[i].tacho == 5)adn5=adn5+1;
	}

//--------------------------------------------------------------------	   
//envejezco a toda la población un día (aumento en 1)
//--------------------------------------------------------------------
   	for(int i =1;i <= indice;i++){

	 	 if (mosquita[i].VoM == 1)mosquita[i].DdV=mosquita[i].DdV+1;
	//printf("%d\n",mosquita[i].DdV);
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
}// end for main

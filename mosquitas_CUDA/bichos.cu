#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include "ran2.h"
#include <cmath>
#include <thrust/host_vector.h>
#include "cpu_timer.h"

// macros utiles para CPU y GPU, y hacer desaparecer numeros magicos
#define Ninicial	    	5
#define NUMEROTACHOS		5
#define MAXIMONUMEROBICHOS	800000

#define ESTADOMUERTO		1
#define ESTADOVIVO		    0
#define Ndias			    400

#define ntachito		    5
#define morhue 			    0.01   	//mortalidad de huevos
#define morlar 			    0.01    //mortalidad de larvas 
#define morpup 			    0.01   	//mortalidad de pupas
#define morad 			    0.01 	//mortalidad diaria adultas
#define moracu 			    0.03	//morhue+morlar+morpup;
#define morpupad 		    0.17	//pupas que no se vuelven adultas
#define tpupad	 		    17  	//pupas se vuelven adultas a los 17 dias en invierno****
#define tovip1 			    2	    //tiempo entre dos oviposiciones (T=30)
#define tovip2a 		    3  	    //tiempo entre dos oviposiciones (T=25)
#define tovip2b			    4   	//tiempo entre dos oviposiciones (T=25)
#define tovip3 			    30	    //tiempo ente dos oviposiciones (T=18)
#define sat 			    800     //saturación de huevos por tacho
#define prop 			    0.6	    //efectividad de la propaganda

long semilla = -975;  			//semilla para el generador de numeros aleatorios ran2()

int tiempo_entre_oviposiciones(int dia){
	int t;
	//defino tiempos de oviposición y maduración en función de la temperatura	
	if(dia >= 1 && dia < 80) t=tovip3;      //<T>=18 // acá es necesario definirlo tomando el extremo
	if(dia >= 80 && dia <= 140)t=tovip2b;   //<T>=23
 	//if(dia < 140 || dia > 260) t=tovip1;    //<T>=30 //acá pasa algo raro con el rango donde esta definido tovip
  	if(dia > 140 || dia < 260) t=tovip1;    //<T>=30
 	if(dia >= 260 && dia <= 320)t=tovip2a;  //<T>=27
 	if(dia > 320) t=tovip3;                 //<T>=18

	return t;}

struct bichos{

	thrust::host_vector<int> estado;  
	thrust::host_vector<int> edad;    
	thrust::host_vector<int> tacho;   
	thrust::host_vector<int> TdV; 
	thrust::host_vector<int> pupacion; 
	thrust::host_vector<int> manzana; 
	thrust::host_vector<int> tach; 

	thrust::host_vector<int> N_mobil; 

	//constructor	
	bichos(int N_){
	// alocamos el maximo posible
	estado.resize(MAXIMONUMEROBICHOS);	
	tacho.resize(MAXIMONUMEROBICHOS);	
	edad.resize(MAXIMONUMEROBICHOS);
	pupacion.resize(MAXIMONUMEROBICHOS);
	TdV.resize(MAXIMONUMEROBICHOS);
	manzana.resize(MAXIMONUMEROBICHOS);
	tach.resize(MAXIMONUMEROBICHOS);

	N_mobil.resize(1);

	std::fill(estado.begin(),estado.end(),0);
	std::fill(edad.begin(),edad.end(),0);
	std::fill(tacho.begin(),tacho.end(),0);
	std::fill(pupacion.begin(),pupacion.end(),0);
	std::fill(TdV.begin(),TdV.end(),0);
	std::fill(manzana.begin(),manzana.end(),0);

	/*condiciones iniciales*/
 	for(int i=0;i < N_;i++){
		estado[i]=ESTADOVIVO; 		//todos vivos inicialmente
		tacho[i]=i;				//tacho en el que se encuentra la mosquita
		edad[i]=ran2(&semilla)*7+19; 		//edad 
		pupacion[i]=tpupad-2+(ran2(&semilla)*5);//dia de pupacion (entre los 15 y 19 dias)
		TdV[i]=ran2(&semilla)*6+27 ;	//tiempo de vida de 27 a 32
		std::cout << estado[i] << "\t" << tacho[i] << "\t" << edad[i] << "\t" << TdV[i] << "\t" << pupacion[i] << "\n";
	}

	N_mobil[0]=N_;
	};	

	
	void mortalidades_varias(int dia){

	int N=N_mobil[0];
	//mortalidades varias
		for(int i=0;i < N;i++){
			if (estado[i] == ESTADOVIVO && edad[i] < pupacion[i]){if(ran2(&semilla) < moracu)estado[i]=ESTADOMUERTO;}
			if (estado[i] == ESTADOVIVO && edad[i] == pupacion[i]){if(ran2(&semilla) < morpupad)estado[i]=ESTADOMUERTO;}
			if (estado[i] == ESTADOVIVO && edad[i] > pupacion[i]){if(ran2(&semilla) < morad)estado[i]=ESTADOMUERTO;}  
		}  //del indice 
	};

	void muerte_x_vejez(int dia){
	int N=N_mobil[0];
	//muerte por vejez
		for(int i=0;i < N;i++){if (estado[i] == ESTADOVIVO && edad[i] >= TdV[i])estado[i]=ESTADOMUERTO;} 

	};

	void descacharrado(int dia,int descach){
	int N=N_mobil[0];	
	//descachrarrado
		if(dia%7 == 0 && dia > 120 && dia < 320){
  			for(int itach=0;itach < descach;itach++){
    			int ntach=ran2(&semilla)*ntachito;
			//printf("%d\n",ntach);
				for(int i=0;i < N;i++){
  					if (estado[i] == ESTADOVIVO && edad[i] < pupacion[i] && tacho[i] == ntach)estado[i]=ESTADOMUERTO;
				}
			}    	
   		}
	};

	void conteo_huevos(int dia){
	int N=N_mobil[0];
	//conteo de huevos
		for(int i=0;i < N; i++){ 
			tach[i]=0;
   			if(edad[i] < pupacion[i] && estado[i] == ESTADOVIVO){ 
    			int j=tacho[i]; 
	    		tach[j]++;
			}
		} 
	};

	void reproducir(int dia,int tovip){	
	int indice=N_mobil[0];
	//nacimientos
	int mosqsat=0;
		for(int i=0;i < indice;i++){
			if(estado[i] == ESTADOVIVO && edad[i] > pupacion[i] && edad[i]%tovip == 0){
etiqueta:			if (tach[tacho[i]] < sat){ 
//				if (tach[tacho[i]] < sat){ 
 					  int iovip=10 + (ran2(&semilla)*25); 
   						for(int ik=0;ik < iovip;ik++){ 
 						estado[indice]=ESTADOVIVO;
 						edad[indice]=1;   
 						tacho[indice]=tacho[i]; 
		         			pupacion[indice]=tpupad-2+(ran2(&semilla)*5);	//dias de pupacion
	         				TdV[indice]=ran2(&semilla)*6+27;  
						int j=tacho[indice];
 						tach[j]++;
						indice++;
   						} 
				}
				else{
			        mosqsat++;   			//sumo las mosquitas que no pudieron poner en este tiempo (solo como dato)
				         for(int j=0;j < ntachito;j++){      //si no tiene lugar en su tacho migra a otro 
			          		if(tach[j] < sat){   	     // se fija si sus huevos van a tener lugar 
			           		tacho[i]=j;          	     //se mueve
					        goto etiqueta;               //y arranca a oviponer
			        		}
			       		 }
				}
  
   			} 
		}
		// actualiza el numero de bichos si no se sobrepasa el maximo
		N_mobil[0]=indice;

	};
	
    //Recalcular -> eliminar muertos y dejar vivos
    void recalcularN(){

		auto zip_iterator=
		thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),tacho.begin(),pupacion.begin(),TdV.begin(),tach.begin(),manzana.begin()));
		// ordenamos segun estado 0-vivo, 1-muerto
		int N=N_mobil[0];
		thrust::sort_by_key(estado.begin(), estado.begin() + N,zip_iterator);		
	
		// y ahora determinamos la posicion del primer muerto = N_mobil
		auto iter=thrust::find(estado.begin(),estado.begin() + N, ESTADOMUERTO);
		N_mobil[0]= iter-estado.begin();//me da la longitud del vector
	};
	
	// Numero de bichos vivos
	int vivos(int dia){

	int N=N_mobil[0];
	int poblacion=0;

  		for(int i=0;i < N; i++){
			if (estado[i] == ESTADOVIVO)poblacion++;	
		}	

		return poblacion;
	};

	int acuaticos(int dia){

	int N=N_mobil[0];
  	int ac=0;  //acuaticas vivas total

  	for(int i =0;i < N; i++){
		if (estado[i] == ESTADOVIVO && edad[i] < pupacion[i])ac++; //acuaticos tot
	}

		return ac;
	};

	int adultos(int dia){

	int N=N_mobil[0];
  	int ad=0;  //adultas total

  	for(int i =0;i < N; i++){
		if (estado[i] == ESTADOVIVO && edad[i] >= pupacion[i])ad++; //adultos tot	
	}

		return ad;
	};

	void envejecer(int dia){
	int N=N_mobil[0];
	//envejecer poblacion
  		for(int i=0;i < N;i++){
			if(dia < 80 || dia > 320){//en invierno envejecen ad, hibernan huev y acu crecen lento

    				if(estado[i] == ESTADOVIVO && edad[i] > pupacion[i])edad[i]++;} //ADULTAS

			else{if(estado[i] == ESTADOVIVO)edad[i]++;} 

		}
	};
};



int main(){

	std::ofstream outfile, outfile1, outfile2;
    	outfile.open("Poblacion_total_CPU.dat");
   	outfile1.open("Poblacion_adultos_CPU.dat");
    	outfile2.open("Poblacion_acuaticos_CPU.dat");

	int descach=round(ntachito*prop);//cantidad de tachos que vacío con la propaganda

	cpu_timer Reloj_CPU;
	Reloj_CPU.tic();

    bichos mosquitas(Ninicial);

	for(int dia = 1; dia <= Ndias; dia++){
	int tovip=tiempo_entre_oviposiciones(dia);

	mosquitas.mortalidades_varias(dia);
	mosquitas.muerte_x_vejez(dia);
	mosquitas.descacharrado(dia,descach);
	mosquitas.conteo_huevos(dia);
	mosquitas.reproducir(dia,tovip);
	mosquitas.recalcularN();//anula todo el cálculo

	int vivas=mosquitas.vivos(dia);
	int adultos=mosquitas.adultos(dia);
	int acuaticos=mosquitas.acuaticos(dia);
	outfile << dia << "\t" << vivas << std::endl;
	outfile1 << dia << "\t" << adultos << std::endl;
	outfile2 << dia << "\t" << acuaticos << std::endl;
	mosquitas.envejecer(dia); //depende del orden
	}//cierro loop para dias
    double t=Reloj_CPU.tac()/60000; //de milisegundos -> minutos
    printf("Tiempo en CPU: %lf minutos\n",t);
//cierro archivos
outfile.close();
outfile1.close();
outfile2.close();
return 0;							
}// end for main


#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include "ran2.h"
#include <cmath>
#include <thrust/device_vector.h>
#include "gpu_timer.h"
#include "parametros.h"

int tiempo_entre_oviposiciones(int dia){
	int t;
	//defino tiempos de oviposiciรณn y maduraciรณn en funciรณn de la temperatura	
	if(dia >= 1 && dia < 80) t=tovip3;      //<T>=18 // acรก es necesario definirlo tomando el extremo
	if(dia >= 80 && dia <= 140)t=tovip2b;   //<T>=23
  	if(dia > 140 || dia < 260) t=tovip1;    //<T>=30
 	if(dia >= 260 && dia <= 320)t=tovip2a;  //<T>=27
 	if(dia > 320) t=tovip3;                 //<T>=18

	return t;}


__global__ void kernel_reproducir(int *estado, int *edad, int *tacho,int *TdV, int *pupacion,int *manzana, int *N_mobil, int dia, int tovip, int *nacidos)
{
    	int indice=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

  		if(id < indice)
  		{
  		RNG philox;         
	    RNG::ctr_type c={{}};
	    RNG::key_type k={{}};
	    RNG::ctr_type r;
	    k[0]=id; 
	    c[1]=dia;
	    c[0]=SEMILLAGLOBAL; 
		
    	r = philox(c, k); 
     	double azar=(u01_closed_closed_32_53(r[0]));
/*Si la mosquita esta viva, esta en edad adulta y en el tiempo de oviposicion entonces*/

			if(estado[id] == ESTADOVIVO && edad[id] > pupacion[id] && edad[id]%tovip == 0){
/*Si el tacho en el que nacio tiene lugar entonces*/
				/*Antes estaba asi y andaba...*/
				int tach=tacho[id];     //tach es un entero que me indica el numero de tacho en el que esta cada mosquita
				
				 int iovip=1;
				 //10 + (azar*25); //iovip es el numero de huevos que pone cada mosquita
						
				atomicAdd(nacidos+tach,iovip); /*sumo iovip HUEVOS en la posicion del vector nacidos (que tiene NTACHOS elementos)
				nacidos[0+tach] en el puntero al primer elemento del vector nacidos desplazado en tach elementos
				el vector nacidos tiene el numero de nacidos en cada tacho Ej: nacidos[0]=numero de nacidos en el tacho 1*/
				
				
		    }//cierro loop para mosquitasvivas y  maduras
  		}//cierro loop de hilos    
};

//mortalidades varias	
__global__ void matar_kernel(int *estado, int *edad, int *tacho,int *pupacion, int *TdV,int *N_mobil, int dia)
{
	int N=N_mobil[0];
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id<N){		
	 	RNG philox;         
	    RNG::ctr_type c={{}};
	    RNG::key_type k={{}};
	    RNG::ctr_type r;
	    k[0]=id; 
	    c[1]=dia;
	    c[0]=SEMILLAGLOBAL; 
		
    	r = philox(c, k); 
     	double azar=(u01_closed_closed_32_53(r[0]));
     	//acá incluye mortalidad de huevos,pupas y adultos con cierta probabilidad, y además muertes por vejez
		    if (estado[id]==ESTADOVIVO && edad[id] < pupacion[id]){if(azar < moracu)estado[id]=ESTADOMUERTO;}
		    if (estado[id]==ESTADOVIVO && edad[id] == pupacion[id]){if(azar < morpupad)estado[id]=ESTADOMUERTO;}
		    if (estado[id]==ESTADOVIVO && edad[id] > pupacion[id]){if(azar < morad)estado[id]=ESTADOMUERTO;}
		    if (estado[id]==ESTADOVIVO && edad[id] >= TdV[id])estado[id]=ESTADOMUERTO;
	}
};

//muerte de las mosquitas por vejez
/*__global__ void matar_viejos_kernel(int *estado, int *edad, int *tacho, int *TdV,int *N_mobil,int dia)
{

	    int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

	    if(id<N){                     
	        if(estado[id]==ESTADOVIVO && edad[id] >= TdV[id])estado[id]=ESTADOMUERTO;
	    }
};
*/
__global__ void descacharrado_kernel(int *estado, int *edad, int *tacho, int *pupacion,int *N_mobil,int dia, int ntach)
{
    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

	    if(id<N){
	    if (estado[id] == ESTADOVIVO && edad[id] < pupacion[id] && tacho[id] == ntach)estado[id]=ESTADOMUERTO;
    	}
};

__global__ void envejecer_kernel(int *estado, int *edad,int *pupacion,int *N_mobil,int dia)
{
    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

        if(id<N){
            if(dia < 80 || dia > 320){
                if(estado[id] == ESTADOVIVO && edad[id] > pupacion[id])edad[id]++;} //ADULTAS
  		    else{
  		        if (estado[id]== ESTADOVIVO)edad[id]++;}
    	}
};

//es muy lento
/*__global__ void conteo_kernel(int *estado, int *edad,int *tacho,int *pupacion, int *tach,int *N_mobil)
{
       	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;
	    
		if(id<N){
			if(edad[id] < pupacion[id] && estado[id] == ESTADOVIVO){ 
    			int j=tacho[id]; 
	    		tach[j]++;
			}
		} 
};
*/


// otro functorcito usado para las estadisticas desagregadas
struct acuaticoeneltacho{
	int m;
    int t;
	acuaticoeneltacho(int m_, int t_):m(m_),t(t_){};
    
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int tach=thrust::get<0>(tupla);
        int edad=thrust::get<1>(tupla);
		return (tach==m && edad<t);
	}
};

// functorcito para contar adultos en la población
struct poblacion_1{
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int pupacion=thrust::get<0>(tupla);
        int edad=thrust::get<1>(tupla);
		return (edad >= pupacion);
	}
};

// functorcito para contar acuáticos en la población
struct poblacion_2{
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int pupacion=thrust::get<0>(tupla);
        int edad=thrust::get<1>(tupla);
		return (edad < pupacion);
	}
};

/////////////////////////////////////////////////////////////////////////////////////////
// Clase bichos: toda la info sobre todos los bichos, y sus funciones
struct bichos{

	thrust::device_vector<int> estado; // Vivo o Muerto 0/1 
	thrust::device_vector<int> edad; // tiene num de mosqu elementos y los valores van de 0 a MAXIMAEDAD   
	thrust::device_vector<int> tacho; // numero de tacho en que se encuentra cada mosquita valores=0 a NUMEROTACHOS   
	thrust::device_vector<int> TdV;  //tiempo de vida de cada mosquita
	thrust::device_vector<int> pupacion; //dia de paso de pupa a adulta de cada mosquita
	thrust::device_vector<int> manzana; //numero de manzana de cada mosquita
	//thrust::device_vector<int> tach; //numero de tacho de cada mosquita
	thrust::device_vector<int> nacidos; // tiene el num de tachos elementos, numero de nacidos por tacho

	thrust::device_vector<int> N_mobil; // Numero de bichos fluctuante (1 elemento)

	// punteros crudos a los arrays para pasarselos a kernels
	int *raw_edad;
	int *raw_tacho;
	int *raw_estado;
	int *raw_TdV;
	//int *raw_tach;
	int *raw_pupacion;
	int *raw_N_mobil;
	int *raw_manzana;
	int *raw_nacidos;
	
	//constructor	
	bichos(int N_){
	// alocamos el maximo posible
	estado.resize(MAXIMONUMEROBICHOS);	
	tacho.resize(MAXIMONUMEROBICHOS);	
	edad.resize(MAXIMONUMEROBICHOS);
	pupacion.resize(MAXIMONUMEROBICHOS);
	TdV.resize(MAXIMONUMEROBICHOS);
	manzana.resize(MAXIMONUMEROBICHOS);
	//tach.resize(MAXIMONUMEROBICHOS);

	N_mobil.resize(1);

	// nacidos en cada tacho, inicialmente 0
	nacidos.resize(NUMEROTACHOS);
	thrust::fill(nacidos.begin(),nacidos.end(),0);

	thrust::fill(estado.begin(),estado.end(),0);
	thrust::fill(edad.begin(),edad.end(),0);
	thrust::fill(tacho.begin(),tacho.end(),0);
	thrust::fill(pupacion.begin(),pupacion.end(),0);
	thrust::fill(TdV.begin(),TdV.end(),0);
	thrust::fill(manzana.begin(),manzana.end(),0);
	
	// inicializacion raw pointers
	raw_edad=thrust::raw_pointer_cast(edad.data());
	raw_tacho=thrust::raw_pointer_cast(tacho.data());
	raw_estado=thrust::raw_pointer_cast(estado.data());
	raw_TdV=thrust::raw_pointer_cast(TdV.data());
	raw_manzana=thrust::raw_pointer_cast(manzana.data());
	raw_N_mobil=thrust::raw_pointer_cast(N_mobil.data());
	//raw_tach=thrust::raw_pointer_cast(tach.data());
	raw_pupacion=thrust::raw_pointer_cast(pupacion.data());
	raw_nacidos=thrust::raw_pointer_cast(nacidos.data());

    //std::cout<<"VoM\ttacho\tedad\tTdV\ttpupad\tmanzana" << std::endl;
	/*condiciones iniciales donde N sera el numero de bichos*/
 	for(int i=0;i < N_;i++){
		estado[i]=ESTADOVIVO; 		    //todos vivos inicialmente
		tacho[i]=i;				        //tacho en el que se encuentra la mosquita
		edad[i]=ran2(&semilla)*7+19; 	//edad 
		pupacion[i]=tpupad-2+(ran2(&semilla)*5);//dia de pupacion (entre los 15 y 19 dias)
		TdV[i]=ran2(&semilla)*6+27 ;	//tiempo de vida de 27 a 32
		manzana[i]=(int) (i/5);         //manzana en la que se encuentra
		
	//	std::cout << estado[i] << "\t" << tacho[i] << "\t" << edad[i] << "\t" << TdV[i] << "\t" << pupacion[i] << "\t" << manzana[i] << "\n";
	 //std::cout << "pupacion" << pupacion[i]<< std::endl;
}

std::cout << "inicializacion lista" << std::endl;

	N_mobil[0]=N_;
	};	

	
	void mortalidades(int dia){

	int N=N_mobil[0];
	
	    matar_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_pupacion,raw_TdV,raw_N_mobil, dia);	
		cudaDeviceSynchronize();
	};
	
	//descachrarrado
	void descacharrado(int dia,int descach){
	int N=N_mobil[0];	
		if(dia%7 == 0 && dia > 120 && dia < 320){
  			for(int itach=0;itach < descach;itach++){
    			int ntach=ran2(&semilla)*NUMEROTACHOS;
  				descacharrado_kernel<<<(N+256-1)/256,256>>>(raw_estado, raw_edad, raw_tacho, raw_pupacion, raw_N_mobil,dia,ntach);
			}    	
   		}
	};

	//conteo de huevos
	void conteo_huevos(int dia){
	int N=N_mobil[0];
//		thrust::fill(tach.begin(),tach.begin() + N,0);
	    //demora demasiado
		//conteo_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_pupacion,raw_N_mobil);
		//cudaDeviceSynchronize();
		
 	/* 	for(int i=0;i < N; i++){ 
			if(edad[i] < pupacion[i] && estado[i] == ESTADOVIVO){ 
    			int j=tacho[i]; 
	    		tach[j]++;
			}
		} 
 */
	};

    //nacimientos
	void reproducir(int dia,int tovip){
	    
	int indice=N_mobil[0];
	if(indice==0) {
		std::cout << "NO HAY MAS MOSQUITAS PARA REPRODUCIRSE" << std::endl; 	
	exit(1);
	}else{
	//nacimientos
	int mosqsat=0;
	std::cout << "antes kernel reproducir " << std::endl;
	//reinicializo en cero los nacidos en el paso anterior que ahora ya no son mas nacidos porque crecieron 
	thrust::fill(nacidos.begin(),nacidos.end(),0);
	// reproduce, calculando nacidos por tacho antes
	kernel_reproducir<<<(indice+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_TdV,raw_pupacion,raw_manzana,raw_N_mobil,dia,tovip,raw_nacidos);
	cudaDeviceSynchronize();
	std::cout << "despues kernel reproducir " << std::endl; 
		// agrego todos los nacidos al final del array original, tacho a tacho
		int index=indice;
        bool nosaturo=1;
		for(int m=0;m<NUMEROTACHOS;m++){
			//std::cout << "listo nacidos " << std::endl; 
 
            //antiguos=thrust::count_if(tacho.begin(),tacho.begin()+N,iguala(m));
//calculo el nunmero de acuaticos en cada tacho
            int antiguos=thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(),edad.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(tacho.begin()+indice,edad.begin()+indice)),
                acuaticoeneltacho(m,tpupad)
			);
			std::cout << "Acuaticos en Tacho " << antiguos <<std::endl;
          	//los nuevos vienen del kernel reproducir  
            int nuevos=nacidos[m];
			std::cout << "NUEVOS NACIDOS " << nuevos <<std::endl; 
			//no saturo me asegura que no me excedo del tamaño del array
            nosaturo=((index+nuevos) < MAXIMONUMEROBICHOS);
            if((antiguos+nuevos) < SAT && nosaturo==1){ 						//cond. saturacion 
            thrust::fill(estado.begin()+index,estado.begin()+index+nuevos,0);	//todos vivos(0)	
			thrust::fill(edad.begin()+index,edad.begin()+index+nuevos,0);		//todos nacen con edad 0	
			thrust::fill(tacho.begin()+index,tacho.begin()+index+nuevos,m); 	//nacen en el tacho m
			thrust::fill(TdV.begin()+index,TdV.begin()+index+nuevos,rand()%3+28); // con tiempo de vida 28 a 30
			index+=nuevos;		//actualizo el indice para me marque siempre en la ultima mosquita que nacio 
			}else if((antiguos+nuevos) > SAT){
				std::cout << "Nos pasamos del maximo por tacho" <<std::endl;
			}else{std::cout << "Nos pasamos del maximo de bichos totales" << nuevos <<std::endl;}
		}
	// problema si esto satura saturan los tachos mas grandes, ver como cambiar esto por ejemplo llnando tachos al azar
        if(nosaturo==0) std::cout << "algunos tachos no se rellenaron por saturacion del total de mosquitas" << std::endl;
	//imprimo el indice
	std::cout << "index " << index <<std::endl; 
	
	// actualiza el indice movil hasta el ultimo bicho vivo
		if(index<MAXIMONUMEROBICHOS) {
			N_mobil[0]=index;
		}
		// caso contrario satura al maximo
		else{
			std::cout << "Demasiados Bichos!" << std::endl;
			 N_mobil[0]=MAXIMONUMEROBICHOS-1;	
			}	
	}	
	
	/* 	for(int i=0;i < indice;i++){
			if(estado[i] == ESTADOVIVO && edad[i] > pupacion[i] && edad[i]%tovip == 0){
				if (tach[tacho[i]] < sat){
					  int iovip=10 + (ran2(&semilla)*25); 
   						for(int ik=0;ik < iovip;ik++){ 
 						estado[indice]=ESTADOVIVO;
 						edad[indice]=1;   
 						tacho[indice]=tacho[i]; 
		         		pupacion[indice]=tpupad-2+(ran2(&semilla)*5);	//dias de pupacion
	         			TdV[indice]=ran2(&semilla)*6+27;  
						manzana[indice]=manzana[i];
						int j=tacho[indice];
 						tach[j]++;
						indice++;
   						} 
				}//cierro tach
				else{	
			        mosqsat++;   			//sumo las mosquitas que no pudieron poner en este tiempo (solo como dato)
				         for(int j=0;j < ntachito;j++){      //si no tiene lugar en su tacho migra a otro 
			          		if(tach[j] < sat){   	     // se fija si sus huevos van a tener lugar 
			           		tacho[i]=j;          	     //se mueve
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
						j=ntachito;
						}
					}
				} 				
   			} 
		} */
//		actualiza el numero de bichos si no se sobrepasa el maximo
		//N_mobil[0]=indice;

	};

    //Recalcular -> eliminar muertos y dejar vivos
    void recalcularN(){

		auto zip_iterator=
		thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),tacho.begin(),pupacion.begin(),TdV.begin(),manzana.begin()));
		// ordenamos segun estado 0-vivo, 1-muerto
		int N=N_mobil[0];
		thrust::sort_by_key(estado.begin(), estado.begin() + N,zip_iterator);		
		//std::cout << "N en recalcular N " << N <<std::endl; 
		// y ahora determinamos la posicion del primer muerto = N_mobil
		auto iter=thrust::find(estado.begin(),estado.begin() + N, ESTADOMUERTO);
		
		N_mobil[0]= iter-estado.begin();//me da la longitud del vector
		std::cout << "N_mobil " << N_mobil[0] <<std::endl;
	};
	
	// Numero de mosquitas vivas
	int vivos(int dia){

	int N=N_mobil[0];

    int poblacion = thrust::count(estado.begin(), estado.begin() + N, ESTADOVIVO);
	return poblacion;
	};

    //población de acuáticos
	int acuaticos(int dia){

	int N=N_mobil[0];
	
    int ac=thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),pupacion.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin() +  N,pupacion.begin() + N)),
                poblacion_1()
            );

	return ac;
	};

    //población de adultos
	int adultos(int dia){

	int N=N_mobil[0];
	    int ad=thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),pupacion.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin() +  N,pupacion.begin() + N)),
                poblacion_2()
            );

		return ad;
	};

	//envejecer población

   	void envejecer(int dia){
	int N=N_mobil[0];
        envejecer_kernel<<<(N + 256-1)/256,256>>>(raw_estado,raw_edad,raw_pupacion,raw_N_mobil,dia);
	}; 

	// Envejezco a toda la población un día (aumento en 1): recorre los bichos y les aumenta la edad en una unidad
	void envejecerthrust(){
	int N=N_mobil[0];
	using namespace thrust::placeholders;
	//envejecen las que estan vivas por eso hasta N 	
	thrust::transform(edad.begin(),edad.begin()+N, edad.begin(),_1+1);
	
	};
};


int main(){

 	std::ofstream outfile, outfile1, outfile2;
   	outfile.open("Poblacion_total_GPU.dat");
   	outfile1.open("Poblacion_adultos_GPU.dat");
    outfile2.open("Poblacion_acuaticos_GPU.dat");

	int descach=round(NUMEROTACHOS*prop);//cantidad de tachos que vacรญo con la propaganda 
	
	gpu_timer Reloj_GPU;
	Reloj_GPU.tic();
    
    bichos mosquitas(Ninicial);

	for(int dia = 1; dia <= Ndias; dia++){
	int tovip=tiempo_entre_oviposiciones(dia);
	
	std::cout << "reproducir" << std::endl;
	mosquitas.reproducir(dia,tovip);
	
	std::cout << "matar" << std::endl;
	mosquitas.mortalidades(dia);//fusione muerte x vejez con mortalidades varias en un solo kernel
	//mosquitas.muerte_x_vejez(dia);
	
	std::cout << "descacharrar" << std::endl;
	mosquitas.descacharrado(dia,descach); 
	
	//Esto no se para que sirve
	//mosquitas.conteo_huevos(dia);
	std::cout << "envejezco poblacion" << std::endl;
	//mosquitas.envejecer(dia);
	mosquitas.envejecerthrust();
	
	std::cout << "recalculo indice de mosquitas vivas" << std::endl;
	mosquitas.recalcularN(); 

	std::cout << "DIA" << dia << std::endl;

	int vivas=mosquitas.vivos(dia);
	int adultos=mosquitas.adultos(dia);
	int acuaticos=mosquitas.acuaticos(dia);
	outfile << dia << "\t" << vivas << std::endl;
	outfile1 << dia << "\t" << adultos << std::endl;
	outfile2 << dia << "\t" << acuaticos << std::endl;
	 

	}

/* 	void avanza_dia(int dia)
    {
        std::cout << "reproducir" << std::endl;
		reproducir(dia); 		//nacimientos

        std::cout << "matar" << std::endl;
        matar(dia);			//mortalidades varias

        matar_viejos(dia);		//mortalidad por vejez
		descacharrar_tacho(dia);	//descacharrar tacho
		envejecer();			//envejecer un dia la poblacion
		recalcularN();
	};
 */
	

    double t=Reloj_GPU.tac()/60000; //de milisegundos -> minutos
    printf("Tiempo en GPU: %lf minutos\n",t);

return 0;

//cierro archivos
outfile.close();
outfile1.close();
outfile2.close();
}// end for main


#include <philox.h> // philox headers
#include <u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG

#include <curand.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include "ran2.h"
#include <cmath>
#include <thrust/device_vector.h>
#include "gpu_timer.h"
#include "parametros.h"

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)
    
std::ofstream outfile, outfile1, outfile2,outfile3,outfile4,outfile5;
	
int tiempo_entre_oviposiciones(int dia){
	int t;
	//defino tiempos de oviposición y maduración en función de la temperatura	
	if(dia >= 1 && dia < 80) t=TOVIP3;      //<T>=18 // acá es necesario definirlo tomando el extremo
	if(dia >= 80 && dia <= 140)t=TOVIP2b;   //<T>=23
  	if(dia > 140 || dia < 260) t=TOVIP1;    //<T>=30
 	if(dia >= 260 && dia <= 320)t=TOVIP2a;  //<T>=27
 	if(dia > 320) t=TOVIP3;                 //<T>=18

	return t;}

__global__ void kernel_reproducir(int *estado, int *edad, int *tacho,int *TdV, int *pupacion,int *manzana, int *N_mobil, int dia, int tovip, int *nacidos)
{                                   

    	int indice=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;
		/*Si la mosquita esta viva, esta en edad adulta y en el tiempo de oviposicion entonces*/
  		if(id < indice && edad[id] > pupacion[id] && edad[id]%tovip == 0) //calculo todo solo sobre las vivas 
  		{
			RNG philox;         
			RNG::ctr_type c={{}};
			RNG::key_type k={{}};
			RNG::ctr_type r;
			k[0]=id; 
			c[1]=dia;
			c[0]=SEMILLAGLOBAL; 
			
			r = philox(c, k); 
			double azar=(u01_closed_open_32_53(r[0]));//numero aleatorios entre [0,1)

			// estado[id] == ESTADOVIVO && edad[id]%tovip == 0){
			/*Si el tacho en el que nacio tiene lugar entonces*/
				/*Antes estaba asi y andaba...*/
			int tach=tacho[id];     //tach es un entero que me indica el numero de tacho en el que esta cada mosquita
				

			int iovip=10+ (azar*25); //iovip es el numero de huevos que pone cada mosquita
						
			atomicAdd(nacidos+tach,iovip); /*sumo iovip HUEVOS en la posicion del vector nacidos (que tiene NTACHOS elementos)
				nacidos[0+tach] en el puntero al primer elemento del vector nacidos desplazado en tach elementos
				el vector nacidos tiene el numero de nacidos en cada tacho Ej: nacidos[0]=numero de nacidos en el tacho 1*/
				
				
		    //cierro loop para mosquitas maduras
  		}//cierro loop de hilos de mosquitas VIVAS   
};

//mortalidades varias	
__global__ void matar_kernel(int *estado, int *edad, int *tacho,int *pupacion, int *TdV,int *N_mobil, int dia)
{
	int N=N_mobil[0];
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id<N){	// es lo mismo que estado[id]==ESTADOVIVO && 	
	 	RNG philox;         
	    RNG::ctr_type c={{}};
	    RNG::key_type k={{}};
	    RNG::ctr_type r;
	    k[0]=id; 
	    c[1]=dia;
	    c[0]=SEMILLAGLOBAL; 
		
    	r = philox(c, k); 
    	double azar;
    	/*CORRECCIÓN: ahora sorteo tres nros al azar y  comparo con las mortalidades,en coherencia a lo que hacía la función en SERIAL*/
    	
     	azar=(u01_closed_open_32_53(r[0]));//numero aleatorios entre [0,1)
     	if (edad[id] < pupacion[id]){if(azar < MORACU)estado[id]=ESTADOMUERTO;}

     	azar=(u01_closed_open_32_53(r[0]));
     	if (edad[id] == pupacion[id]){if(azar < MORPUPAD)estado[id]=ESTADOMUERTO;}

     	azar=(u01_closed_open_32_53(r[0]));
		if (edad[id] > pupacion[id]){if(azar < MORAD)estado[id]=ESTADOMUERTO;}     	

        //envejecer
		if (edad[id] >= TdV[id]){estado[id]=ESTADOMUERTO;}
	}
};

__global__ void descacharrado_kernel(int *estado, int *edad, int *tacho, int *pupacion,int *N_mobil,int dia, int ntach)
{
    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

	    if(id<N){// esta condicion es igual a decir que la mosquita estรก viva
	    if (edad[id] < pupacion[id] && tacho[id] == ntach)estado[id]=ESTADOMUERTO;
    	}
};

//elimine el estado[id]=ESTADOVIVO, ya que al final sólo quedan las mosquitas vivas
__global__ void envejecer_kernel(int *estado, int *edad,int *pupacion,int *N_mobil,int dia)
{
    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

        if(id<N){
            if(dia < 80 || dia > 320){
                if(edad[id] > pupacion[id])edad[id]++;} //ADULTAS
  		    else{
  		        edad[id]++;}
    	}
};


struct transferirdetacho{
	int m;
	int tpupad;
	int* ptr;
	int cuantos;
	int dia;

	transferirdetacho(int m_,int tpupad_,int *ptr_,int cuantos_, int dia_):
	m(m_),tpupad(tpupad_),ptr(ptr_),cuantos(cuantos_), dia(dia_){};

	__device__ 
	int operator()(thrust::tuple<int,int> tup, int counter){
		int tachoactual=thrust::get<0>(tup);
		int edad=thrust::get<1>(tup);
		 
		int tachonuevo=tachoactual; //me parece que esto debería ser sólo "int  tachonuevo";
		if(tachoactual==m && edad>=tpupad)
		{
			/* sortear nuevo tacho de la misma manzana*/
			RNG philox;         
			RNG::ctr_type c={{}};
			RNG::key_type k={{}};
			RNG::ctr_type r;
			k[0]=counter; 
			c[1]=dia;
			c[0]=SEMILLAGLOBAL; 

			r = philox(c, k); 
			float azar=(u01_closed_closed_32_53(r[0])); //distribución uniforme de nro randoms entre [0,1]
			int indicedetachoelegido=int(azar*cuantos);//cuantos: nro de tachos en la manzana

			
			//tachonuevo=tachoactual; // este es solo para test

			if(indicedetachoelegido<cuantos)// agregar acá que tiene indicedeltachonuevo != m
			tachonuevo=ptr[indicedetachoelegido];
		}
		return tachonuevo;
	}
};

// struct para generar randoms uniformes
struct uniformRanInt{
	int dia;
	int medio;
	int ancho;
	uniformRanInt(int medio_, int ancho_, int dia_):medio(medio_),ancho(ancho_),dia(dia_)
	{};

	__device__ int operator()(int i)
	{
	 	RNG philox;         
	    RNG::ctr_type c={{}};
	    RNG::key_type k={{}};
	    RNG::ctr_type r;
	    k[0]=i; 
	    c[1]=dia;
	    c[0]=SEMILLAGLOBAL; 

		r = philox(c, k); 
     	float azar=(u01_closed_open_32_53(r[0]));//numero aleatorios entre [0,1)
		return int(medio+ancho*azar);
	}
};	


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

struct aereaeneltacho
{
	int m;
    int t;
	aereaeneltacho(int m_, int t_):m(m_),t(t_){};   
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int tach=thrust::get<0>(tupla);
        int edad=thrust::get<1>(tupla);
		return (tach==m && edad>t);
	}
};


struct acuaticoeneltachoANA{
	int m;

	acuaticoeneltachoANA(int m_):m(m_){};
    
	__device__ bool operator()(thrust::tuple<int,int,int> tupla)
	{
        int tacho=thrust::get<0>(tupla);
        int edad=thrust::get<1>(tupla);
        int pupacion=thrust::get<2>(tupla);
		return (tacho==m && edad < pupacion);
	}
};

//functorcito para contar adultos en la población
struct poblacion_1{
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int edad=thrust::get<0>(tupla);//NUEVO: cambie el orden de la tupla
        int pupacion=thrust::get<1>(tupla);
		return (edad >= pupacion);
	}
};

// functorcito para contar acuáticos en la población
struct poblacion_2{
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int edad=thrust::get<0>(tupla); //NUEVO:cambie el orden de la tupla
        int pupacion=thrust::get<1>(tupla);
		return (edad < pupacion);
	}
};

// un functorcito usado para las estadisticas desagregadas
struct iguala{
	int m;
	iguala(int m_):m(m_){};

	__device__ bool operator()(int man)
	{
		return man==m;
	}
};


/*
__global__ void tachosenmanzana(int manzananum)
{
    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;
	    if(id<N){// esta condicion es igual a decir que la mosquita está viva
	    if (manzana[id] == manzananum)vectachmanzana=tacho[id];
    	}
		return (lista tachos en esa manzana);
};
*/

struct bichos{

	// arrays grandes en device, numero_mosquitas elementos: info de cada mosquita
	thrust::device_vector<int> estado; // Vivo o Muerto 0/1 
	thrust::device_vector<int> edad; // tiene num de mosqu elementos y los valores van de 0 a MAXIMAEDAD   
	thrust::device_vector<int> tacho; // numero de tacho en que se encuentra cada mosquita valores=0 a NUMEROTACHOS   
	thrust::device_vector<int> TdV;  //tiempo de vida de cada mosquita
	thrust::device_vector<int> pupacion; //dia de paso de pupa a adulta de cada mosquita
	thrust::device_vector<int> manzana; //numero de manzana de cada mosquita

	// arrays medianos em device, numero_tachos elementos
	thrust::device_vector<int> nacidos; // tiene el num de tachos elementos, numero de nacidos por tacho

	// arrays mediano en host, numero_manzanas elementos
	std::vector<std::vector<int> > tachos_por_manzana; //tachos_por_manzana[i]=vector de tachos de manzana i 

	// numero de tachos elementos
	std::vector<int> manzana_del_tacho;

    	//array para almacenar nro de tachos por manzana
   	thrust::device_vector<int> NroTachos;
    
	// array de un elemento = device variable
	thrust::device_vector<int> N_mobil; // Numero de bichos fluctuante (1 elemento)

	// punteros crudos a los arrays para pasarselos a kernels
	int *raw_edad;
	int *raw_tacho;
	int *raw_estado;
	int *raw_TdV;
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

		N_mobil.resize(1);

		//tachos_por_manzana.resize(int(N_/5.0));
		tachos_por_manzana.resize(NUMEROMANZANAS);

		manzana_del_tacho.resize(MAXIMONUMEROBICHOS);

        NroTachos.resize(NUMEROMANZANAS);
        
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
		raw_pupacion=thrust::raw_pointer_cast(pupacion.data());
		raw_nacidos=thrust::raw_pointer_cast(nacidos.data());
	    
        outfile3 << "CONDICIONES INICIALES" << std::endl;
		outfile3 <<"estado\tedad\ttacho\tmanzana" << std::endl;
		/*condiciones iniciales donde N sera el numero de bichos*/
		for(int i=0;i < N_;i++){
			estado[i] = ESTADOVIVO; 		             //todos vivos inicialmente
			tacho[i] = i;				                 //tacho en el que se encuentra la mosquita
			edad[i] = ran2(&semilla)*7+19; 	             //edad: son todas adultas al principio 
			pupacion[i] = TPUPAD-2+(ran2(&semilla)*5);   //dia de pupacion (entre los 15 y 19 dias)
			TdV[i] = ran2(&semilla)*6+27 ;	             //tiempo de vida de 27 a 32
			manzana[i] = NUMEROMANZANAS*ran2(&semilla);  //manzana en la que se encuentra
			//manzana[i] = (int) (tacho[i]/5);            //manzana en la que se encuentra
			tachos_por_manzana[manzana[i]].push_back(tacho[i]);
			manzana_del_tacho[tacho[i]]=manzana[i];

			outfile3 << estado[i] << "\t\t" << edad[i] << "\t\t" << tacho[i] << "\t\t" << manzana[i] << "\n";
		}

		// una verificacion del llenado de tachos_por_manzana
		int nmanzanas=tachos_por_manzana.size();
		for(int i=0;i<nmanzanas;i++){
			outfile3 << "\n\n manzana " << i << "\n tachos: ";
			int ntachos=tachos_por_manzana[i].size();
			int contTach=0;
			for(int j=0;j<ntachos;j++){
				outfile3 << (tachos_por_manzana[i])[j] << ", ";//tacho que se encuentra en la manzana
				contTach++;
				NroTachos[i]=contTach; //contador para contar nro de tachos por manzana
			}
			outfile3 << "\n Nro de tachos en la manzana\t" << contTach << "\n";
			outfile3 << std::endl;
		}
		outfile3 << "\n";
		outfile3 << "manzana del tacho 16 " << manzana_del_tacho[16] << std::endl;
		outfile3 << "manzana del tacho 10 " << manzana_del_tacho[10] << std::endl;
		outfile3 << "manzana del tacho 1 " << manzana_del_tacho[1] << std::endl;
    	outfile3 << "\n";

        outfile3 << "NTachos x manzana" << std::endl;
        for(int i=0;i<nmanzanas;i++){
        outfile3 <<  NroTachos[i] << "\n";
        }

		outfile3 << "inicializacion lista:\n" << "Nro de Tachos\t" << NUMEROTACHOS << "\n" << "Nro de Manzanas\t" << NUMEROMANZANAS <<std::endl;
		N_mobil[0]=N_;
	};	

	// devuelve un numero de tacho random de la manzana m
	int nuevo_tacho_misma_manzana(int m){
		int ntachos=tachos_por_manzana[m].size();
		int r=int((rand()*1.0/RAND_MAX)*ntachos);		
		int nuevo_tacho=(tachos_por_manzana[m])[r];
		return nuevo_tacho;
	}
	
	void mortalidades(int dia){

	int N=N_mobil[0];

	    matar_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_pupacion,raw_TdV,raw_N_mobil, dia);	
		cudaDeviceSynchronize();
	};

	//descachrarrado

	//void descacharrado(int dia,float *E){//DESCOMENTAR para una efectividad de propaganda distinta para cada manzana y cte durante los dias de descacharrado. 

	void descacharrado(int dia){//DESCOMENTAR para una efectividad de propaganda distinta para cada manzana y que varia durante los dias de descacharrado. 

	//void descacharrado(int dia,int descach){//DESCOMENTAR para una efectividad fija 0.6, igual para todas las manzanas, y constante durante los dias de descacharrado.	
		/*if(dia%7 == 0 && dia > 120 && dia < 320){
  			for(int itach=0;itach < descach;itach++){
    			int ntach=ran2(&semilla)*NUMEROTACHOS;
    			std::cout << "tacho que se descacharran\t" << ntach << "\n";
  				descacharrado_kernel<<<(N+256-1)/256,256>>>(raw_estado, raw_edad, raw_tacho, raw_pupacion, raw_N_mobil,dia,ntach);
  				cudaDeviceSynchronize();
			}    	
   		}*/

                int N=N_mobil[0];

   		thrust::device_vector<int> D(28); //array para almacenar los dias que se descacharra
   		thrust::device_vector<float> E(NUMEROMANZANAS);  //un array para almacenar la efectividad de propaganda x manzana
   		
   		int j=0;
   		
   		if(dia%7 == 0 && dia > 120 && dia < 320){
   		    D[j]=dia;
   		    for(int i=0;i < NUMEROMANZANAS;i++){
   		    E[i]=ran2(&semilla)*0.8;//para generar distinta efectividad por manzana y por dia que se descacharra
   		    int NroDescach=round(NroTachos[i]*E[i]);//Nro de Tachos de esa manzana por la efectividad
   		    outfile5 << D[j] << "\t" << i << "\t" << E[i] << "\t" << NroTachos[i] << "\t" << NroDescach << std::endl;
  			    for(int itach=0;itach < NroDescach;itach++){
    	    			int ntach=(tachos_por_manzana[i])[itach];//identifica los tachos que se encuentran en la manzana para eliminar
  				descacharrado_kernel<<<(N+256-1)/256,256>>>(raw_estado, raw_edad, raw_tacho, raw_pupacion, raw_N_mobil,dia,ntach);
  				cudaDeviceSynchronize();
			    }//cierro for para eliminar los tachos
   		    }//cierro for para las manzanas
   		    outfile5 << "\n";
   		    j++;//incremento el contador para los elementos en D[j]
   		}//cierro if
	};


    //nacimientos
	void reproducir(int dia,int tovip)
	{
	    
		int indice=N_mobil[0];
		if(indice==0) {
			std::cout << "NO HAY MAS MOSQUITAS PARA REPRODUCIRSE" << std::endl; 	
		exit(1);
		}else{

		//nacimientos
		//std::cout << "antes kernel reproducir " << std::endl;
		//reinicializo en cero los nacidos en el paso anterior que ahora ya no son mas nacidos porque crecieron 
		thrust::fill(nacidos.begin(),nacidos.end(),0);

		// reproduce, calculando nacidos por tacho antes
		kernel_reproducir<<<(indice+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_TdV,raw_pupacion,raw_manzana,raw_N_mobil,dia,tovip,raw_nacidos);
		cudaDeviceSynchronize();

		//std::cout << "despues kernel reproducir " << std::endl; 
			// agrego todos los nacidos al final del array original, tacho a tacho
			int index=indice;
			for(int m=0;m<NUMEROTACHOS;m++){
				//calculo el nunmero de acuaticos en cada tacho
				int antiguos=thrust::count_if(
					thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(),edad.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(tacho.begin()+indice,edad.begin()+indice)),
					acuaticoeneltacho(m,TPUPAD)
				);

				 //NUEVO: cuando el TPUPAD es variable, indice=nro de bichos hasta el momento
                /*int antiguos=thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(), edad.begin(),pupacion.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(tacho.begin() + indice, edad.begin() + indice, pupacion.begin() + indice)),
                acuaticoeneltachoANA(m)
                );*/ 
				
				//std::cout << "Acuaticos en Tacho " << antiguos << " TACHO "<< m <<std::endl;
				//los nuevos vienen del kernel reproducir  
				int nuevos=nacidos[m];

				if(nuevos+antiguos>SAT){
					nuevos=SAT-antiguos;	
					
				/*NUEVO Para transferir de tacho*/				
					int manzanadeltacho = manzana_del_tacho[m];
					int cuantos=(tachos_por_manzana[manzanadeltacho]).size();

					std::cout << "tacho que se satura " << m <<" y la manzana del tacho saturado es " << manzanadeltacho;
					std::cout << ", en esa manzana hay " << cuantos << " tachos para sortear\n";

					
					int* ptr_h=(tachos_por_manzana[manzanadeltacho]).data();
					thrust::device_vector<int> tachosDeLaManzana(cuantos);
					for(int k=0;k<cuantos;k++){
						tachosDeLaManzana[k]=ptr_h[k];
						std::cout << "Manzana " << m << " tacho " << tachosDeLaManzana[k] << "\n";
					}					
					
					int* ptr_d=thrust::raw_pointer_cast(tachosDeLaManzana.data());
					
					/*Esta transformación aplica una función unaria a cada elemento de una secuencia de entrada y almacena el resultado en la posición correspondiente en una secuencia de salida. En este caso aplica la función unaria transferirdetacho() a tacho */					
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(),edad.begin())),
						thrust::make_zip_iterator(thrust::make_tuple(tacho.begin()+indice,edad.begin()+indice)),
						thrust::make_counting_iterator(0),
						tacho.begin(),
						transferirdetacho(m,TPUPAD,ptr_d,cuantos, dia)
					);
				}
				/*HASTA ACA TRANSFIERE DE TACHO*/

			    std::cout << "NUEVOS NACIDOS " << nuevos <<" TACHO "<< m <<std::endl; 

				thrust::fill(estado.begin()+index,estado.begin()+index+nuevos,ESTADOVIVO);	        //NUEVO	
				thrust::fill(edad.begin()+index,edad.begin()+index+nuevos,0);		                //NUEVO	
				thrust::fill(tacho.begin()+index,tacho.begin()+index+nuevos,m); 	                //nacen en el tacho m

				//thrust::fill(pupacion.begin()+index,pupacion.begin()+index+nuevos,15);

				// index en counting iteraror necesario para distintos randoms en cada tacho
				thrust::transform(
					thrust::make_counting_iterator(index),thrust::make_counting_iterator(index+nuevos),
					pupacion.begin()+index,uniformRanInt(15,5,dia)
				);
			
				
				thrust::transform(
					thrust::make_counting_iterator(index),thrust::make_counting_iterator(index+nuevos),
					TdV.begin()+index,uniformRanInt(27,6,dia)
				);

                // la mosquita pone huevos en el tacho m de la manzana[tacho[m]]
				thrust::fill(manzana.begin()+index,manzana.begin()+index+nuevos,manzana_del_tacho[m]);         
				index+=nuevos;		//actualizo el indice para me marque siempre en la ultima mosquita que nacio 
			}//cierro for
		
		// actualiza el indice movil hasta el ultimo bicho vivo
			if(index<MAXIMONUMEROBICHOS) {
				N_mobil[0]=index;
			}
			else{ ////satura la memoria reservada salgo del prog
				std::cout << "Demasiados Bichos!" << std::endl;
				exit(1);
			}	
				
		}	
		
	};
	
    //Recalcular -> eliminar muertos y dejar vivos
    void recalcularN(){

		auto zip_iterator=
		thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),tacho.begin(),pupacion.begin(),TdV.begin(),manzana.begin()));
		// ordenamos segun estado 0-vivo, 1-muerto
		int N=N_mobil[0];
		thrust::sort_by_key(estado.begin(), estado.begin() + N,zip_iterator);		
	
		// y ahora determinamos la posicion del primer muerto = N_mobil
		auto iter=thrust::find(estado.begin(),estado.begin() + N, ESTADOMUERTO);
		N_mobil[0]= iter-estado.begin();//me da la longitud del vector
		//std::cout << "N_mobil " << N_mobil[0] <<std::endl;
	};
	
	int vivos(int dia){

	int N=N_mobil[0];

    int poblacion = thrust::count(estado.begin(), estado.begin() + N, ESTADOVIVO);
	return poblacion;
	};

    //poblaciรณn de acuรกticos
	int acuaticos(int dia){

	int N=N_mobil[0];
	
    int ac= thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),pupacion.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin() +  N,pupacion.begin() + N)),
                poblacion_2()
            );

		return ac;
	};

    //población de adultos
	int adultos(int dia){

	int N=N_mobil[0];
	//el predicado poblacion_1()corresponde a adultos
	    int ad=thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),pupacion.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin() +  N,pupacion.begin() + N)),
                poblacion_1()
            );

		return ad;
	};	

	//envejecer población
   	void envejecer(int dia){
	int N=N_mobil[0];
        envejecer_kernel<<<(N + 256-1)/256,256>>>(raw_estado,raw_edad,raw_pupacion,raw_N_mobil,dia);
        cudaDeviceSynchronize();
	}; 
		// Estadisticas de distinto tipo sobre el array de bichos
	//void imprimir_estadisticas(int Manz,int dia){
	void imprimir_estadisticas(int dia){
		int N=N_mobil[0];
		
		std::cout << "hay " << N << " mosquitas en total" << std::endl;
        if(dia==210){
        
        outfile3 << "hay " << N << " mosquitas en total en el dia\t" << dia << std::endl;    
		//for(int i=0;i<Manz;i++){
        for(int m=0;m<NUMEROMANZANAS;m++){
            //ciento cuantas mosquitas hay en cada manzana
		    int contarMosq=thrust::count_if(manzana.begin(),manzana.begin()+N,iguala(m));
		    outfile3 << "hay\t" << contarMosq << "\tmosquitas en la manzana\t" << m << std::endl;
        }    
        outfile3 << "\n";
        
        for(int t=0;t<NUMEROTACHOS;t++){
            //cuento cuantos mosquitas hay en cada tacho
            int contarTach=thrust::count_if(tacho.begin(),tacho.begin()+N,iguala(t));
            outfile3 << "hay\t"<< contarTach << "\tmosquitas en el tacho\t" << t << "\t de la manzana\t" << manzana_del_tacho[t] << std::endl;
		}
		
		    outfile3 << "\n";
		    int nmanzanas=tachos_por_manzana.size();
		    for(int i=0;i<nmanzanas;i++){
			outfile3 << "\n\n manzana " << i << "\n tachos: ";
			int ntachos=tachos_por_manzana[i].size();
			int contTach=0;
			    for(int j=0;j<ntachos;j++){
				outfile3 << (tachos_por_manzana[i])[j] << ", ";
				contTach++;
			    }
			    
			outfile3 << "\n";
			outfile3 << " Nro de tachos en la manzana\t" << contTach;
			outfile3 << std::endl;
		    }
		
        };//CIERRO IF
		
        outfile3 << "\n";
        
        for(int m=0;m<NUMEROMANZANAS;m++){
            //ciento cuantas mosquitas hay en cada manzana
		    int contarMosq=thrust::count_if(manzana.begin(),manzana.begin()+N,iguala(m));
		    outfile4 << dia <<"\t" << m << "\t" << contarMosq << "\n";
        }
        outfile4 << "\n";
	};
};



int main(){

    outfile.open("Poblacion_total_GPU.dat");		//imprime población total de mosquitas hembras: adultas + acuáticas
    outfile1.open("Poblacion_adultos_GPU.dat");		//imprime poblacion de mosquitas hembras adultas
    outfile2.open("Poblacion_acuaticos_GPU.dat");	//imprime población de mosquitas hembras acuáticas
    outfile3.open("Condiciones_iniciales_GPU.dat");	//imprime condiciones iniciales
    outfile4.open("Dia_vs_manzana_vs_N.dat");		//imprime en columnas Dia | manzana | Nro de mosquitas en la manzana
    outfile5.open("Dia_vs_manzana_vs_efectividad.dat");	//imprime en columnas Dia | manzana | efectividad | Nro de Tachos | Nro de tachos que se descacharran
    
    outfile << "dia\t" << "N"<< std::endl;//N=Adultos + Acuáticos
    outfile1 << "dia\t" << "Ad"<< std::endl; //Adultos
    outfile2 << "dia\t" << "Ac" << std::endl;//Acuáticos
    outfile5 << "dia\t" << "manz.\t" << "efect.\t" << "NTachos\t"<< "NDescach"<<std::endl;

	//int descach=round(NUMEROTACHOS*PROP);//cantidad de tachos que vacío con la propaganda

	gpu_timer Reloj_GPU;
	Reloj_GPU.tic();
	
	/*NUMEROTACHOS=NRO DE MOSQUITAS es un valor que se ingresa en archivo parametro.h*/		
	bichos mosquitas(NUMEROTACHOS);
    
	for(int dia = 1; dia <= NDIAS; dia++){
	std::cout << "DIA" << dia << std::endl;
    	int tovip=tiempo_entre_oviposiciones(dia);
	
	std::cout << "matar" << std::endl;
	mosquitas.mortalidades(dia);//fusione muerte x vejez con mortalidades varias en un solo kernel

	std::cout << "descacharrar" << std::endl;
	//mosquitas.descacharrado(dia,descach); //DESCOMENTAR para una efectividad fija 0.6, igual para todas las manzanas, y constante durante los dias de descacharrado.
    	mosquitas.descacharrado(dia);//DESCOMENTAR para una efectividad de propaganda distinta para cada manzana y que varia durante los dias de descacharrado. 
    	//mosquitas.descacharrado(dia,E);//DESCOMENTAR para una efectividad de propaganda distinta para cada manzana y cte durante los dias de descacharrado. 

	std::cout << "reproducir" << std::endl;
	mosquitas.reproducir(dia,tovip);

	std::cout << "recalcular indice de mosquitas vivas" << std::endl;
	mosquitas.recalcularN(); 

	int vivas=mosquitas.vivos(dia);
	int adultos=mosquitas.adultos(dia);
	int acuaticos=mosquitas.acuaticos(dia);

	outfile << dia << "\t" << vivas << std::endl;
	outfile1 << dia << "\t" << adultos << std::endl;
	outfile2 << dia << "\t" << acuaticos << std::endl;

	std::cout << "envejecer poblacion" << std::endl;
	mosquitas.envejecer(dia);
	
	//mosquitas.imprimir_estadisticas(NUMEROMANZANAS,dia)
	mosquitas.imprimir_estadisticas(dia);
	std::cout << "\n";
	}
    double t=Reloj_GPU.tac()/60000; //de milisegundos -> minutos
    printf("Tiempo en GPU: %lf minutos\n",t);
//cierro archivos
outfile.close();
outfile1.close();
outfile2.close();
outfile3.close();
outfile4.close();
outfile5.close();

/*Cleanup*/
/*CURAND_CALL(curandDestroyGenerator(rng));
CUDA_CALL(cudaFree(poisson_numbers_d));
free(poisson_numbers_h);
*/
return 0;							
}// end for main

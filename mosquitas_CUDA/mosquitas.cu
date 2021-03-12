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
		 
		int tachonuevo=tachoactual;
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
			float azar=(u01_closed_closed_32_53(r[0]));
			int indicedetachoelegido=int(azar*cuantos);

			
			//tachonuevo=tachoactual; // este es solo para test

			if(indicedetachoelegido<cuantos)
			tachonuevo=ptr[indicedetachoelegido];
		}
		return tachonuevo;
	}
};

// NUEVO: otro functorcito usado para las estadisticas desagregadas

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

	// arrays medianos en device, numero_tachos elementos
	thrust::device_vector<int> nacidos; // tiene el num de tachos elementos, numero de nacidos por tacho

	// arrays medianos en host, numero_manzanas elementos
	std::vector<std::vector<int> > tachos_por_manzana; //tachos_por_manzana[i]=vector de tachos de manzana i 

	// numero de tachos elementos
	std::vector<int> manzana_del_tacho;

	
	
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
		
		//Porque /5.0 ????
		//tachos_por_manzana.resize(int(N_/5.0)+1);

		//no seria un vector de nmanzanas elementos, cada elemento a su vez es un vector de ntachos?
		tachos_por_manzana.resize(NUMEROMANZANAS); 

		manzana_del_tacho.resize(MAXIMONUMEROBICHOS);
		
		

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

		//std::cout<<"VoM\ttacho\tedad\tTdV\ttpupad\tmanzana" << std::endl;
		/*condiciones iniciales donde N sera el numero de bichos*/
		for(int i=0;i < N_;i++){
			estado[i] = ESTADOVIVO; 		    //todos vivos inicialmente
			tacho[i] = i;				        //tacho en el que se encuentra la mosquita
			edad[i] = ran2(&semilla)*7+19; 	//edad: son todas adultas al principio 
			pupacion[i] = TPUPAD-2+(ran2(&semilla)*5);//dia de pupacion (entre los 15 y 19 dias)
			TdV[i] = ran2(&semilla)*6+27 ;	//tiempo de vida de 27 a 32
						
			
			//asigno los tachos a las manzanas al azar
			manzana_del_tacho[tacho[i]]=int(ran2(&semilla)*NUMEROMANZANAS); //manzana en la que está el tacho m la asigno al azar 
			manzana[i]=manzana_del_tacho[tacho[i]]; //manzana en la que está el tacho i
			
			
			tachos_por_manzana[manzana[i]].push_back(tacho[i]); //me cuenta cuantos tachos tengo en la manzana para despues sortear
			

			//std::cout << estado[i] << "\t" << tacho[i] << "\t" << edad[i] << "\t" << TdV[i] << "\t" << pupacion[i] << "\t" << manzana[i] << "\n";
			//std::cout << "pupacion" << pupacion[i]<< std::endl;
		}

		// una verificacion del llenado de tachos_por_manzana
		int nmanzanas=tachos_por_manzana.size();
		for(int i=0;i<nmanzanas;i++){
			//std::cout << "\n\n manzana " << i << "\n tachos: ";
			int ntachos=tachos_por_manzana[i].size();
			for(int j=0;j<ntachos;j++){
				//std::cout << (tachos_por_manzana[i])[j] << ", ";
			}	
			//std::cout << std::endl;
		}
		//std::cout << "manzana del tacho 3 " << manzana_del_tacho[3] << std::endl;
		//std::cout << "manzana del tacho 1 " << manzana_del_tacho[1] << std::endl;
		//std::cout << std::endl;


		std::cout << "inicializacion lista" << std::endl;
		N_mobil[0]=N_;
	};	

	// devuelve un numero de tacho random de la manzana m
	int nuevo_tacho_misma_manzana(int m){
		int ntachos=tachos_por_manzana[m].size();
		int r=int((rand()*1.0/RAND_MAX)*ntachos);		
		int nuevo_tacho=(tachos_por_manzana[m])[r];
		return nuevo_tacho;
	}
	
	// devuelve numero de manzana sorteada entre las cuatro manzanas vecinas de la manzana m
		int sorteo_manzana_vecina(int manz){
		// numero de las manzanas vecinas de una manzana
		std::vector<int> manzanas_vecinas(5);
				
		int x=manz%LADO;
		int y=int(manz/LADO);
		manzanas_vecinas[0]=(x-1+LADO)%LADO+LADO*y; //izqu
		manzanas_vecinas[1]=(x+1+LADO)%LADO+LADO*y; //derecha
		manzanas_vecinas[2]=LADO*((y-1+LADO)%LADO)+x; //abajo
		manzanas_vecinas[3]=LADO*((y+1+LADO)%LADO)+x; //arriba
		manzanas_vecinas[4]=manz; //centro
		
		int nvecinos=6; //cuento los vecinos y el centro
		int r=int((rand()*1.0/RAND_MAX)*nvecinos); //numero aleatorio entre 0 y 6
		//si sale 5 o 6 elijo el centro (es una forma trucha de darle mas probabilidad al centro)
		int manzana_sorteada;
		if(r>3)
			{
			manzana_sorteada=manz;
			}
			else{		
			manzana_sorteada=manzanas_vecinas[r];
		}
		return manzana_sorteada;
	}


	void mortalidades(int dia){

	int N=N_mobil[0];
	//mortalidades varias y muerte por vejez
	/*Corrección 1: en el código secuencial  llamabamos tres veces al generador de nros aleatorios para eliminar huevos,pupas y adultos, mientras que en el kernel solamente sorteabamos un nro al azar y comparabamos. Para ello sortee 3 nros al azar en el kernel, pero la poblacion seguía cayendo a 0 en el dia= 316,independientemente de la SEMILLAGLOBAL que utilicemos para philox, es decir, no había mas mosquitas que reproducir. Finalmente, yendo a la función descacharrado() e imprimiendo el ntach (tacho a descacharrar),el nro de acuáticos en el tacho, el nro de mosquitas que se reproducen y el nro de mosquitas despues de eliminar muertos, me di cuenta que simplemente teníamos mala suerte. A veces salia sorteado el tacho donde estaban todos los acuáticos o antiguos, entonces todas los acuáticos pasaban a ESTADOMUERTO y al siguiente dia no habia mosquitas para reproducir.
	Solución: cambié la semilla para el generador de nros aleatorios ran2(), paso de -975 a -739 que era un nro que estaba en el código de Fabi comentado*/
	
	    matar_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_pupacion,raw_TdV,raw_N_mobil, dia);	
		cudaDeviceSynchronize();
        //SERIAL
	/*		for(int i=0;i < N;i++){
			if (estado[i] == ESTADOVIVO && edad[i] < pupacion[i]){if(ran2(&semilla) < MORACU)estado[i]=ESTADOMUERTO;}
			if (estado[i] == ESTADOVIVO && edad[i] == pupacion[i]){if(ran2(&semilla) < MORPUPAD)estado[i]=ESTADOMUERTO;}
			if (estado[i] == ESTADOVIVO && edad[i] > pupacion[i]){if(ran2(&semilla) < MORAD)estado[i]=ESTADOMUERTO;}  
			if (estado[i] == ESTADOVIVO && edad[i] >= TdV[i])estado[i]=ESTADOMUERTO;
		}  //del indice */
	};

	//descachrarrado
	void descacharrado(int dia,int descach){
	int N=N_mobil[0];
		if(dia%7 == 0 && dia > 120 && dia < 320){
  			for(int itach=0;itach < descach;itach++){
    			int ntach=ran2(&semilla)*NUMEROTACHOS;
    			//std::cout << ntach << "\n";
  				descacharrado_kernel<<<(N+256-1)/256,256>>>(raw_estado, raw_edad, raw_tacho, raw_pupacion, raw_N_mobil,dia,ntach);
  				cudaDeviceSynchronize();
			}    	
   		}
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
				//std::cout << "listo nacidos " << std::endl; 
	
				//antiguos=thrust::count_if(tacho.begin(),tacho.begin()+N,iguala(m));
				//calculo el nunmero de acuaticos en cada tacho
				int antiguos=thrust::count_if(
					thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(),edad.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(tacho.begin()+indice,edad.begin()+indice)),
					acuaticoeneltacho(m,TPUPAD)
				);
				//std::cout << "ACUATICOS " << antiguos <<" TACHO "<< m <<std::endl;
				/* //NUEVO: cuando el TPUPAD es variable, indice=nro de bichos hasta el momento
				int antiguosANA=thrust::count_if(
					thrust::make_zip_iterator(thrust::make_tuple(edad.begin(), pupacion.begin(),tacho.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(edad.begin() + indice, pupacion.begin() + indice, tacho.begin() + indice)),
					acuaticoeneltachoANA(m)
				); */
				
				//std::cout << "Acuaticos en Tacho " << antiguos << " TACHO "<< m <<std::endl;
				//los nuevos vienen del kernel reproducir  
				int nuevos=nacidos[m];
				//Ahora bien, si con los nuevos supero el maximo de huevos por tacho (SAT)
				if(nuevos+antiguos>SAT){
					nuevos=SAT-antiguos;	//ponen lo que pueden en el mismo tacho
					
				/*NUEVO Para transferir de tacho*/
				/*Muevo LOS ADULTOS a otro tacho de la misma manzana*/				
					//int manzanadeltacho = manzana_del_tacho[m]; //pone en la misma manzana
				/*Muevo LOS ADULTOS a otro tacho de la misma manzana o de una manzana vecina*/
					int estamanzana = manzana_del_tacho[m];
					int manzanadeltacho = sorteo_manzana_vecina(estamanzana); //pone en la misma manzana o en una vecina
					//cuantos tachos tengo en la manzana? 
					int cuantos=(tachos_por_manzana[manzanadeltacho]).size();

					//std::cout << "la manzana del tacho saturado es " << manzanadeltacho;
					//std::cout << ", en esa manzana hay " << cuantos << " tachos para sortear\n";

					
					int* ptr_h=(tachos_por_manzana[manzanadeltacho]).data();
					thrust::device_vector<int> tachosDeLaManzana(cuantos);
					for(int k=0;k<cuantos;k++){
						tachosDeLaManzana[k]=ptr_h[k];
					}					
					int* ptr_d=thrust::raw_pointer_cast(tachosDeLaManzana.data());
										
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(),edad.begin())),
						thrust::make_zip_iterator(thrust::make_tuple(tacho.begin()+indice,edad.begin()+indice)),
						thrust::make_counting_iterator(0),
						tacho.begin(),
						transferirdetacho(m,TPUPAD,ptr_d,cuantos, dia)
					);

					//Una vez que se llenaron los tachos de la manzana, pone en las manzanas vecinas.


				}
				/*HASTA ACA TRANSFIERE DE TACHO DENTRO DE UNA MISMA MANZANA*/

				//std::cout << "NUEVOS NACIDOS " << nuevos <<" TACHO "<< m << " MANZANA "<< manzana_del_tacho[m] << std::endl; 

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


				thrust::fill(manzana.begin()+index,manzana.begin()+index+nuevos,(int) m/5);         //NUEVO
				index+=nuevos;		//actualizo el indice para me marque siempre en la ultima mosquita que nacio 
			}
		
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
};



int main(){

	std::ofstream outfile, outfile1, outfile2;
    outfile.open("Poblacion_total_GPU.dat");
   	outfile1.open("Poblacion_adultos_GPU.dat");
    outfile2.open("Poblacion_acuaticos_GPU.dat");

	int descach=round(NUMEROTACHOS*PROP);//cantidad de tachos que vacío con la propaganda

	gpu_timer Reloj_GPU;
	Reloj_GPU.tic();

    bichos mosquitas(NINICIAL);
    //cambié el orden de las funciones y las ordené de acuerdo  al código C++ y Fortran, el orden que tenia anteriormente afectaba el resultado
    double treprod=0;
	double trecalc=0;
	double tdescacha=0;

	for(int dia = 1; dia <= NDIAS; dia++){
	std::cout << "DIA" << dia << std::endl;
    int tovip=tiempo_entre_oviposiciones(dia);
	
	//std::cout << "matar" << std::endl;
	mosquitas.mortalidades(dia);//fusione muerte x vejez con mortalidades varias en un solo kernel
	
	gpu_timer Reloj_descacharrar;
	Reloj_descacharrar.tic();
	//std::cout << "descacharrar" << std::endl;
	mosquitas.descacharrado(dia,descach); 
	tdescacha= tdescacha+Reloj_descacharrar.tac()/60000; //de milisegundos -> minutos

	gpu_timer Reloj_reproducir;
	Reloj_reproducir.tic();
	//std::cout << "reproducir" << std::endl;
	mosquitas.reproducir(dia,tovip);
	treprod= treprod+Reloj_reproducir.tac()/60000; //de milisegundos -> minutos
   

	gpu_timer Reloj_recalcular;
	Reloj_recalcular.tic();
	//std::cout << "recalcular indice de mosquitas vivas" << std::endl;
	mosquitas.recalcularN(); 
	trecalc= trecalc+Reloj_recalcular.tac()/60000; //de milisegundos -> minutos

	int vivas=mosquitas.vivos(dia);
	int adultos=mosquitas.adultos(dia);
	int acuaticos=mosquitas.acuaticos(dia);

	outfile << dia << "\t" << vivas << std::endl;
	outfile1 << dia << "\t" << adultos << std::endl;
	outfile2 << dia << "\t" << acuaticos << std::endl;

	//std::cout << "envejecer poblacion" << std::endl;
	mosquitas.envejecer(dia);
	
	}
    double t=Reloj_GPU.tac()/60000; //de milisegundos -> minutos
    printf("Tiempo Total en GPU: %lf minutos\n",t);
	printf("Tiempo en descacharrar: %lf minutos\n",tdescacha);
	printf("Tiempo en reproducir: %lf minutos\n",treprod);
	printf("Tiempo en recalcular: %lf minutos\n",trecalc);	

//cierro archivos
outfile.close();
outfile1.close();
outfile2.close();
return 0;							
}// end for main

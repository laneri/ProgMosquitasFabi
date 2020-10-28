#include <philox.h> // philox headers
#include <u01.h>    // to get uniform deviates [0,1]
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
			double azar=(u01_closed_closed_32_53(r[0]));

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
     	double azar=(u01_closed_closed_32_53(r[0]));
     	//acá incluye mortalidad de huevos,pupas y adultos con cierta probabilidad, y además muertes por vejez
		    if (edad[id] < pupacion[id]){if(azar < MORACU)estado[id]=ESTADOMUERTO;}
		    if (edad[id] == pupacion[id]){if(azar < MORPUPAD)estado[id]=ESTADOMUERTO;}
		    if (edad[id] > pupacion[id]){if(azar < MORAD)estado[id]=ESTADOMUERTO;}
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
     	float azar=(u01_closed_closed_32_53(r[0]));
		return int(medio+ancho*azar);
	}
};	
//thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(N),output.begin(),
// uniformRanInt(27,5,dia));

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

//MODIFIQUÉ EL ORDEN DE LAS TUPLA
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

struct bichos{


	thrust::device_vector<int> estado;  // Vivo o Muerto 0/1 
	thrust::device_vector<int> edad;    //edad que tiene la mosquita   
	thrust::device_vector<int> tacho;  // numero de tacho en que se encuentra cada mosquita valores=0 a NUMEROTACHOS   
	thrust::device_vector<int> TdV;     //tiempo de vida de cada mosquita
	thrust::device_vector<int> pupacion; //dia de paso de pupa a adulta de cada mosquita
	thrust::device_vector<int> manzana; //numero de manzana de cada mosquita
	thrust::device_vector<int> nacidos; // tiene el num de tachos elementos, numero de nacidos por tacho

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

	/*condiciones iniciales*/
 	for(int i=0;i < N_;i++){
		estado[i]=ESTADOVIVO; 		        //todos vivos inicialmente
		tacho[i]=i;			        	    //tacho en el que se encuentra la mosquita
		edad[i]=ran2(&semilla)*7+19; 		//edad 
		pupacion[i]=TPUPAD - 2 + (ran2(&semilla)*5);    //dia de pupacion (entre los 15 y 19 dias)
		TdV[i]=ran2(&semilla)*6+27 ;	 ón//tiempo de vida de 27 a 32
		manzana[i]=(int) (i/5);
		std::cout << estado[i] << "\t" << tacho[i] << "\t" << edad[i] << "\t" << TdV[i] << "\t" << pupacion[i] << "\n";
	}

	N_mobil[0]=N_;
	};	

	
	void mortalidades(int dia){

	int N=N_mobil[0];
	//mortalidades varias y muerte por vejez
	    matar_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_pupacion,raw_TdV,raw_N_mobil, dia);	
		cudaDeviceSynchronize();
	};

	//descachrarrado
	void descacharrado(int dia,int descach){
	int N=N_mobil[0];
	    // SERIAL
		/*	if(dia%7 == 0 && dia > 120 && dia < 320){
  			for(int itach=0;itach < descach;itach++){
    		    int ntach=ran2(&semilla)*NUMEROTACHOS;
				for(int i=0;i < N;i++){
  					if (estado[i] == ESTADOVIVO && edad[i] < pupacion[i] && tacho[i] == ntach)estado[i]=ESTADOMUERTO;
				}
			}    	
   		}*/
   		//CUDA
		if(dia%7 == 0 && dia > 120 && dia < 320){
  			for(int itach=0;itach < descach;itach++){
  			    /*ESTO ES RARO: al parecer no le gusta ntach=ran2(&semilla)*NUMEROTACHOS el cual sortea nros random entre [0,NUMEROTACHOS)(1), asi que lo que hice fue tirar nros random entre [0,NUMEROTACHOS] y si ntach=NUMEROTACHOS=5 entonces vuelvo a sortear otro nro random que caiga  en el intervalo (1), haciendo esto la poblacion de mosquitas no me cae a 0, probe con varias semillaglobal y el resultado no es tan sensible a la variación de la semilla, algo que si observabamos antes. Ahora, esto está bien ?*/
    			int ntach=ran2(&semilla)*NUMEROTACHOS + 1;//ntach=(0,1,2,3,4) + 1 
    			if(ntach==NUMEROTACHOS)ntach=ran2(&semilla)*NUMEROTACHOS;// ntach= (0,1,2,3,4)
  				descacharrado_kernel<<<(N+256-1)/256,256>>>(raw_estado, raw_edad, raw_tacho, raw_pupacion, raw_N_mobil,dia,ntach);
  				cudaDeviceSynchronize();
			}    	
   		}
	};


    //nacimientos
	void reproducir(int dia,int tovip){
	    
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
            /*int antiguos=thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(),edad.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(tacho.begin()+indice,edad.begin()+indice)),
                acuaticoeneltacho(m,TPUPAD)
			);*/
			
			//Comenté el anterior porque el código actualizado que me pasó fabi considera el TPUPAD variable, indice=nro de bichos hasta el momento
            int antiguos=thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(), edad.begin(),pupacion.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(tacho.begin() + index, edad.begin() + index, pupacion.begin() + index)),
                acuaticoeneltachoANA(m)
            ); 
            
			//std::cout << "Acuaticos en Tacho " << antiguos << " TACHO "<< m <<std::endl;
          	//los nuevos vienen del kernel reproducir  
            int nuevos=nacidos[m];
            //me parece que si sucede esta condición  la mosquita tiene que irse a otro tachito, se podria hilar la transferencia de tacho aquí
			if(nuevos+antiguos>SAT){
				nuevos=SAT-antiguos;	
			};

			//std::cout << "NUEVOS NACIDOS " << nuevos <<" TACHO "<< m <<std::endl; 

            thrust::fill(estado.begin()+index,estado.begin()+index+nuevos,ESTADOVIVO);	        //estado del nacido	
			thrust::fill(edad.begin()+index,edad.begin()+index+nuevos,1);		                //edad del nacido: dia 1 y no 0 	
			thrust::fill(tacho.begin()+index,tacho.begin()+index+nuevos,m); 	                //tacho en el que nace

			// index en counting iteraror necesario para distintos randoms en cada tacho
			thrust::transform(
				thrust::make_counting_iterator(index),thrust::make_counting_iterator(index+nuevos),//edad en que se vuelve pupa
				pupacion.begin()+index,uniformRanInt(15,5,dia)
			);
		
			thrust::transform(
				thrust::make_counting_iterator(index),thrust::make_counting_iterator(index+nuevos),//tiempo de vida
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
	};
	
	int vivos(int dia){

	int N=N_mobil[0];

    int poblacion = thrust::count(estado.begin(), estado.begin() + N, ESTADOVIVO);
	return poblacion;
	};

    //poblaciรณn de acuรกticos
	int acuaticos(int dia){

	int N=N_mobil[0];
	//el predicado poblacion_2()corresponde a los acuaticos
    int ac=thrust::count_if(
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
    
	for(int dia = 1; dia <= NDIAS; dia++){
	std::cout << "DIA" << dia << std::endl;
    	int tovip=tiempo_entre_oviposiciones(dia);
	
	std::cout << "matar" << std::endl;
	mosquitas.mortalidades(dia);//fusione muerte x vejez con mortalidades varias en un solo kernel

	std::cout << "descacharrar" << std::endl;
	mosquitas.descacharrado(dia,descach); 

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
	
	}
    double t=Reloj_GPU.tac()/60000; //de milisegundos -> minutos
    printf("Tiempo en GPU: %lf minutos\n",t);
//cierro archivos
outfile.close();
outfile1.close();
outfile2.close();
return 0;							
}// end for main

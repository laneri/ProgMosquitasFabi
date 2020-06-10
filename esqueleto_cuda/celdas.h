// thrust headers
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>

// cpp headers
#include <cassert>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "parametros.h"


typedef thrust::device_ptr<int> device_ptr_int;
typedef thrust::device_ptr<float> device_ptr_float;
typedef float REAL;

using namespace std;


// force functor aqui está como paso del tiempo t al t+uno
struct AvanzoTiempo_functor_nuevo{
    
    int tdia;
        

    AvanzoTiempo_functor_nuevo(int _tdia)
    :tdia(_tdia){};

 
    template<typename Tuple> //importante poner esto
    __device__ 
    void operator()(Tuple estados)
    {
      		
                device_ptr_float MpicanCoh = thrust::get<0>(estados); //NUMERO de mosquitos que pican por cada cohorte
                int index=thrust::get<1>(estados); //indice de celda
                     
                                       
                int NCohortes=cohLen;//este numero se define en parametos.h

                //esta transformacion se necesita para calculos posteriores
                float *MpicanCohptr=thrust::raw_pointer_cast(MpicanCoh);
                
                float POMosquitostot=0;
                
                for(int coh=0;coh<NCohortes;coh++){
                  MpicanCohptr[coh]= 25;
                    
       //          printf("MpicanCohptr=%f , MpicanCohptr[coh]);     
                                                                               
                  POMosquitostot+=MpicanCohptr[coh]; //poblacion total de mosquitos en el tiempo actual

                }//fin del loop mosquitos cohortes
            }
             
};   //hasta aca el functor


class celdas{
	private:
                int ncel;
                
               // para describir las cohortes de mosquitos por ej...
                thrust::device_vector<device_ptr_float> Mm; //mosquitos que pican por cohorte un dado dia
                
                  
                
                vector<float> MosquitosAdultos;
                
            
	         
       
                
                
	public:

                
                    
               
            int getNC(){return ncel;}; //devuelve el numero de celdas 

	           
                // Definicion del Objeto celdas
            celdas(int nceldas, int mcohortes){
                //ncel numero de manzanas o celdas
                ncel=nceldas; 
                int ncoh=mcohortes; //numero de cohortes de mosquitos en cada manzana

          
                
                //Mosquitos
                std::cout << "construyendo multidim array..." << std::endl;
                        
                                           
                        
                        for(int i=0;i<ncel;i++){ 
                            std::cout << "alocando memoria" << i << std::endl;
                            Mm[i]=thrust::device_malloc<float>(ncoh);
                           
                           
                        }
                               
			
			InicializarM(); //Inicializo poblacion de mosquitos
        
		
 		}

		~celdas(){
                
			         
                        device_ptr_float x;  
                        device_ptr_int y;
                        std::cout << "destruyendo multidim array..." << std::endl;
                        for(int i=0;i<NC;i++)
                        { 
                                std::cout << "liberando memoria" << i << std::endl;
                                x=Mm[i];
                                thrust::device_free(x);
                                
                                                            
                                
                        }
                                             
                }

               
               

          //Inicializo poblacionde de mosquitos y sus cohortes en cada manzana      
                void InicializarM(){
                    //cout<< "suma=" << thrust::reduce(inpvec.begin(),inpvec.end()) << std::endl;
                    for(int celda=0;celda<NC;celda++){
                    device_ptr_float Mmaux=device_ptr_float(Mm[celda]);
                    //Todos los mosquitos son inicialmente susceptibles
                    for(int cohorte=0;cohorte<cohLen;cohorte++){
                    	Mmaux[cohorte]=10.0;
                    }
                
                }
		// TODO: paralelizar este rellenado		
		// TODO: chequear que los accesos son legales, i.e. en [0,N*M-1]		
	}
                
                template <typename T>
                void PrintVectorCohortes(thrust::device_vector<T> &cohvec, int dia)
                {
                    for(int i=0;i<NC;i++){
                       T xptr=cohvec[i];
                       cout<< "celda " << i << " dia " << dia << " -> ";
                       cout<< float(thrust::reduce(xptr,xptr+cohLen)) << std::endl;
                    }
                }
                


    
		void dinamica(int trun, int Nciudades)
		{
			using namespace thrust::placeholders;
                        
                        
 			for(int dia=0;dia<trun;dia++)
			{ 
                          
                            //imprimo en pantalla
                             //   PrintVector("Temp",Tgpu); //Temperatura por celda cada dia
 
                          //      PrintVectorCohortes(Mm,dia);//Mosquitos que pican por celda cada dia
                                                               
          
                           thrust::counting_iterator<int> indices_begin=thrust::make_counting_iterator(0);
                           thrust::counting_iterator<int> indices_end=indices_begin+NC;
                                 
      //#ifdef DEBUG
//                            for(int c=0;c<4;c++){ 
//                             std::cout << " celda " << c << std::endl;
//                             device_ptr_float it=device_ptr_float(Mm[c]); //celda 0
//                             for(int i=0;i<4;i++) std::cout << "Mm" << " --------> " << it[i] << std::endl; //cortes de celda 0  
//                            }
//                            std::cout << "dia "<<dia << std::endl;
//  //                          exit(1);
//#endif                      
 
                           
                           thrust::for_each(
                                   thrust::make_zip_iterator(thrust::make_tuple(Mm.begin(),indices_begin)),	
                                   thrust::make_zip_iterator(thrust::make_tuple(Mm.end(),indices_end)),	
                                   AvanzoTiempo_functor_nuevo(dia)                            
                                );	
                               
                                                  
                                  
                          MosquitosAdultos.push_back(total_mosquitos_adultos());      
                        }
                      
                      
                       std::ofstream fmostot("MosquitosAdultosTotales"); //nombre del archivo
                       print_SerieTemporalGenerica(fmostot,MosquitosAdultos); //serie a imprimir
                        
 		}

 
        float total_mosquitos_adultos()
 		{
            thrust::host_vector<float> MosqAdultos(cohLen);
            int i=0;
            for(int c=0;c<NC;c++){ 
            //std::cout << " celda " << c << std::endl;
            device_ptr_float it=device_ptr_float(Mm[c]); 
            MosqAdultos[i]=thrust::reduce(it,it+cohLen,float(0.0));
            i=i+1;
            //for(int j=0;j<4;j++) std::cout << "Mm" << " --------> " << it[j] << std::endl; //cortes de celda 0  
            }
 			return thrust::reduce(MosqAdultos.begin(),MosqAdultos.end(),float(0.0));

        }
 		// simple routine to print contents of a thrust vector
               template <typename Vector>
               void PrintVector(const std::string& name, const Vector& v)
               {
                typedef typename Vector::value_type T;
                //std::cout << " " << std::setw(20) << name << " ";
                thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
                std::cout << std::endl;
               }
               
               // simple routine to print contents of a thrust vector in a file
               template <typename Vector>
               void PrintVectorFile(const std::string& name, const Vector& v, std::ofstream& fout)
               {
                typedef typename Vector::value_type T;
//  fout << " " << std::setw(20) << name << " ";
                thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(fout, " "));
                fout << std::endl;
               }
                //template <typename Vector>
              
                void print_SerieTemporal(std::ofstream &fout){
                    
			for(int i=0;i<DiasSimul;i++){
//                       fout << InfectadosTotales[i] << "\n";
                        }
                    fout << "\n" << std::endl;
		}
		
		void print_SerieTemporalGenerica(std::ofstream &fout, std::vector<float> &serie){
                    
			for(int i=0;i<DiasSimul;i++){
                       fout << serie[i] << "\n";
                        }
                    fout << "\n" << std::endl;
		}

		
 };

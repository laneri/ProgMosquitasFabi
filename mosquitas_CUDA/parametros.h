/*
Macros utiles para CPU y GPU
Relaciones a tener en cuenta para modificar parรกmetros
- NUMEROTACHOS=NINICIAL
*/

#define NINICIAL 		    20	        //número inicial de mosquitos N=125 para 5 mosquitas x manzana.
#define NUMEROTACHOS		20          //máximo número de tachos Ntachos=Nmosquitas
#define MAXIMONUMEROBICHOS	80000000    //número maximo de huevos
#define LADO           	    2           //Lado de la grilla L=5
#define NUMEROMANZANAS		LADO*LADO   //número de manzanas que sea L*L=5*5=25

//#define NUMERODEHUEVOS		10 		//número de huevos por oviposicion	


#define ESTADOMUERTO		1
#define ESTADOVIVO	    	0
#define NDIAS		    	400	        //nro de dias en el año
#define NREALIZACIONES		1	        //nro de realizaciones para promediar poblaciones con diferente semillas

#define MORHUE 		        0.01   	    //mortalidad de huevos
#define MORLAR 		        0.01       //mortalidad de larvas 
#define MORPUP 		        0.01   	    //mortalidad de pupas
#define MORAD 	    		0.01 	    //mortalidad diaria adultas
#define MORACU 	        	0.03	    //morhue+morlar+morpup;
#define MORPUPAD 	    	0.17	    //pupas que no se vuelven adultas
#define TPUPAD	    		17  	    //pupas se vuelven adultas a los 17 dias en invierno****
#define TOVIP1 	        	2	        //tiempo entre dos oviposiciones (T=30)
#define TOVIP2a     		3  	        //tiempo entre dos oviposiciones (T=25)
#define TOVIP2b		        4   	    //tiempo entre dos oviposiciones (T=25)
#define TOVIP3 		        30	        //tiempo ente dos oviposiciones (T=18)
#define SAT 		       	800         //saturación de huevos por tacho
#define PROP 		    	0.6         //efectividad de la propaganda

#define TPUPAD  	    	17  	    //pupas se vuelven adultas a los 17 dias en invierno****

//#define SEMILLAGLOBAL	   	22399989
#define SEMILLAGLOBAL	   	123456789
//#define SEMILLAGLOBAL	   	975

//long semilla = -739;  			//semilla para el generador de numeros aleatorios ran2()

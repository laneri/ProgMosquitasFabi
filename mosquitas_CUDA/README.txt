El código calcula la población de mosquitas en 20 manzanas, considerando 1 tacho por manzana (es decir, 20 tachos) y una mosquita por tacho. 

en el archivo bichos.h:
- Cada mosquita i en el programa tiene definido:
1) su estado               -> 0(vivo) o 1(muerto)
2) su edad                 -> entre 12 y 16 dias 
3) tacho en el que vive    -> i
4) tiempo de vida		   -> entre 28 y 30 dias
5) manzana 		         -> 5 tachos por manzana--------------------------------------> nuevo

Considera
- Mortalidades varias (mata las mosquitas con cierta probabilidad)
- Nacimientos por saturación ()
- Descacharrado
//- Mortalidad por vejez //---------------------------------------------------------> nuevo:esto está comentado en el código porque cuando lo activo se muere todo 
- envejece la población de mosquitas un día
- elimina los muertos de los arrays

en el archivo bichos.cu:

-La función test1() calcula por dia:
1) el número total de mosquitas después de reproducirse (agregue saturación, 800 huevos por tacho)
2) el número de mosquitas que se mueren con alguna probabilidad ( prob. de morirse de huevo-> larva, de larva-> pupa, de pupa ->adulta)
3) el número de mosquitas que se mueren por viejas
4) el número de mosquitas después de descacharrar un tacho
5) envejecimiento de 1 día para la edad de las mosquitas
6) se remueven las mosquitas muertas
7) estadística: cuantas mosquitas hay por tacho,cuantas mosquitas hay por determinada edad y cuando tachos hay por manzana. 

-La función test2() calcula la población de mosquitas de todos los tachos en un año (400 días)


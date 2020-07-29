El código calcula la población de mosquitas en 20 manzanas. 

Cada mosquita i en el programa tiene definido:
su estado               -> 0(vivo) o 1(muerto)
su edad                 -> entre 12 y 16 dias 
tacho en el que vive    -> i
tiempo de vida		-> entre 28 y 30 dias
manzana 		-> 1 tacho por manzana

La función test() calcula por dia:
1) el número total de mosquitas después de reproducirse (agregue saturación, 800 huevos por tacho)
2) el número de mosquitas que se mueren con alguna probabilidad ( prob. de morirse de huevo-> larva, de larva-> pupa, de pupa ->adulta)
3) el número de mosquitas que se mueren por viejas
4) el número de mosquitas después de descacharrar un tacho
5) envejecimiento de 1 día para la edad de las mosquitas
6) se remueven las mosquitas muertas
7) estadística: cuantas mosquitas hay por tacho,cuantas mosquitas hay por determinada edad y cuando tachos hay por manzana. 

La función testGPU() calcula la población de mosquitas de todos los tachos en un año (400 días)


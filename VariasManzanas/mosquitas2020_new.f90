! Programa de mosquitAs en una sola manzana
!---------------------------------------------------------------------
!En la naturaleza, cada oviposicion son aprox. 64 huevos y la mitad son hembras
!Solo modelaremos la dinamica de las hembras ( y despues multiplicamos por 2)
!En esta versión considero un numero fijo de 32 huevos-hembra x oviposicion.
!Supongo mortalidad diaria de huevos, pupas, larvas y adultas indep de Temperatura
!Ademas, el 83% de las larvas pasan a adultas jovenes, es decir,
!mueren con probab 0.17 al pasar del agua al aire
!Del paper de Otero, el nro. de oviposiciones por hembra depende de la temperatura: 
!1 ovip. a 20 grados, 4 o 5 a 25 grados y 6 a 30 grados.
!En esta version suponemos que ponen huevos cada "tovip" dias (ver mas abajo)
!Adulta muere entre los 27 y los 32 dias (distrib uniforme)*****
!Pupa se vuelve adulta a los 17 (12+5) dias (fijo)*****
!Entonces:
!Cada agente es mosquita(i), vector de 4 componentes
!mosquita(i,1)=1 o 0 (si está viva o muerta)
!mosquita(i,2)=ntachito (en que tachito vive=cohorte)
!mosquita(i,3)=edad (avanza de a 1 dia)
!mosquita(i,4)=dias que va a vivir (distribuido entre 27 y32, tomado al azar)****
!Además tengo SATURACION (sat): numero maximo de huevos permitidos por tacho 
!PROPAGANDA=DESCACHARRADO (prop): efectividad de la campaña publicitaria
!AGREGO "las 4 estaciones": discretizo la curva de temperaturas que tenemos de BsAs 
!es para 400 dias, empezando en enero 2015:*******
!dias<80 o dias>320, T=30°C, 6 oviposiciones
!80<dias<160 o 240<dias<320, T=25°C, entre 4 y 5 oviposiciones 
! 160<dias<240, T=18°C, 1 oviposicion
!------------------------------------------------------------------
!modificacion de esta version 2mosquitas.f90 (13/6/2020)
!-------------------------------------------------------------------
!el día 1 es 1/7 en este programa (<T>=18°C durante los 1ros 80 dias)
!el descacharrado se hace solo en dic, enero y febrero (día 150 al 240)
!las hembras no oviponen en invierno
!siguiendo paper de Bergero, oviposicion: 20 (10 porque solo pongo hembras)en vez de 64 
!edad de muerte de adultas: distribucion mas angosta (entre 28 y 30 dias en total, o sea, entre 11 y 13 de adultas)
!---------------------------------------------------------------------

 parameter (nmax=80000000)   
 implicit double precision (a-h,o-z)
 integer iseed1,iseed2,ad,ac,ad1,ad2,ad3,ad4,ad5,adn1,adn2,adn3,adn4,adn5
 integer mosquita(nmax,4),tach(nmax)   !mosquitas y sus propiedades
 INTEGER tiempo,ttotal,tpupad,tovip,tovip1,tovip2b,tovip2a,tovip3,sat,descach,mor  
 real ran2  
 EXTERNAL ran2

! open (unit=1,file="2mosq4.dat") 
! open (unit=2,file="2newmosq4.dat") 
 open (unit=3,file="mosquitas_totales.dat") 

! parámetros:

!IDUM = -147 !semilla del generador de numeros aleatorios (c/ semilla es para una manzana)
! IDUM = -581 !semilla del generador de numeros aleatorios
! IDUM = -739 !semilla del generador de numeros aleatorios
IDUM = -975  !semilla del generador de numeros aleatorios


 inhem=5     !nro de hembras en la Condic Inic (c/hembra pone en un tacho distinto)
 ntachito=5  !nro de tachitos 	  
 !iovip=10   !nro de huevos hembras en cada ovip (10 huevos,paper Bergero), lo pongo entre 7 y 10 variable
 ttotal=400  !numero total de dias de la simulacion

 morhue=0.01d0   !mortalidad diaria huevos
 morlar=0.01d0   !mortalidad diaria larvas
 morpup=0.01d0   !mortalidad diaria pupas

 moracu=morhue+morlar+morpup  !mortalidad total de las acuaticas

 morad= 0.01d0   !mortalidad diaria adultas 
 morpupad=0.17d0   !el 17 por ciento de las pupas no se vuelven adultas
 
 tpupad1=9  !pupas se vuelven adultas a los 9 días en verano (desde oviposicion)****
 tpupad2=13  !pupas se vuelven adultas a los 13 dias en otoño y primavera****
 tpupad3=17  !pupas se vuelven adultas a los 17 dias en invierno****
 
 tovip1=2     !tiempo entre dos oviposiciones (T=30: 6 ovip= 12/2)
 tovip2a=3    !tiempo entre dos oviposiciones (T=27: 4 ovip= 12/3)  
 tovip2b=4    !tiempo entre dos oviposiciones (T=23: 3 ovip= 12/4)   
 tovip3=29    !tiempo entre dos oviposiciones (T=18: 0 ovip)**** 

 sat=800      !numero maximo de huevos en cada tacho
 prop=0.6    !efectividad de la propaganda
 descach=nint(ntachito*prop)  !cantidad de tachos que vacío con la propaganda
   
!-------------------------------------------
! inicializo la poblacion mosquítica
!-------------------------------------------

 mosquita=0
 indice=0      !ocupacion en la matriz (nro de bichos, va creciendo con el tiempo)

   do kk=1,inhem   
   mosquita(kk,1)=1     !vive
   idia=ran2(IDUM)*5+12 !tiene entre 12 y 17 días de vida
   mosquita(kk,2)=kk    !está en el tachito kk
   mosquita(kk,3)=idia  !tiene "idia" dias de vida
   mor=ran2(IDUM)*3+28 
   mosquita(kk,4)=mor   !dia de su muerte (entre los 28 y 30 dias)
   indice=indice+1
   print*,kk,mosquita(kk,1),mosquita(kk,2),mosquita(kk,3),mosquita(kk,4)
   end do
  write(*,*)
  !print*,indice
  write(*,*)
!-------------------------------------------------------------------	
 do tiempo=1,ttotal !empieza una realizacion del programa
  !print*,indice
!-----------------------------------------------------------------------------------------------------
! defino la temperatura segun la estacion y el tiempo entre oviposiciones y de maduración de acuaticos
!-----------------------------------------------------------------------------------------------------
  if(tiempo.lt.80 .or. tiempo.gt.320) then  !<T>=18
  tpupad=tpupad3   !******
  tovip=tovip3  
  end if
  if(tiempo.gt.80 .and. tiempo.lt.140) then  !<T>=23
  tpupad=tpupad2
  tovip=tovip2b
  end if
  if(tiempo.ge.140 .and. tiempo.le.260) then   !<T>=30
  tpupad=tpupad1
  tovip=tovip1
  end if
  if(tiempo.ge.260 .and. tiempo.le.320) then !<T>=27
  tpupad=tpupad2   
  tovip=tovip2a 
  end if
!---------------------------------------------------------------------
! cuento cuantos huevos hay en cada tacho
!---------------------------------------------------------------------- 
    tach=0
  
 do i=1,indice
   if(mosquita(i,3).lt.tpupad.and.mosquita(i,1).eq.1) then !cuento solo acuaticos, es decir edad<tpupad
    j=mosquita(i,2) 
    tach(j)=tach(j)+1
   end if
 end do 

!--------------------------------------------------------------------------
! MORTALIDADADES VARIAS (morirse antes de ser vieja): mato acuaticos con prob moracu,
! a las que estan por volverse adultas picadoras con prob morpupad
! y a las adultas con prob morad (no es la muerte por vejez, ojo)
!--------------------------------------------------------------------------
 do i =1,indice
	if (mosquita(i,1).eq.1.and.mosquita(i,3).lt.tpupad) then
  if(ran2(IDUM).lt. 0.03d0)mosquita(i,1)=0  
   end if
	if (mosquita(i,1).eq.1.and.mosquita(i,3).eq.tpupad) then
  	 if(ran2(IDUM).lt. 0.17d0)mosquita(i,1)=0  
   end if
	if (mosquita(i,1).eq.1.and.mosquita(i,3).gt.tpupad) then
  	 if(ran2(IDUM).lt. 0.01d0)mosquita(i,1)=0  
   end if
  write(4,*)indice,mosquita(i,1)
	   end do  !del indice

! -------------------------------------------------------------------	    
! NACIMIENTOS CON SATURACION
!-------------------------------------------------------------------------- 
 do i =1,indice   !loop de toda la poblacion

if (tach(mosquita(i,2)).lt.800) then
  if(mosquita(i,1).eq.1 .and. mosquita(i,3).gt.tpupad)then   !si estan vivas y están maduras
   if(mod(mosquita(i,3),tovip).eq.0) then   !cada tovip pone una camada de iovip huevos
     iovip=ran2(IDUM)*4+7  !pone entre 7 y 11 huevos  *****
    do ik=1,iovip 
      indice=indice+1  !indice de la nueva bicha
      mosquita(indice,1)=1 !está viva
      mosquita(indice,3)=1 		!tiene 1 dia   
      mosquita(indice,2)=mosquita(i,2)    !nace en el tacho de su madre 
      mor=ran2(IDUM)*3+28 
      mosquita(indice,4)=mor    !dias que va a vivir ******
      tach(mosquita(indice,2))=tach(mosquita(indice,2))+1 !sumo en 1 el contador de acuaticos
    end do
  end if    !del periodo entre oviposiciones y saturacion
end if      !de vivas y maduras
   end if   !de la saturac
end do  ! de poblacion total (indice)
 
!--------------------------------------------------------------------------
!MUERTE POR VEJEZ: la mato cuando su edad alcanza el tiempo de vida asignado
!--------------------------------------------------------------------------
do i =1,indice
  if (mosquita(i,1).eq.1 .and. mosquita(i,3).ge.mosquita(i,4))mosquita(i,1)=0
end do 
!--------------------------------------------------------------------			  		 
!DESCACHARRADO: elimino los acuaticos de algunos tachos, 
!una vez por semana a partir de diciembre
!-------------------------------------------------------------------- 
   if(mod(tiempo,7).eq.0 .and. tiempo.gt.150 .and. tiempo.lt.240) then
    do itach=1,descach!tachos que voy a sacar
      ntach=ran2(IDUM)*ntachito + 1 !elijo uno de los tachitos al azar*****REVISAR****
      !print*,ntach
       do i =1,indice    !y lo vacío
    if (mosquita(i,1).eq.1.and.mosquita(i,3).lt.tpupad.and.mosquita(i,2).eq.ntach)mosquita(i,1)=0
       end do
    end do    
   end if
!--------------------------------------------------------------------			  		 
! cuento poblacion adulta y acuatica
!--------------------------------------------------------------------			  
  ad=0  !adultos total
  ac=0  !acuaticos vivos total 
!  ad1=0 !adultos vivos por tacho
!  ad2=0
!  ad3=0
!  ad4=0
!  ad5=0
!  adn1=0 !adultos nuevos por tacho
!  adn2=0
!  adn3=0
!  adn4=0
!  adn5=0
  
    do i =1,indice
if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.tpupad)ad=ad+1 !adultos tot
if (mosquita(i,1).eq.1.and.mosquita(i,3).lt.tpupad)ac=ac+1 !acuaticos tot

!if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.tpupad.and.mosquita(i,2).eq.1)ad1=ad1+1
!if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.tpupad.and.mosquita(i,2).eq.2)ad2=ad2+1
!if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.tpupad.and.mosquita(i,2).eq.3)ad3=ad3+1
!if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.tpupad.and.mosquita(i,2).eq.4)ad4=ad4+1
!if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.tpupad.and.mosquita(i,2).eq.5)ad5=ad5+1

!if (mosquita(i,1).eq.1.and.mosquita(i,3).eq.tpupad.and.mosquita(i,2).eq.1)adn1=adn1+1
!if (mosquita(i,1).eq.1.and.mosquita(i,3).eq.tpupad.and.mosquita(i,2).eq.2)adn2=adn2+1
!if (mosquita(i,1).eq.1.and.mosquita(i,3).eq.tpupad.and.mosquita(i,2).eq.3)adn3=adn3+1
!if (mosquita(i,1).eq.1.and.mosquita(i,3).eq.tpupad.and.mosquita(i,2).eq.4)adn4=adn4+1
!if (mosquita(i,1).eq.1.and.mosquita(i,3).eq.tpupad.and.mosquita(i,2).eq.5)adn5=adn5+1


    end do 

!--------------------------------------------------------------------	   
!envejezco a toda la población un día (aumento en 1)
!--------------------------------------------------------------------
   do i =1,indice
	  if (mosquita(i,1).eq.1)mosquita(i,3)=mosquita(i,3)+1
	end do  

!--------------------------------------------------------------------
!guardo los datos
!--------------------------------------------------------------------
	   write(3,*)tiempo,ad,ac,ad+ac		
!	   write(1,*)tiempo,ad1,ad2,ad3,ad4,ad5	
!	   write(2,*)tiempo,adn1,adn2,adn3,adn4,adn5	
	   !write(*,*)tiempo,ad

!-------------------------------------------------------------------	   
!-------------------------------------------------------------------	   
 end do  ! del tiempo
!-------------------------------------------------------------------	   
   close(3)
!   close(2)
!   close(1)

 	    end

!----------------------------------------------------------------------------
 FUNCTION ran2(IDUM)
 INTEGER idum,IM1,IM2,IMM1,IA1,IA2,IQ1,IQ2,IR1,IR2,NTAB,NDIV
 REAL ran2,AM,EPS,RNMX
 PARAMETER (IM1=2147483563,IM2=2147483399,AM=1./IM1,IMM1=IM1-1,IA1=40014)
 PARAMETER (IA2=40692,IQ1=53668,IQ2=52774,IR1=12211,IR2=3791,NTAB=32)
 PARAMETER(NDIV=1+IMM1/NTAB,EPS=1.2e-7,RNMX=1.-EPS)
 INTEGER idum2,j,k,iv(NTAB),iy
 SAVE iv,iy,idum2
 DATA idum2/123456789/, iv/NTAB*0/, iy/0/
 if (idum.le.0) then
 idum=max(-idum,1)
 idum2=idum
 do 11 j=NTAB+8,1,-1
 k=idum/IQ1
 idum=IA1*(idum-k*IQ1)-k*IR1
 if (idum.lt.0) idum=idum+IM1
 if (j.le.NTAB) iv(j)=idum
11      continue
 iy=iv(1)
 endif
 k=idum/IQ1
 idum=IA1*(idum-k*IQ1)-k*IR1
 if (idum.lt.0) idum=idum+IM1
 k=idum2/IQ2
 idum2=IA2*(idum2-k*IQ2)-k*IR2
 if (idum2.lt.0) idum2=idum2+IM2
 j=1+iy/NDIV
 iy=iv(j)-idum2
 iv(j)=idum
 if(iy.lt.1)iy=iy+IMM1
 ran2=min(AM*iy,RNMX)
 return
 END

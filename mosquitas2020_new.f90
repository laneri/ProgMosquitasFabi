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
!Adulta muere entre los 27 y los 32 dias (distrib uniforme)
!Pupa se vuelve adulta a los 17 (12+5) dias (fijo)
!Entonces:
!Cada agente es mosquita(i), vector de 4 componentes
!mosquita(i,1)=1 o 0 (si está viva o muerta)
!mosquita(i,2)=ntachito (en que tachito vive=cohorte)
!mosquita(i,3)=edad (avanza de a 1 dia)
!mosquita(i,4)=dias que va a vivir (distribuido entre 27 y32, tomado al azar)
!Además tengo SATURACION (sat): numero maximo de huevos permitidos por tacho 
!PROPAGANDA=DESCACHARRADO (prop): efectividad de la campaña publicitaria
!AGREGO "las 4 estaciones": discretizo la curva de temperaturas que tenemos de BsAs 
!es para 400 dias, empezando en enero 2015:
!dias<80 o dias>320, T=30°C, 6 oviposiciones
!80<dias<160 o 240<dias<320, T=25°C, entre 4 y 5 oviposiciones 
! 160<dias<240, T=18°C, 1 oviposicion
!el día 1 es comienzo del verano en este programa (<T>=30°C durante los 1ros 80 dias)
!---------------------------------------------------------------------

! parameter (nmax=80000000)   
 parameter (nmax=800000)   
 implicit double precision (a-h,o-z)
 integer iseed1,iseed2,ad,ac,ad1,ad2,ad3,ad4,ad5,adn1,adn2,adn3,adn4,adn5
 integer mosquita(nmax,4),tach(nmax)   !mosquitas y sus propiedades
 INTEGER tiempo,ttotal,tpupad,tovip,tovip1,tovip2b,tovip2a,tovip3,sat,descach  
 !real RAN2  
 !EXTERNAL RAN2
 real ran2  
 EXTERNAL ran2


 open (unit=1,file="adultos_vivos_por_tacho_F.dat") 
 open (unit=2,file="adultos_nuevo_por_tacho_F.dat") 
 open (unit=3,file="acuaticos_vivos_total.dat") 

! parámetros:

 inhem=5     !numero de hembras en la Condic Inic (c/hembra pone en un tacho distinto)
 ntachito=5  !numero de tachitos 	  
 iovip=32    !numero de huevos hembras en cada oviposicion
 ttotal=400  !numero total de dias de la simulacion

 morhue=0.01d0   !mortalidad diaria huevos
 morlar=0.01d0   !mortalidad diaria larvas
 morpup=0.01d0   !mortalidad diaria pupas

 moracu=morhue+morlar+morpup  !mortalidad total de las acuaticas 


 morad= 0.01d0   !mortalidad diaria adultas 
 morpupad=0.17d0   !el 17 por ciento de las pupas no se vuelven adultas

 !print*,morhue,morlar,morpup,moracu,morad,morpupad!! no imprime nada, cuidado

 tpupad=17    !pupas se vuelven adultas que pican a los 17 dias
 
 tovip1=2     !tiempo entre dos oviposiciones (T=30)
 tovip2a=3    !tiempo entre dos oviposiciones (T=25)  
 tovip2b=4    !tiempo entre dos oviposiciones (T=25)   
 tovip3=10    !tiempo ente dos oviposiciones (T=18)

 sat=800      !numero maximo de huevos en cada tacho

! IDUM = -147 !semilla del generador de numeros aleatorios (c/ semilla es para una manzana)
! IDUM = -581 !semilla del generador de numeros aleatorios
! IDUM = -739 !semilla del generador de numeros aleatorios
 IDUM = -975  !semilla del generador de numeros aleatorios

 prop=0.6    !efectividad de la propaganda
 descach=nint(ntachito*prop)  !cantidad de tachos que vacío con la propaganda
 !print*,tpupad,tovip1,tovip2a,tovip2b,tovip3,sat,descach !imprime valores
 !write(*,*)
 !print*,inhem,ntachito,iovip,ttotal!imprime valores
!-------------------------------------------
! inicializo la poblacion mosquítica
!-------------------------------------------
 print*,'1) Condiciones iniciales para cada una de las hembras'
 mosquita=0
 indice=0      !ocupacion en la matriz (nro de bichos, va creciendo con el tiempo)

   do kk=1,inhem   
   mosquita(kk,1)=1     !vive
   idia=ran2(IDUM)*5+12 !tiene entre 12 y 17 días de vida
   mosquita(kk,2)=kk    !está en el tachito kk
   mosquita(kk,3)=idia  !tiene "idia" dias de vida
   mor=ran2(IDUM)*6+27 
   mosquita(kk,4)=mor   !dia de su muerte (entre los 27 y 32 dias)
   indice=indice+1
   print*,kk,mosquita(kk,1),mosquita(kk,2),mosquita(kk,3),mosquita(kk,4)
    end do
  
!-------------------------------------------------------------------	
do tiempo=1,ttotal  !empieza una realizacion del programa
!-------------------------------------------------------------------------
! defino la temperatura segun la estacion y el tiempo entre oviposiciones
!-------------------------------------------------------------------------
  if(tiempo.lt.80 .or. tiempo.gt.320) tovip=tovip1  !T=30
  if(tiempo.gt.160 .and. tiempo.lt.240) tovip=tovip3  !T=18
  if(tiempo.ge.80 .and. tiempo.le.160)then   !T=25
    if(ran2(IDUM).lt.0.5)then  !variabilidad en la frecuencia de oviposicion
    	 tovip=tovip2a
    else
   	 tovip=tovip2b
    end if
  end if
  if(tiempo.ge.240 .and. tiempo.le.320)then  !T=25
    if(ran2(IDUM).lt.0.5)then  !variabilidad en la frecuencia de oviposicion
	 tovip=tovip2a
    else
	 tovip=tovip2b
    end if
  end if
!---------------------------------------------------------------------
! cuento cuantos huevos hay en cada tacho
!---------------------------------------------------------------------- 
   tach=0
  
 do i=1,indice
   if(mosquita(i,3).lt.17.and.mosquita(i,1).eq.1) then !cuento solo acuaticos, es decir edad<tpupad
    j=mosquita(i,2) 
    tach(j)=tach(j)+1
   end if
 end do 
 !   print*,tiempo,tach(j)
!--------------------------------------------------------------------------
! MORTALIDADADES VARIAS (morirse antes de ser vieja): mato acuaticos con prob moracu,
! a las que estan por volverse adultas picadoras con prob morpupad
! y a las adultas con prob morad (no es la muerte por vejez, ojo)
!--------------------------------------------------------------------------
 do i =1,indice
	if (mosquita(i,1).eq.1.and.mosquita(i,3).lt.tpupad) then 
  	 if (ran2(IDUM).lt. 0.03d0)mosquita(i,1)=0  
  end if
	if (mosquita(i,1).eq.1.and.mosquita(i,3).eq.tpupad) then 
  	 if (ran2(IDUM).lt. 0.17d0)mosquita(i,1)=0  
  end if
	if (mosquita(i,1).eq.1.and.mosquita(i,3).gt.tpupad) then 
  	 if(ran2(IDUM).lt. 0.01d0)mosquita(i,1)=0  
   end if
!print*,mosquita(i,1)
	   end do  !del indice

! -------------------------------------------------------------------	    
! NACIMIENTOS CON SATURACION
!-------------------------------------------------------------------------- 
    !tovip=tovip1
    !tovip=tovip2b
    !tovip=tovip2a
    !tovip=tovip3
    do i =1,indice   !loop de toda la poblacion

   if (tach(mosquita(i,2)).lt.800) then    !solo si el tacho no está saturado 
if(mosquita(i,1).eq.1.and.mosquita(i,3).gt.17)then   !si estan vivas y están maduras
  if(mod(mosquita(i,3),tovip).eq.0) then   !cada tovip pone una camada de iovip huevos
   do ik=1,iovip 
 indice=indice+1  !indice de la nueva bicha
 mosquita(indice,1)=1 !está viva
 mosquita(indice,3)=1 		!tiene 1 dia   
 mosquita(indice,2)=mosquita(i,2)    !nace en el tacho de su madre 
 mor=ran2(IDUM)*6+27 
 mosquita(indice,4)=mor    !dias que va a vivir
 tach(mosquita(indice,2))=tach(mosquita(indice,2))+1 !sumo en 1 el contador de acuaticos
   end do
  end if    !del periodo entre oviposiciones y saturacion
end if      !de vivas y maduras
   end if   !de la saturac
!print*,tach(mosquita(i,2))
end do  ! de poblacion total (indice)
 
!--------------------------------------------------------------------------
!MUERTE POR VEJEZ: la mato cuando su edad alcanza el tiempo de vida asignado
!--------------------------------------------------------------------------
do i =1,indice
  if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.mosquita(i,4))mosquita(i,1)=0
!print*,mosquita(i,1)
end do 
!--------------------------------------------------------------------			  		 
!DESCACHARRADO: elimino los acuaticos de algunos tachos, 
!una vez por semana a partir del día 20 (esto se puede cambiar, obvio)
!-------------------------------------------------------------------- 
   if(mod(tiempo,7).eq.0.and.tiempo.gt.20) then
    do itach=1,descach!tachos que voy a sacar
    ntach=ran2(IDUM)*ntachito  !elijo uno de los tachitos al azar
	do i =1,indice    !y lo vacío
  	if (mosquita(i,1).eq.1.and.mosquita(i,3).lt.17.and.mosquita(i,2).eq.ntach)mosquita(i,1)=0
	!print*,mosquita(i,1)
	end do
    end do    
   end if
!--------------------------------------------------------------------			  		 
! cuento poblacion adulta y acuatica
!--------------------------------------------------------------------			  
  ad=0  !adultos total
  ac=0  !acuaticos vivos total 
  ad1=0 !adultos vivos por tacho
  ad2=0
  ad3=0
  ad4=0
  ad5=0
  adn1=0 !adultos nuevos por tacho
  adn2=0
  adn3=0
  adn4=0
  adn5=0
  
    do i =1,indice
if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.17)ad=ad+1 !adultos tot
if (mosquita(i,1).eq.1.and.mosquita(i,3).lt.17)ac=ac+1 !acuaticos tot

if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.17.and.mosquita(i,2).eq.1)ad1=ad1+1
if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.17.and.mosquita(i,2).eq.2)ad2=ad2+1
if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.17.and.mosquita(i,2).eq.3)ad3=ad3+1
if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.17.and.mosquita(i,2).eq.4)ad4=ad4+1
if (mosquita(i,1).eq.1.and.mosquita(i,3).ge.17.and.mosquita(i,2).eq.5)ad5=ad5+1

if (mosquita(i,1).eq.1.and.mosquita(i,3).eq.17.and.mosquita(i,2).eq.1)adn1=adn1+1
if (mosquita(i,1).eq.1.and.mosquita(i,3).eq.17.and.mosquita(i,2).eq.2)adn2=adn2+1
if (mosquita(i,1).eq.1.and.mosquita(i,3).eq.17.and.mosquita(i,2).eq.3)adn3=adn3+1
if (mosquita(i,1).eq.1.and.mosquita(i,3).eq.17.and.mosquita(i,2).eq.4)adn4=adn4+1
if (mosquita(i,1).eq.1.and.mosquita(i,3).eq.17.and.mosquita(i,2).eq.5)adn5=adn5+1


    end do 

!--------------------------------------------------------------------	   
!envejezco a toda la población un día (aumento en 1)
!--------------------------------------------------------------------
   do i =1,indice
	  if (mosquita(i,1).eq.1)mosquita(i,3)=mosquita(i,3)+1

   !print*,mosquita(i,3)
   end do  

!--------------------------------------------------------------------
!guardo los datos
!--------------------------------------------------------------------
	   write(3,*)tiempo,ad,ac,ad+ac		
	   write(1,*)tiempo,ad1,ad2,ad3,ad4,ad5	
	   write(2,*)tiempo,adn1,adn2,adn3,adn4,adn5	
	   !write(*,*)tiempo,ad

!-------------------------------------------------------------------	   
!-------------------------------------------------------------------	   
 end do  ! del tiempo
!-------------------------------------------------------------------	   
   close(2)
   close(1)

 	    end

!----------------------------------------------------------------------------
 FUNCTION ran2(idum)
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


!FUNCTION RAN2(IDUM)
! PARAMETER (M=714025,IA=1366,IC=150889,RM=1.4005112E-6)
! DIMENSION IR(97)
! save IR, IY
! DATA IFF /0/
! IF(IDUM.LT.0.OR.IFF.EQ.0)THEN
!   IFF=1
!   IDUM=MOD(IC-IDUM,M)
!   DO 11 J=1,97
!IDUM=MOD(IA*IDUM+IC,M)
!IR(J)=IDUM
!11 CONTINUE
!   IDUM=MOD(IA*IDUM+IC,M)
!   IY=IDUM
! ENDIF
! J=1+(97*IY)/M
! IF(J.GT.97.OR.J.LT.1)PAUSE
! IY=IR(J)
! RAN2=IY*RM
! IDUM=MOD(IA*IDUM+IC,M)
! IR(J)=IDUM
! RETURN
! END


# El Séptimo Campo

**Cómo construimos Uberth, qué nos corrigió el camino, y dónde quedamos.**

> Todas las cifras de este documento salen de corridas medidas. Donde un número
> viene de una sola semilla o de una sola corrida, está dicho.

---

## La pregunta con la que empezamos

La app le muestra al repartidor seis campos: punto A, punto B, pago, tiempo
estimado, kilómetros, y una banderita que dice si hay tarifa dinámica —nunca
cuánta. Tiene siete segundos para decidir.

Ninguno le dice lo único que importa: **cuánto vas a ganar por hora,
descontando los kilómetros que nadie te paga.**

Ese es el séptimo campo. La app nunca lo va a mostrar, porque optimiza la
tarifa para el cliente, no para el conductor. Así que lo calculamos nosotros.

---

## Construimos dos sistemas, no uno

Esta decisión la tomamos antes de escribir una línea, y de ella cuelga todo lo
demás.

No hicimos "un sistema con un agente adentro". Hicimos **dos sistemas
independientes**, cada uno con su propio kit de información y una frontera que
no se cruza.

**El mundo** puede usar lo que quiera, incluso cosas que solo existen en
México: el censo DENUE, la población INEGI por manzana, las sondas de tráfico
de TomTom. Es Monterrey y tiene derecho a ser mexicano.

**Uberth** no. Solo consume lo que se obtiene de **un par de coordenadas y una
pantalla de celular**: la app, OpenStreetMap, y su memoria de lo que le pasó.

El DENUE es México. OpenStreetMap es el planeta. Un agente que aprende a
depender de un dataset mexicano no es un producto, es una demo.

Y no lo dejamos a la disciplina: **16 tests** escanean el código del agente
buscando identificadores de ciudad y tiran la suite si aparece uno. Los
probamos metiendo violaciones a propósito, porque un test que nunca viste
fallar no prueba nada. Nos cazó a nosotros —alguien escribió "Monterrey" en un
comentario. La regla salió más estricta que el equipo.

---

## El mundo, y lo que nos enseñó

Nada de lo que hay debajo lo elegimos:

| capa | dato real |
|---|---|
| calles | OpenStreetMap — **95,429 nodos**, **242,223 aristas** |
| comercios | DENUE / INEGI — **12,943 restaurantes** |
| destinos | población INEGI por manzana — **2,330,207 personas** |
| tráfico | TomTom Traffic Stats, curva horaria medida de Monterrey |
| clima | archivo histórico Open-Meteo, fechas reales |
| geografía | retícula H3 resolución 7 — **127 celdas** |

Y el dato nos corrigió antes de que nos corrigiera la realidad.

Habíamos asumido que la curva de tráfico se transfiere entre metrópolis
mexicanas —mismos horarios de comida y de trabajo. La serie medida dijo que
no: **Monterrey hace pico al mediodía y no tiene el valle de media tarde** del
centro del país. Lo teníamos al revés, y el dato ganó.

---

## Las cuatro calibraciones

Uno de nosotros maneja. Miró las cifras del simulador y dijo que no cuadraban.
Esa conversación desató cuatro correcciones encadenadas, **cada una escondida
detrás de la anterior**.

### Estábamos calibrando contra el número equivocado

Ajustábamos para que el *total del turno* cayera en un rango plausible —y
caía, con **88 pesos por viaje a 1.2 entregas por hora**. Un total se puede
acertar con dos errores que se cancelan. Solo la cifra *por viaje* lo expone, y
para mirarla hacía falta alguien que hubiera hecho el trabajo.

### El radio de ofertas asumía un repartidor libre

Le ofrecíamos pedidos cuando estaba cerca del restaurante *en ese instante*,
pero a mitad de entrega nadie puede actuar sobre "en este instante". Al vuelo
eran **0.70 km**; encolados, **5.61**.

### Le aplicábamos tráfico de auto a una moto

Las sondas de TomTom son autos, y una moto de reparto se filtra entre
carriles. El embotellamiento no le cuesta lo mismo.

### Y debajo de los tres, el que tapaban

Con el ciclo ya corto, el repartidor pasaba **310 de 500 minutos** con la
pantalla vacía. Habíamos construido una ciudad sin trabajo.

> De paso aprendimos algo sobre tarifas que no habríamos deducido solos: cuando
> un repartidor dice *"los normales son de 30 a 40"*, no describe una pendiente,
> describe **el mismo precio repetido**. Eso es una tarifa mínima, y ninguna
> curva suave la imita. Tres intentos fallaron antes de entenderlo.

---

## Las reglas que le dimos a Uberth

Acá está el producto. Uberth no recibe conclusiones: corre su propio análisis
con **once factores** y estas reglas.

### Precio de reserva, no umbral

La barra no es un número que alguien eligió. Es **cuánto vale rechazar esta
oferta y esperar la siguiente**, calculado de la tasa de llegada y del turno
que queda. Es la única pieza no trivial del agente.

### Descuento por confianza

Cada estimación lleva valor, confianza y antigüedad, y el puntaje final se
descuenta por la confianza agregada. La consecuencia es la que importa: **el
primer turno en una ciudad desconocida es prudente, no temerario**, y se va
soltando conforme junta evidencia propia. Es la diferencia entre un agente que
se exporta y uno que apuesta.

### Sesgo del ETA por zona

El arbitraje más limpio que existe, y no necesita ninguna fuente externa:
compara lo que la app prometió contra lo que realmente pasó. Después de cien
viajes sabe cuánto le mienten, y por zona.

### Memoria de cocinas por sucursal

Cuánto tarda *esa* sucursal, no el promedio del ramo. Sin dato, usa un prior de
**9 minutos** y lo declara.

### Calidad del destino

Un pedido bien pagado que te vara en un desierto residencial cuesta un regreso
sin cobrar. Se calcula de la densidad de comercios en la celda de entrega, y
**voltea rankings de inmediato**.

### Clima como rampa, no como interruptor

Lluvia y calor entran de **32 a 42 °C** de forma continua: 38 y 44 grados no
son el mismo turno.

### Geometría de fin de turno

Nadie trabaja hasta que se le acaba el mundo: hay hora de salida y una casa a
la que volver, y los kilómetros de regreso no los paga nadie.

Este factor pesa más de lo que parece. Medido sobre un turno: **de 39 rechazos,
22 son geometría de cierre** y solo 13 son la barra siendo exigente. Quien mire
el tablero y piense "rechaza demasiado" está leyendo mal la mayoría de esos
rechazos.

### Idle flexible por evidencia, no por impaciencia

Lo obvio era un tope —"después de N minutos acepta lo que sea"—, pero eso es
arbitrario. Lo que hicimos es una corrección bayesiana: **una pantalla vacía
durante veinte minutos es un dato en contra de la tasa de llegada que el agente
creía.** La barra baja sola, de forma monótona, con un piso para que nunca
justifique trabajar perdiendo.

---

## Las configuraciones que lo mejoraron

Con el mundo ya corregido, las constantes de Uberth quedaron ajustadas a un
mundo que resultó ser un artefacto. Barrimos cientos de configuraciones en
paralelo, **con semillas de búsqueda y semillas que la búsqueda nunca vio**.

Tres perillas movieron el resultado de verdad.

### 1. La creencia de filtrado en moto

Habíamos arreglado el mundo para que la moto se filtrara y **nunca se lo
dijimos al agente**: seguía calculando cada tramo con el multiplicador completo
de auto, creyendo andar a **8.7 km/h** cuando el mundo lo movía a **31.8**.
Todo viaje le parecía ruinoso, y **los largos bien pagados los peores de
todos** —así se especializaba en trabajo corto y barato.

Lo metimos como variable a buscar, sin decirle el valor correcto:

| creencia | vs. la regla fija | peor caso |
|---|---|---|
| 1.00 (como estaba) | 98.4% | 44.3% |
| 0.40 | 95.7% | 66.4% |
| **0.15** | **100.6%** | **78.3%** |

La búsqueda caminó sola hasta **0.15 — exactamente la constante que el mundo ya
usaba.** Nadie se la dijo. Y no solo sube el promedio: el peor escenario mejora
de 44% a 78%.

### 2. La flexibilidad de aceptación

La barra es demostrablemente incompleta: sabe lo que cuesta esperar, pero no
sabe que **rechazar mucho hace que la app te muestre menos ofertas**. Así que le
dimos permiso de aceptar por debajo de su propia barra:

| flexibilidad | vs. la regla fija | acepta |
|---|---|---|
| 1.10 (más exigente) | 68.0% | 16.4% |
| 1.00 | 75.4% | 18.3% |
| **0.78** | **94.2%** | **22.5%** |
| 0.30 | 96.6% | 29.9% |

### 3. La aversión al riesgo bajó

De **0.35 a 0.10**: cobrarse demasiado por la exposición nocturna y el calor lo
volvía tímido sin compensación.

### Y dos perillas muertas

El tiempo de espera antes de reposicionarse y el costo de oportunidad por
minuto daban resultados **idénticos en todo su rango**. Uberth está ocioso 11 de
480 minutos: la rama de reposicionamiento nunca se activa. Saberlo vale tanto
como una mejora.

---

## Cómo quedó contra la regla de una sola línea

El rival es honesto y es duro: **acepta todo lo que pague más de 40 pesos.** Sin
modelo, sin memoria, sin clima. En un mundo donde el pago es la única señal
confiable, esa regla es mucho mejor de lo que suena.

| | vs. la regla fija | días ganados | aceptación |
|---|---|---|---|
| antes de afinar | 79.0% | 4 de 26 | 17.8% |
| **después** | **97.8%** | **10 de 26** | **25.6%** |

Y en los seis escenarios que grabamos:

| escenario | Uberth vs. la regla |
|---|---|
| Viernes común, limpio | 79.4% |
| Jueves caótico, 5 cierres | 95.8% |
| **Miércoles templado, limpio** | **125.3%** |
| Miércoles templado, 5 cierres | 114.5% |
| **Martes lluvioso, limpio** | **110.9%** |
| Martes lluvioso, 5 cierres | 104.3% |

**En pesos por kilómetro manejado —el número que le importa al repartidor,
porque los kilómetros los pone él— Uberth gana en los seis.** 7.17 contra 6.28.
8.33 contra 7.22. 9.48 contra 8.05.

En el promedio general todavía queda **2% abajo**, y decirlo en voz alta es lo
que hace creíble todo lo anterior. También sabemos exactamente por qué.

---

## Por qué esos 2%, y por qué es un hallazgo

La calidad de una oferta se abre **2.88 veces** entre la mejor y la peor. Pero
lo que la decide es **la espera de cocina** —que varía cuatro veces y **no tiene
ninguna relación con la distancia**, correlación de **0.003**— y eso **no
aparece en la tarjeta antes de aceptar**.

Cuando la única señal confiable es el pago, un umbral sobre el pago es casi lo
óptimo. **Uberth no pierde por tonto: pierde porque la información que decide un
viaje está oculta al momento de decidir.**

Ese es el hallazgo, y es el argumento de producto: el séptimo campo no se puede
calcular perfecto **porque la app no muestra lo que haría falta**. Uberth llega
al 97.8% con lo que hay visible. El resto está del otro lado del vidrio.

---

## Y todo en 0.86 milisegundos

Uberth decide en **0.86 ms**. La ventana de DiDi es de **siete mil**. Un turno
completo de ocho horas —**497 decisiones**— corre en menos de medio segundo.

No hay ningún modelo de lenguaje conectado. Las frases que se leen en el tablero
son los mismos números que ya decidieron, impresos. Si borrás todo el texto,
Uberth toma exactamente las mismas decisiones.

---

**El simulador es Monterrey. Uberth no es de ninguna parte.**

Seis campos en la app: el séptimo lo calcula él.

---

## Nota sobre cifras retiradas

Versiones anteriores del pitch citaban un piso de **55 MXN**, **120 tests**,
**21 ms** por decisión, y una tabla de resultados con **+43% por kilómetro
ganando 16 de 18 semillas**. Todas esas cifras son de antes de recalibrar la
tarifa y **no deben citarse**. Los valores vigentes son los de este documento:
piso de 40 MXN, 132 tests, 0.86 ms, y los resultados sobre 26 escenarios.

Ver `Docs/architecture/STATE.md` para el registro completo.

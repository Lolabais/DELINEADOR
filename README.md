EL CÓDIGO ORIGINAL HA SIDO PROGRAMADO EN PYTHON.
    CON ASESORÍA DE IA, SE HAN REPROGRAMADO LOS ALGORITMOS
    GEOESPACIALES.
    SEGUIR LOS SIGUIENTES PASOS:
    NO SE RECOMIENDA PARA CUENCAS GRANDES (> 500 MIL Ha.)
    El objetivo es delinear cuencas pequeñas o microcuencas que drenan a
    un punto de interés. Muy útil en captación de agua de lluvia.

    1.  Seleccione el Umbral de Acumulación:
        NÚMERO DE CELDAS AGUAS ARRIBA DEL
        PUNTO PARA CONSIDERAR QUE ES UN CAUCE.
        - Use el slider para definir la densidad de la red de drenaje.
        - Umbral bajo (~10): Red muy densa (más arroyos pequeños).
        - Umbral alto (~500): Red menos densa (ríos principales).
        - La visualización de la red puede variar según el umbral y la
          resolución del DEM.

    2.  Elija el Método de Delineación:
        a) Clic en Mapa:
           - Haga clic en "Delinear Nueva Cuenca (Clic en Mapa)".
           - Seleccione su archivo DEM (formato GeoTIFF).
           - Se mostrará un mapa interactivo con el DEM y la red de drenaje.
           - **¡IMPORTANTE! USO DEL ZOOM Y PAN:**
             Antes de hacer clic para seleccionar el punto de salida, use
             las herramientas de la barra de navegación de Matplotlib
             (generalmente en la parte inferior o superior de la ventana del gráfico):
               - **Zoom (Lupa):** Haga clic en el icono de la lupa. Luego,
                 haga clic y arrastre el mouse sobre el mapa para dibujar
                 un rectángulo. El mapa se acercará a esa área. Puede hacer
                 zoom repetidamente. Use los botones de flecha (izquierda/derecha)
                 en la barra para deshacer/recrear vistas.
               - **Pan (Flechas Cruzadas):** Haga clic en el icono de las
                 flechas cruzadas. Luego, haga clic y arrastre el mouse sobre
                 el mapa para mover la vista (desplazarse).
               - **Desactivar Zoom/Pan:** Haga clic nuevamente en el icono
                 activo (lupa o flechas) para desactivarlo ANTES de hacer
                 clic para seleccionar el punto.
           - Una vez que haya hecho zoom y/o pan para ver claramente el cauce
             deseado, **asegúrese de que ninguna herramienta (zoom/pan) esté activa**
             y haga clic directamente sobre la línea azul del cauce en el punto
             exacto de salida. El programa ajustará el clic al cauce más cercano
             con mayor acumulación.

        b) Coordenadas Manuales: Si conoce las coordenadas del punto
           - Haga clic en "Delinear Nueva Cuenca (Coords Manuales)".
           - Seleccione su archivo DEM (formato GeoTIFF).
           - Ingrese las coordenadas X (Longitud) e Y (Latitud) del punto
             de salida en las ventanas emergentes. Asegúrese de que las
             coordenadas estén dentro del área cubierta por su DEM y en el
             mismo Sistema de Coordenadas de Referencia (CRS).

    3.  Resultados:
        - Se mostrará un mapa con la cuenca delineada, la red de drenaje
          dentro de ella y el punto de salida ajustado (marcado en rojo).
        - Se calculará y mostrará el área de la cuenca.
        - Opcionalmente, puede guardar la cuenca como un archivo GeoTIFF.
        - Después de delinear, se habilitarán los botones para calcular
          índices morfométricos, escurrimiento e hidrograma.

    IMPORTANTE SOBRE UNIDADES Y COORDENADAS:
    - El programa ASUME que las unidades de elevación del DEM son METROS.
    - Si el DEM usa un Sistema de Coordenadas Proyectado (ej. UTM),
      el programa ASUME que las unidades de distancia son METROS.
    - Si el DEM usa Coordenadas Geográficas (Latitud/Longitud en grados),
      el programa realizará conversiones APROXIMADAS a metros/km, lo cual
      puede introducir imprecisiones, especialmente en áreas grandes.
    - Verifique siempre las unidades y el CRS de su DEM de entrada.
      Resultados incorrectos (muy altos o bajos) pueden deberse a
      unidades de entrada incorrectas (ej. pies en lugar de metros).
    - Problemas de 'nodata': Asegúrese que el valor nodata esté correctamente
      definido en su archivo GeoTIFF. El programa intenta leerlo, pero
      inconsistencias pueden causar errores.
    

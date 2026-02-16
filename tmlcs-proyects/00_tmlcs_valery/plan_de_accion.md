## Plan de Acción Detallado: Mejoras Futuras para Legacy-1bit LLM

Este plan de acción desglosa las "Mejoras Futuras" identificadas en el `README.md` en tareas concretas y ordenadas por fases lógicas.

---

### Fase 5: Persistencia del Modelo y Diagnóstico (Checkpointing y Métricas)

**Objetivo:** Permitir guardar/cargar el estado del modelo y obtener una visión más detallada del progreso del entrenamiento.

**1. Implementar Checkpointing del Modelo:**
    *   **Tarea 5.1.1: Definir formato de archivo para guardar el modelo:** Decidir una estructura para serializar la `struct LegacyLLM` (pesos, sesgos, embeddings) a disco. Un formato binario simple es preferible para eficiencia.
    *   **Tarea 5.1.2: Crear función `save_model`:** Desarrollar `void save_model(LegacyLLM* model, const char* filepath)` en `src/model.c` que escriba el estado actual del modelo al archivo especificado.
    *   **Tarea 5.1.3: Crear función `load_model`:** Desarrollar `LegacyLLM* load_model(const char* filepath)` en `src/model.c` que lea un archivo y reconstruya la `struct LegacyLLM`.
    *   **Tarea 5.1.4: Integrar guardar/cargar en `main.c`:** Modificar el bucle de entrenamiento para guardar el modelo periódicamente (ej., cada N épocas) y añadir la opción de cargar un modelo al inicio si existe un checkpoint.
    *   **Tarea 5.1.5: Actualizar `model.h`:** Añadir las declaraciones de `save_model` y `load_model`.
    *   **Tarea 5.1.6: Implementar pruebas unitarias para checkpointing:** Añadir tests en `tests/test_llm.c` para verificar que el modelo se guarda y carga correctamente y que los estados son idénticos.

**2. Añadir Métricas de Entrenamiento Avanzadas y Logging:**
    *   **Tarea 5.2.1: Definir métricas adicionales:** Considerar métricas como la perplejidad, o una métrica de "precisión" simple si se puede definir para la predicción de siguiente token.
    *   **Tarea 5.2.2: Modificar el bucle de entrenamiento en `main.c`:** Calcular e imprimir estas nuevas métricas además de la pérdida promedio.
    *   **Tarea 5.2.3: Implementar una función de logging simple:** Crear una función como `void log_training_progress(int epoch, float loss, float perplexity, ...)` para centralizar la salida de información.
    *   **Tarea 5.2.4 (Opcional): Implementar logging a archivo:** Permitir que la información de entrenamiento se guarde en un archivo de log para análisis posterior.

---

### Fase 6: Evaluación de Rendimiento y Mejoras del Core

**Objetivo:** Cuantificar el impacto de las optimizaciones SSE y mejorar el manejo de datos.

**3. Realizar Análisis de Rendimiento (SSE vs. Non-SSE):**
    *   **Tarea 6.1.1: Implementar utilidades de temporización básicas:** Crear funciones para medir el tiempo de ejecución (ej., usando `clock()` o `gettimeofday` en C).
    *   **Tarea 6.1.2: Añadir puntos de temporización en funciones críticas de `math_ops.c`:** Medir el tiempo de ejecución de las versiones SSE y no SSE de operaciones como la multiplicación de matrices, softmax, etc.
    *   **Tarea 6.1.3: Modificar `main.c` o añadir un nuevo test de benchmark:** Crear un modo de ejecución que compare el rendimiento con `USE_SSE_BUILD=0` y `USE_SSE_BUILD=1` para operaciones clave o un subconjunto del entrenamiento.
    *   **Tarea 6.1.4: Reportar diferencias de rendimiento:** Presentar los resultados de las mediciones de tiempo para cuantificar las ganancias de SSE.

**4. Mejorar el Manejo de Datos:**
    *   **Tarea 6.2.1: Implementar batching dinámico:** Ajustar `src/data_utils.c` para agrupar secuencias de longitud similar y/o implementar un relleno eficiente (padding) para manejar diferentes tamaños de secuencia dentro de un batch, mejorando la eficiencia del hardware.
    *   **Tarea 6.2.2: Explorar la integración de datasets más grandes:** Modificar `src/data_utils.c` para manejar archivos de texto más grandes de manera eficiente (ej., lectura por chunks, memory mapping si es apropiado para el hardware destino).
    *   **Tarea 6.2.3 (Opcional): Investigar y añadir un tokenizador más sofisticado:** Explorar la implementación de un tokenizador Byte-Pair Encoding (BPE) o similar para una mejor gestión de vocabulario y manejo de palabras fuera de vocabulario.

---

### Fase 7: Exploración del Modelo e Inferencia

**Objetivo:** Habilitar la generación de texto y experimentar con variaciones del modelo.

**5. Implementar Modo de Inferencia:**
    *   **Tarea 7.1.1: Crear función `generate_text`:** Desarrollar `char* generate_text(LegacyLLM* model, const char* prompt, int max_length)` que tome un prompt y genere una secuencia de texto.
    *   **Tarea 7.1.2: Implementar muestreo para la selección de tokens:** Añadir lógica para seleccionar el siguiente token (ej., muestreo codicioso, top-k o top-p si se considera la complejidad).
    *   **Tarea 7.1.3: Integrar el modo de inferencia en `main.c`:** Permitir al usuario cambiar entre el modo de entrenamiento y el modo de inferencia mediante argumentos de línea de comandos.

**6. Explorar Diferentes Esquemas de Cuantización Ternaria:**
    *   **Tarea 7.2.1: Investigar métodos alternativos de cuantización:** Estudiar literatura sobre diferentes umbrales o métodos estocásticos para cuantizar pesos a -1, 0, 1.
    *   **Tarea 7.2.2: Implementar un esquema de cuantización alternativo:** Desarrollar una nueva lógica de cuantización en `src/model.c` o un nuevo archivo.
    *   **Tarea 7.2.3: Modificar `apply_ternary_weight_updates`:** Permitir la selección entre el esquema de cuantización actual y el nuevo mediante un parámetro del modelo o una macro de compilación.

---

Este plan ofrece un camino estructurado para continuar el desarrollo del Legacy-1bit LLM, permitiendo abordar las mejoras de manera incremental.

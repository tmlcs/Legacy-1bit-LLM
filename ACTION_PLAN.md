# Plan de Acción Detallado del Proyecto Legacy-1bit LLM

Este plan de acción se basa en los hallazgos de la auditoría de calidad realizada, centrándose en las áreas de mejora identificadas y las futuras mejoras mencionadas en el `README.md` del proyecto. Las tareas están priorizadas para guiar el desarrollo.

## I. Mejoras de Testing (Prioridad: Alta)

La expansión de la cobertura y robustez de las pruebas es fundamental para asegurar la corrección y el rendimiento del modelo.

### 1. **Expansión de Pruebas Unitarias**
*   **Descripción:** Añadir pruebas unitarias exhaustivas para funciones críticas en `math_ops.c`, `forward.c`, `backward.c` y `model.c` (específicamente para `apply_ternary_weight_updates`).
*   **Tareas:**
    *   Identificar todas las funciones críticas que carecen de pruebas unitarias adecuadas.
    *   Crear casos de prueba para cubrir diferentes escenarios (entradas válidas, bordes, errores).
    *   Implementar nuevas funciones de prueba en `tests/test_llm.c` utilizando el framework de pruebas existente.
    *   Asegurar que tanto las implementaciones con SSE como sin SSE de `math_ops.c` estén cubiertas por pruebas unitarias.
*   **Esfuerzo Estimado:** Grande

### 2. **Pruebas de Integración**
*   **Descripción:** Desarrollar pruebas que validen el flujo de datos y la interacción entre múltiples componentes, como un pase `forward` completo o un ciclo `forward-backward` con pequeñas entradas conocidas.
*   **Tareas:**
    *   Definir un conjunto de entradas y salidas esperadas para un pase `forward` completo a través de una o más capas del modelo.
    *   Definir un conjunto de entradas, salidas y gradientes esperados para un ciclo `forward-backward` simplificado.
    *   Implementar estas pruebas de integración en `tests/test_llm.c`.
*   **Esfuerzo Estimado:** Medio

### 3. **Pruebas de Rendimiento**
*   **Descripción:** Crear pruebas específicas para cuantificar las ganancias de rendimiento de la optimización SSE, comparando el tiempo de ejecución de operaciones clave con y sin SSE.
*   **Tareas:**
    *   Identificar operaciones matemáticas clave que se benefician de SSE (ej., multiplicación de matrices, operaciones de activación).
    *   Desarrollar micro-benchmarks para medir el tiempo de ejecución de estas operaciones en ambas configuraciones (SSE y no SSE).
    *   Integrar estos benchmarks en el proceso de prueba o en un objetivo `perf` mejorado.
*   **Esfuerzo Estimado:** Medio

## II. Mejoras en la Calidad del Código (Prioridad: Media)

Refinar la legibilidad y modularidad del código para facilitar el mantenimiento y la comprensión.

### 1. **Comentarios Detallados para Algoritmos Complejos**
*   **Descripción:** Añadir comentarios explicativos en secciones de código que implementan algoritmos complejos (ej., detalles del `forward` y `backward` pass, lógica de gradient checkpointing, normalización de capas).
*   **Tareas:**
    *   Revisar `forward.c`, `backward.c`, `math_ops.c`, `model.c` e `legacy_llm.h`.
    *   Añadir comentarios que expliquen el "por qué" y el "cómo" de las implementaciones.
*   **Esfuerzo Estimado:** Medio

### 2. **Modularización del Bucle de Entrenamiento en `main.c`**
*   **Descripción:** Refactorizar la lógica principal del bucle de entrenamiento en `main.c` en funciones de utilidad más pequeñas y dedicadas.
*   **Tareas:**
    *   Extraer la inicialización del modelo y gradientes en una función.
    *   Extraer el manejo de la carga de datos y tokenización en funciones específicas.
    *   Crear una función para un paso de entrenamiento individual (forward, backward, update).
    *   Crear una función para el bucle de época (iterar sobre batches, calcular pérdida promedio).
    *   Actualizar `main.c` para utilizar estas nuevas funciones.
*   **Esfuerzo Estimado:** Medio

## III. Mejoras del Sistema de Construcción (`Makefile`) (Prioridad: Media)

Optimizar el `Makefile` para una mayor flexibilidad y claridad.

### 1. **Objetivos de Prueba Separados**
*   **Descripción:** Modificar el `Makefile` para permitir la ejecución de pruebas para compilaciones con y sin SSE de forma independiente.
*   **Tareas:**
    *   Crear objetivos `test_no_sse` y `test_sse` que compilen y ejecuten las pruebas correspondientes.
    *   Mantener el objetivo `test` actual como un objetivo "all_tests" que ejecute ambos.
*   **Esfuerzo Estimado:** Pequeño

### 2. **Evaluación y Mejora del Script `analyze_perf.sh`**
*   **Descripción:** Revisar el contenido del script `analyze_perf.sh` y mejorarlo para proporcionar un análisis de rendimiento más significativo.
*   **Tareas:**
    *   Leer y entender la lógica actual del script.
    *   Asegurarse de que el script extraiga y presente métricas de rendimiento relevantes (ej., tiempo por época, tiempo por paso de forward/backward, etc.).
    *   Considerar la visualización básica o la exportación a un formato analizable (ej., CSV).
*   **Esfuerzo Estimado:** Pequeño a Medio

## IV. Futuras Mejoras (Prioridad: Media a Baja, según necesidad)

Estas son características adicionales que podrían implementarse para mejorar la funcionalidad y usabilidad del proyecto.

### 1. **Registro (Logging) Avanzado**
*   **Descripción:** Implementar un sistema de registro más robusto que `printf` para guardar el progreso del entrenamiento, métricas y posibles errores en un archivo.
*   **Tareas:**
    *   Diseñar una API de logging simple (ej., `log_info`, `log_error`).
    *   Integrar el logging en el bucle de entrenamiento y otras funciones críticas.
    *   Permitir la configuración del nivel de logging.
*   **Esfuerzo Estimado:** Medio

### 2. **Gestión de Hiperparámetros Externa**
*   **Descripción:** Permitir la configuración de hiperparámetros (ej., `LEARNING_RATE`, `NUM_EPOCHS`, `MODEL_DIM`) a través de argumentos de línea de comandos o un archivo de configuración simple en lugar de hardcodearlos.
*   **Tareas:**
    *   Modificar `main.c` para analizar argumentos de línea de comandos.
    *   Opcional: Implementar la lectura de un archivo de configuración (ej., formato clave-valor simple).
*   **Esfuerzo Estimado:** Medio

### 3. **Modo de Inferencia Dedicado**
*   **Descripción:** Desarrollar un modo separado que permita cargar un modelo entrenado y generar texto basado en un prompt dado, sin la sobrecarga de los componentes de entrenamiento.
*   **Tareas:**
    *   Crear una nueva función `infer_llm` que tome un prompt y un modelo.
    *   Manejar la tokenización de entrada para inferencia.
    *   Generar secuencias de tokens utilizando muestreo.
    *   Crear un nuevo ejecutable o un flag en `main.c` para activar el modo de inferencia.
*   **Esfuerzo Estimado:** Grande

### 4. **Métricas de Entrenamiento Avanzadas**
*   **Descripción:** Integrar métricas de evaluación más allá de la pérdida, como la perplejidad y, si es aplicable, la precisión si se añade un cabezal de clasificación.
*   **Tareas:**
    *   Calcular y reportar perplejidad por época (ya parcialmente implementado).
    *   Explorar otras métricas relevantes para LLMs.
*   **Esfuerzo Estimado:** Pequeño

## V. Refinamientos (Prioridad: Baja)

Estas tareas son menos críticas pero contribuirían a la madurez general del proyecto.

### 1. **Exploración de Esquemas de Cuantificación Ternaria Alternativos**
*   **Descripción:** Investigar y posiblemente experimentar con otros métodos para cuantificar pesos a -1, 0, 1, o incluso esquemas de cuantificación de bits bajos diferentes (ej., binarios, 2 bits).
*   **Tareas:**
    *   Investigación de literatura sobre cuantificación de pesos.
    *   Prototipo de un esquema alternativo.
*   **Esfuerzo Estimado:** Grande (Investigación y Desarrollo)

### 2. **Manejo Mejorado de Conjuntos de Datos**
*   **Descripción:** Mejorar la pipeline de datos para soportar batching dinámico, conjuntos de datos más grandes y tokenización más eficiente.
*   **Tareas:**
    *   Implementar un cargador de datos que pueda manejar archivos de texto más grandes.
    *   Considerar estrategias de pre-procesamiento de datos.
*   **Esfuerzo Estimado:** Medio a Grande

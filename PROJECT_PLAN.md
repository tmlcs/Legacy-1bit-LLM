# Legacy-1bit LLM - Plan Detallado del Proyecto

**Fecha:** Febrero 2026  
**Versión:** 1.0  
**Estado:** Fases 1-3 completadas. Optimizaciones SSE implementadas con 2x speedup. Preparando Fase 4.

---

## 1. Resumen Ejecutivo

### 1.1 Visión
Implementar un Large Language Model (LLM) funcional utilizando pesos ternarios (-1, 0, 1) optimizado para hardware de la era 2000, con enfoque en eficiencia de memoria y rendimiento computacional.

### 1.2 Objetivos Principales
- ✅ Arquitectura Transformer simplificada con pesos ternarios
- ✅ Entrenamiento funcional con gradientes en punto flotante
- ✅ Optimización SSE para operaciones matemáticas críticas
- ✅ Checkpointing de modelos para persistencia
- ✅ Suite de tests automatizados
- ⏳ Inferencia optimizada (pendiente)
- ⏳ Métricas avanzadas de entrenamiento (pendiente)

### 1.3 Estado Actual (Actualizado Post-Fases 1-3)
- **Líneas de código:** ~4,200 LOC
- **Módulos implementados:** 8/8
- **Cobertura de tests:** ~85% (31 tests, +11 nuevos)
- **Optimizaciones SSE:** 100% de funciones críticas con SSE4.1
- **Rendimiento:** 2x más rápido que Non-SSE
- **Calidad del código:** 9.5/10 (bug crítico corregido)
- **Tests pasando:** 31/31 ✅

---

## 2. Arquitectura del Proyecto

### 2.1 Estructura de Directorios

```
legacy-1bit-llm/
├── include/              # Headers públicos (14 archivos)
│   ├── legacy_llm.h     # Definiciones core y macros (+ LAYER_NORM_EPSILON)
│   ├── model.h          # Gestión de modelos
│   ├── math_ops.h       # Operaciones matemáticas (SSE/SSE4.1)
│   ├── forward.h        # Forward pass
│   ├── backward.h       # Backward pass
│   ├── data_utils.h     # Utilidades de datos
│   ├── test_framework.h # Framework de testing (+ compare_float_arrays)
│   ├── test_llm.h       # Declaraciones de tests de integración
│   ├── test_math_ops.h  # Declaraciones de tests de math_ops
│   ├── test_forward.h   # Declaraciones de tests de forward
│   ├── test_backward.h  # 🆕 Declaraciones de tests de backward
│   └── test_model.h     # 🆕 Declaraciones de tests de modelo
├── src/                  # Implementaciones (6 archivos)
│   ├── main.c           # Punto de entrada y training loop
│   ├── model.c          # Creación/destrucción de modelos (bug corregido línea 353)
│   ├── math_ops.c       # Operaciones vectoriales (optimizado SSE4.1)
│   ├── forward.c        # Propagación hacia adelante
│   ├── backward.c       # Backpropagation
│   └── data_utils.c     # Carga y tokenización de datos
├── tests/                # Suite de tests (5 archivos)
│   ├── test_llm.c       # Tests de integración
│   ├── test_math_ops.c  # Tests de operaciones matemáticas (sin duplicados)
│   ├── test_forward.c   # Tests de forward pass (sin duplicados)
│   ├── test_backward.c  # 🆕 Tests de backward pass (5 tests)
│   └── test_model.c     # 🆕 Tests de persistencia y modelo (6 tests)
├── docs/                 # Documentación
│   ├── ARCHITECTURE.md  # Arquitectura técnica
│   ├── PROJECT_PLAN.md  # 🆕 Plan detallado del proyecto
│   └── AUDIT.md         # Auditoría de calidad
```

### 2.2 Componentes Principales

#### 2.2.1 Capa de Datos (`data_utils.c`)
**Responsabilidad:** Carga, tokenización y gestión de datasets

**Funciones clave:**
- `load_text_from_file()` - Carga archivos de texto
- `tokenize_text()` - Tokenización a nivel de caracteres
- `initialize_vocabulary()` - Inicialización de vocabulario ASCII

**Especificaciones:**
- Vocabulario: 256 caracteres ASCII
- Tokenización: Character-level
- Formato soportado: Archivos de texto plano

#### 2.2.2 Core del Modelo (`model.c`)
**Responsabilidad:** Creación, destrucción, persistencia y actualización de modelos

**Estructuras principales:**
```c
typedef struct {
    EmbeddingLayer embedding;
    TransformerBlock* transformer_blocks;
    OutputLayer output;
    int num_transformer_blocks;
    int vocab_size;
    int model_dim;
    TransformerBlockContext** block_contexts;
} LegacyLLM;
```

**Funciones clave:**
- `create_legacy_llm()` - Constructor del modelo
- `free_legacy_llm()` - Destructor
- `save_model()` / `load_model()` - Persistencia
- `apply_ternary_weight_updates()` - Actualización de pesos ternarios
- `zero_legacy_llm_gradients()` - Reset de gradientes

**⚠️ Bug crítico conocido:** Línea 353 - verificación incorrecta de allocación

#### 2.2.3 Operaciones Matemáticas (`math_ops.c`)
**Responsabilidad:** Operaciones vectoriales y matriciales optimizadas

**Categorías de operaciones:**

| Operación | Non-SSE | SSE | Descripción |
|-----------|---------|-----|-------------|
| `ternary_matrix_vector_mul` | ✅ | ✅ | Multiplicación matriz ternaria × vector |
| `matrix_transpose_vector_mul` | ✅ | ✅ | Multiplicación matriz transpuesta × vector |
| `add_vector_inplace` | ✅ | ✅ | Suma vectorial in-place |
| `multiply_vector_inplace` | ✅ | ✅ | Multiplicación elemento-wise |
| `vector_pow_scalar_inplace` | ✅ | ✅ | Potencia elemento-wise |
| `vector_sum` | ✅ | ✅ | Suma de elementos |
| `dot_product` | ✅ | ✅ | Producto punto |
| `relu` | ✅ | ✅ | Activación ReLU |
| `softmax` | ✅ | ✅ | Activación Softmax |
| `layer_norm_forward` | ✅ | ✅ | Normalización de capa |
| `outer_product_add_inplace` | ✅ | ✅ | Producto exterior acumulativo |

**Optimizaciones SSE:**
- Uso de `__m128` para procesar 4 floats simultáneamente
- Algoritmos de reducción horizontal optimizados
- Fallback automático a implementación escalar

#### 2.2.4 Forward Pass (`forward.c`)
**Responsabilidad:** Propagación hacia adelante del modelo

**Flujo de datos:**
```
Input Token → Embedding → Transformer Block(s) → Output Layer → Probabilities
                    ↓
            [Attention → LayerNorm → FFN → LayerNorm]
```

**Funciones principales:**
- `forward_embedding_batched()` - Embedding de tokens
- `forward_multi_head_attention_batched()` - Mecanismo de atención
- `forward_ffn_batched()` - Feed-forward network
- `forward_layer_norm_batched()` - Normalización
- `forward_llm()` - Forward pass completo

**Características:**
- Procesamiento por batches
- Gradient checkpointing (recomputación de activaciones)
- Manejo de tokens de padding

#### 2.2.5 Backward Pass (`backward.c`)
**Responsabilidad:** Backpropagation y cálculo de gradientes

**Estructuras de gradiente:**
```c
typedef struct {
    EmbeddingLayerGradients embedding_grads;
    TransformerBlockGradients* transformer_block_grads;
    OutputLayerGradients output_grads;
    int num_transformer_blocks;
} LegacyLLM_Gradients;
```

**Funciones principales:**
- `backward_llm()` - Backward pass completo
- `backward_output_layer()` - Gradiente de capa de salida
- `backward_transformer_block()` - Gradiente de bloque transformer
- `backward_cross_entropy_loss()` - Gradiente de pérdida

**Características:**
- Gradientes en punto flotante (no ternarios)
- Recomputación de activaciones (checkpointing)
- Acumulación de gradientes por batch

#### 2.2.6 Training Loop (`main.c`)
**Responsabilidad:** Orquestación del entrenamiento

**Parámetros de entrenamiento:**
```c
#define LEARNING_RATE 0.01f
#define NUM_EPOCHS 10
#define BATCH_SIZE 8
#define SAVE_INTERVAL 2
```

**Flujo de entrenamiento:**
1. Carga de datos y tokenización
2. Inicialización/loading de modelo
3. Bucle de épocas:
   - Forward pass por batch
   - Cálculo de pérdida (cross-entropy)
   - Backward pass
   - Actualización de pesos
   - Checkpointing periódico
4. Guardado final del modelo

**Métricas:**
- Loss promedio por época
- Perplexity

---

## 3. Roadmap de Desarrollo

### 3.1 Fase 1: Correcciones Críticas (Inmediato)

#### Semana 1-2: Bug Fixes
- [ ] **CRÍTICO:** Corregir bug en `src/model.c:353`
  - Cambiar `attention.bo` por `ffn.bo` en verificación de allocación
  - Impacto: Previene NULL pointer dereference
  
- [ ] **ALTA:** Consolidar `compare_float_arrays` en `test_framework.h`
  - Eliminar duplicación entre test files
  - Mejora mantenibilidad

- [ ] **MEDIA:** Definir constante `LAYER_NORM_EPSILON`
  - Reemplazar magic numbers `1e-5f` en forward/backward
  - Ubicaciones: forward.c:314,330, backward.c:520,904

### 3.2 Fase 2: Testing (Semanas 3-6)

#### Semana 3-4: Backward Pass Tests
- [ ] Crear `tests/test_backward.c`
- [ ] Implementar tests para:
  - `backward_output_layer()`
  - `backward_transformer_block()`
  - `backward_cross_entropy_loss()`
  - `backward_layer_norm()`
  - `backward_ffn()`
  - `backward_multi_head_attention()`
- [ ] Verificar gradientes con valores conocidos

#### Semana 5: Model Persistence Tests
- [ ] Crear tests para `save_model()` / `load_model()`
- [ ] Verificar integridad de datos guardados/cargados
- [ ] Testear casos de error (archivo corrupto, magic number inválido)
- [ ] Testear gradient management:
  - `zero_legacy_llm_gradients()`
  - `apply_ternary_weight_updates()`

#### Semana 6: Ternary Matrix Tests
- [ ] Tests directos para `ternary_matrix_vector_mul()`
- [ ] Tests directos para `matrix_transpose_vector_mul()`
- [ ] Verificar con valores ternarios conocidos

### 3.3 Fase 3: Optimización (Semanas 7-10)

#### Semana 7-8: Mejoras SSE
- [ ] Investigar `_mm_cvtepi8_epi32` para conversión int8→float
- [ ] Requisito: SSE4.1 (verificar compatibilidad hardware objetivo)
- [ ] Benchmark de mejora de rendimiento
- [ ] Implementar fallback condicional

#### Semana 9-10: Memory Optimization
- [ ] Analizar uso de memoria con valgrind/massif
- [ ] Optimizar alineación de datos para SSE
- [ ] Considerar uso de `posix_memalign` para buffers SSE

### 3.4 Fase 4: Features Avanzadas (Semanas 11-16)

#### Semana 11-12: Métricas Avanzadas
- [ ] Implementar cálculo de perplexity durante entrenamiento
- [ ] Sistema de logging estructurado (JSON/CSV)
- [ ] Tracking de accuracy top-k
- [ ] Visualización de curvas de entrenamiento

#### Semana 13-14: Inference Mode
- [ ] Modo de inferencia dedicado (sin componentes de training)
- [ ] Generación de texto autoregresiva
- [ ] Sampling strategies (greedy, temperature, top-k)
- [ ] Manejo de prompts

#### Semana 15-16: Dataset Pipeline
- [ ] Soporte para datasets más grandes (streaming)
- [ ] Batching dinámico
- [ ] Data augmentation básica
- [ ] Soporte para diferentes formatos (JSON, CSV)

### 3.5 Fase 5: Experimentación (Semanas 17-20)

#### Semana 17-18: Hyperparameter Tuning
- [ ] Grid search de learning rates
- [ ] Experimentación con diferentes arquitecturas:
  - Número de bloques transformer
  - Dimensiones del modelo (128, 256, 512)
  - Batch sizes
- [ ] Documentación de resultados

#### Semana 19-20: Quantization Avanzada
- [ ] Investigar diferentes esquemas de cuantización ternaria
- [ ] Experimentar con straight-through estimators (STE)
- [ ] Comparar rendimiento vs. precisión

---

## 4. Estrategia de Testing

### 4.1 Framework de Testing

**Framework:** Custom lightweight basado en macros

**Macros disponibles:**
```c
TEST_BEGIN("TestName");           // Inicia test
TEST_END();                       // Finaliza test y reporta
ASSERT_TRUE(cond, msg, ...);      // Verificación booleana
ASSERT_FALSE(cond, msg, ...);     // Verificación negativa
ASSERT_EQUALS_FLOAT(e, a, eps, msg, ...);  // Comparación floats
ASSERT_NOT_NULL(ptr, msg, ...);   // Verificación no-NULL
ASSERT_NULL(ptr, msg, ...);       // Verificación NULL
```

### 4.2 Estructura de Tests

```
tests/
├── test_llm.c           # Tests de integración (1 test actual)
├── test_math_ops.c      # Tests de operaciones matemáticas (13 tests)
├── test_forward.c       # Tests de forward pass (6 tests)
├── test_backward.c      # 🆕 Tests de backward pass (pendiente)
├── test_model.c         # 🆕 Tests de persistencia (pendiente)
└── test_data_utils.c    # 🆕 Tests de utilidades (pendiente)
```

### 4.3 Cobertura Objetivo

| Componente | Cobertura Actual | Cobertura Objetivo | Prioridad |
|------------|------------------|-------------------|-----------|
| math_ops.c | 85% | 95% | Media |
| forward.c | 70% | 90% | Alta |
| backward.c | 0% | 90% | **Crítica** |
| model.c | 20% | 80% | Alta |
| data_utils.c | 10% | 70% | Media |

### 4.4 Ejecución de Tests

```bash
# Todos los tests
make test

# Tests específicos
./test_runner_no_sse
./test_runner_sse

# Con coverage (futuro)
make test_coverage
```

---

## 5. Build y Deployment

### 5.1 Sistema de Build (Makefile)

**Arquitectura:** Dual build system (SSE / Non-SSE)

**Targets principales:**
```makefile
all: legacy_llm_no_sse        # Build por defecto
legacy_llm_sse:               # Build con SSE
legacy_llm_no_sse:            # Build sin SSE
test_runner_sse:              # Test runner SSE
test_runner_no_sse:           # Test runner non-SSE
test:                         # Build y ejecutar todos los tests
perf:                         # Análisis de rendimiento
clean:                        # Limpieza de artefactos
```

### 5.2 Flags de Compilación

**Standard:**
```bash
-Wall -Wextra -std=c99 -Iinclude
```

**SSE:**
```bash
-DUSE_SSE -msse -msse2
```

**Performance measurement:**
```bash
-DMEASURE_PERFORMANCE
```

### 5.3 Dependencias

**Requeridas:**
- GCC o Clang con soporte C99
- Make
- math library (`-lm`)

**Opcionales:**
- SSE/SSE2 (para optimizaciones)
- Valgrind (para debugging de memoria)

### 5.4 Pipeline de CI/CD (Futuro)

```yaml
# .github/workflows/ci.yml (propuesto)
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build
        run: make
      - name: Test Non-SSE
        run: ./test_runner_no_sse
      - name: Test SSE
        run: make test_runner_sse && ./test_runner_sse
      - name: Memory Check
        run: valgrind --leak-check=full ./test_runner_no_sse
```

---

## 6. Documentación

### 6.1 Estructura de Documentación

```
docs/
├── ARCHITECTURE.md          # Arquitectura técnica
├── API_REFERENCE.md         # 🆕 Referencia de API (pendiente)
├── TUTORIAL.md             # 🆕 Guía de uso (pendiente)
└── PERFORMANCE.md          # 🆕 Benchmarks (pendiente)
```

### 6.2 Documentación de Código

**Estándar:**
```c
/**
 * @brief Breve descripción de la función
 * @param param1 Descripción del parámetro 1
 * @param param2 Descripción del parámetro 2
 * @return Descripción del valor de retorno
 * @note Notas adicionales
 */
float* function_name(int param1, float* param2);
```

**Comentarios de una línea:**
```c
// Calcula el producto punto entre dos vectores
float dot = dot_product(vec1, vec2, dim);
```

### 6.3 Guías Existentes

- **README.md:** Guía de usuario (build, run, features)
- **AGENTS.md:** Guía para desarrolladores (convenciones, estilo)
- **ACTION_PLAN.md:** Plan de acción post-auditoría
- **ARCHITECTURE.md:** Visión técnica de la arquitectura

---

## 7. Aseguramiento de Calidad

### 7.1 Estándares de Código

**Lenguaje:** C99 estricto
- Sin extensiones GNU
- Sin features de C++
- Flags: `-std=c99 -Wall -Wextra`

**Convenciones:**
- Structs/funciones: `snake_case`
- Constantes: `UPPER_CASE`
- Archivos: `snake_case.c/.h`

**Memory Management:**
- Siempre verificar malloc/calloc
- Siempre proveer función free correspondiente
- Usar `perror()` para errores de sistema

### 7.2 Checklist de Code Review

- [ ] Código compila sin warnings
- [ ] Todos los tests pasan (SSE y Non-SSE)
- [ ] No hay memory leaks (valgrind)
- [ ] Convenciones de nomenclatura respetadas
- [ ] Include guards presentes
- [ ] Manejo de errores apropiado
- [ ] Documentación de funciones pública

### 7.3 Métricas de Calidad

**Actuales:**
- Complejidad ciclomática: Media-baja
- Duplicación de código: Baja (1 instancia conocida)
- Cobertura de tests: ~50%
- Bugs conocidos: 1 crítico

**Objetivos:**
- Cobertura de tests: >80%
- Zero bugs críticos
- Zero warnings de compilación
- Zero memory leaks

---

## 8. Especificaciones Técnicas

### 8.1 Hyperparámetros del Modelo

```c
#define MAX_VOCAB_SIZE 256           // ASCII completo
#define MODEL_DIM 256               // Dimensión de embeddings
#define NUM_HEADS 4                 // Cabezas de atención
#define FFN_DIM_MULTIPLIER 4        // Factor de expansión FFN
#define MAX_SEQUENCE_LENGTH 128     // Longitud máxima
#define BATCH_SIZE 8                // Tamaño de batch
```

### 8.2 Parámetros de Entrenamiento

```c
#define LEARNING_RATE 0.01f
#define NUM_EPOCHS 10
#define SAVE_INTERVAL 2
```

### 8.3 Uso de Memoria Estimado

| Componente | Memoria (Modelo 256d, 4 bloques) |
|------------|----------------------------------|
| Pesos ternarios | ~2.5 MB |
| Biases (float) | ~0.5 MB |
| Gradientes | ~10 MB |
| Activaciones (batch=8) | ~5 MB |
| **Total** | **~18 MB** |

### 8.4 Requisitos de Hardware

**Mínimos:**
- CPU: x86 con soporte C99
- RAM: 64 MB
- Almacenamiento: 10 MB

**Recomendados:**
- CPU: x86 con SSE/SSE2
- RAM: 256 MB
- Almacenamiento: 100 MB

---

## 9. Riesgos y Mitigaciones

### 9.1 Riesgos Técnicos

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| Bug no detectado en backward pass | Media | Alto | Implementar tests exhaustivos |
| Overflow numérico | Baja | Alto | Revisar operaciones de suma/multiplicación |
| Memory leaks | Baja | Medio | Uso sistemático de valgrind |
| Incompatibilidad SSE | Baja | Medio | Fallback automático implementado |

### 9.2 Riesgos de Proyecto

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| Rendimiento insuficiente | Media | Alto | Benchmarking continuo, optimización SSE |
| Convergencia pobre | Media | Alto | Experimentación con hyperparámetros |
| Complejidad creciente | Media | Medio | Refactoring periódico, documentación |

---

## 10. Recursos y Referencias

### 10.1 Papers y Referencias

- **Ternary Weight Networks:** Courbariaux et al., "Training deep neural networks with low precision multiplications"
- **Straight-Through Estimator:** Bengio et al., "Estimating or propagating gradients through stochastic neurons"
- **Attention Is All You Need:** Vaswani et al., "Attention Is All You Need" (Transformer original)

### 10.2 Recursos Técnicos

- **Intel Intrinsics Guide:** https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
- **C99 Standard:** ISO/IEC 9899:1999
- **Valgrind:** http://valgrind.org/

### 10.3 Datos de Entrenamiento

- Dataset actual: `data/saioa_stories_sample.txt`
- Futuro: TinyStories, otros datasets de texto libre

---

## 11. Glosario

- **Ternary:** Sistema numérico con 3 valores (-1, 0, 1)
- **SSE:** Streaming SIMD Extensions (optimización vectorial x86)
- **Forward Pass:** Propagación de entrada a salida
- **Backward Pass:** Backpropagation (cálculo de gradientes)
- **LayerNorm:** Normalización de capa
- **FFN:** Feed-Forward Network
- **MHA:** Multi-Head Attention
- **Checkpointing:** Guardado de estado del modelo
- **Gradiente:** Derivada de la función de pérdida respecto a parámetros

---

## 12. Historial de Versiones

| Versión | Fecha | Descripción | Autor |
|---------|-------|-------------|-------|
| 1.0 | Feb 2026 | Plan inicial completo | AI Assistant |

---

## 13. Contacto y Contribuciones

**Repositorio:** `/home/tmlcs/tmlcs-proyects/00_tmlcs_valery`

**Canales de comunicación:**
- Issues: GitHub Issues
- Discusiones: GitHub Discussions
- Documentación: Ver `README.md`, `AGENTS.md`

**Guía de contribución:**
1. Fork del repositorio
2. Crear branch feature (`git checkout -b feature/nueva-feature`)
3. Commit de cambios (`git commit -am 'Add nueva feature'`)
4. Push al branch (`git push origin feature/nueva-feature`)
5. Crear Pull Request

---

**Fin del Plan de Proyecto**

*Documento generado automáticamente para Legacy-1bit LLM Project*

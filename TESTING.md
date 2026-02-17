# Guía de Pruebas del Proyecto Legacy-1bit LLM

Este documento describe cómo ejecutar las pruebas y análisis de calidad implementados en el proyecto.

## 1. Pruebas Unitarias Generales (`make test`)

Este objetivo ejecuta todas las pruebas unitarias existentes (ej. `test_llm`, `test_math_ops`, etc.) en versiones no-SSE y SSE.

**Comando:**
```bash
make test
```
**Salida esperada:**
Verá la compilación de `test_runner_no_sse` y `test_runner_sse`, seguido de la ejecución de todas las pruebas.

## 2. Prueba de Correctitud de Optimizaciones SSE (`make test_sse_correctness`)

Esta prueba compara la salida de funciones matemáticas críticas entre sus implementaciones SSE y no-SSE para asegurar que producen resultados idénticos (dentro de una pequeña tolerancia de punto flotante).

**Comando:**
```bash
make test_sse_correctness
```
**Detalles:**
Este comando compila dos ejecutables (`test_sse_correctness_no_sse` y `test_sse_correctness_sse`), los ejecuta y compara sus salidas mediante el script `test_sse_comparison.sh`.

**Salida esperada:**
```
--- Running SSE Correctness Test ---
--- Building Non-SSE version ---
...
--- Building SSE version ---
...
--- Running Non-SSE tests and capturing output ---
--- Running SSE tests and capturing output ---
--- Comparing outputs ---
SSE Correctness Test PASSED!
```
Si hay alguna discrepancia, el script reportará los fallos.

## 3. Prueba de Fugas de Memoria con Valgrind (`make test_memory`)

Esta prueba ejecuta un pequeño ciclo de entrenamiento bajo [Valgrind](https://valgrind.org/) (herramienta `memcheck`) para detectar posibles fugas de memoria o accesos inválidos.

**Requisitos:**
Debe tener Valgrind instalado en su sistema.
```bash
sudo apt-get install valgrind # En Debian/Ubuntu
# o
sudo dnf install valgrind # En Fedora/RHEL/CentOS
```

**Comando:**
```bash
make test_memory
valgrind --leak-check=full --show-leak-kinds=all ./test_memory data/saioa_stories_sample.txt
```
**Salida esperada:**
Al final de la salida de Valgrind, debería ver:
```
==YOUR_PID== HEAP SUMMARY:
==YOUR_PID==     in use at exit: 0 bytes in 0 blocks
...
==YOUR_PID== All heap blocks were freed -- no leaks are possible
...
==YOUR_PID== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```
Si Valgrind reporta fugas (`definitely lost`, `possibly lost`) o errores de memoria (`Invalid read/write`), el test ha fallado.

## 4. Prueba de Verificación de Gradientes (`make test_grad_check`)

Esta prueba verifica la correctitud de la implementación del `backward pass` para los parámetros continuos (sesgos y parámetros de normalización de capa) mediante la técnica de "Gradient Checking".

**Comando:**
```bash
make test_grad_check
```
**Salida esperada:**
Si la prueba pasa, al final verá:
```
--------------------------------
Gradient Check PASSED!
--------------------------------
```
Si falla, se listarán las discrepancias entre los gradientes analíticos y numéricos.

## 5. Análisis de Rendimiento (`make perf`)

Este objetivo compila y ejecuta las versiones SSE y no-SSE del `legacy_llm` (el programa de entrenamiento principal) y luego utiliza el script `analyze_perf.sh` para comparar sus tiempos de ejecución y calcular el "speedup" de las optimizaciones SSE.

**Comando:**
```bash
make perf
```
**Salida esperada:**
```
--- Non-SSE Build Performance Summary ---
...
Overall Total Execution Time: XXXX ms

--- SSE Build Performance Summary ---
...
Overall Total Execution Time: YYYY ms

--- SSE Performance Comparison ---
Non-SSE Total Time: XXXX ms
SSE Total Time:     YYYY ms
SSE Speedup:        Z.ZZZx

SSE is faster than Non-SSE!
```
Donde `Z.ZZZx` es el factor de aceleración.
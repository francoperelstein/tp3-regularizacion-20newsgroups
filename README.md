# TP3 - Regularización en TNG: L1/L2, Dropout y BatchNorm sobre 20 Newsgroups

Trabajo práctico de **Redes Neuronales en Bioingeniería** (ITBA). Continúa el [TP1](https://github.com/francoperelstein/tp-softmax-20newsgroups) (regresión softmax) y el [TP2](https://github.com/francoperelstein/tp2-mlp-20newsgroups) (MLP): se agregan como hiperparámetros la regularización L1, L2 y L1+L2, Dropout y BatchNormalization, y se analizan los valores de los pesos, los tiempos de entrenamiento, el learning rate y el ruido de las curvas, contrastando con la clase de Regularización.

## Cómo correrlo
1. Abrir `TP3_Regularizacion_20Newsgroups.ipynb` en Google Colab con GPU.
2. Instalar dependencias con `requirements.txt` (la primera celda del notebook instala `mlflow` y `optuna`).
3. Ejecutar todas las celdas en orden. La corrida completa tarda aproximadamente 1 hora y 20 minutos: la búsqueda de 150 trials tarda unos 26 minutos y los experimentos controlados, el resto.

## Contenido del notebook
Setup, datos, preprocesamiento y TF-IDF (del TP2) · MLP con regularización · entrenamiento · búsqueda con Optuna (un estudio por optimizador, con pruning) · experimentos controlados con 5 semillas · valores absolutos de pesos · tiempos, learning rate y ruido · modelo final · evaluación final · conclusiones y limitaciones.

## Resultados principales
- **Búsqueda:** 150 trials (39 podados). Mejor val_acc 0.7508 con Adam, 1 capa de 128 neuronas y Dropout 0.5; la búsqueda no eligió penalización L1/L2 (λ ≈ 0).
- **Experimentos controlados:** con early stopping, la referencia sin regularizar, Dropout y Dropout + max-norm alcanzan el mismo máximo de val_acc (≈0.749). Dropout + max-norm es el más robusto a entrenar de más (≈0.743 a la época 40 contra ≈0.705).
- **L1/L2:** L2 achica los pesos y L1 los dispersa (84% a 97% de los pesos por debajo de 0.01), pero con λ ≥ 1e-5 bajan el accuracy.
- **BatchNorm y learning rate altos:** aceleran la convergencia pero sobreajustan más rápido.
- **Modelo final (test, 5 semillas):** 0.6857 ± 0.0013 (val 0.7457 ± 0.0011), con Dropout 0.5 sobre la base y 6 épocas. Como referencia, el MLP del TP2 dio 0.6808 y el softmax del TP1 0.6569 (aunque se utilizaron protocolos distintos).

## Reproducibilidad
Semillas fijas en la búsqueda y en los experimentos. El test se usa únicamente en la evaluación final. No se incluyen `mlruns`, bases de Optuna ni modelos entrenados (ver `.gitignore`).

## Uso de IA
Usé Claude (Anthropic) como asistente para diseñar los experimentos, escribir y depurar código y organizar el análisis.

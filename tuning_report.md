# StreamLoop — Ajuste de hiperparametros del modelo Churn

## Objetivo

Optimizar la deteccion de clientes que cancelaran su
suscripcion mediante RandomizedSearchCV y GridSearchCV,
comparando los resultados frente a un Random Forest
baseline.

## Dataset y preparacion

- Fuente: IBM Telco Customer Churn.
- Observaciones: 7.043 clientes.
- Objetivo: `Churn`, convertido a No=0 y Yes=1.
- Variables predictoras: 19.
- `customerID` excluido del entrenamiento.
- `TotalCharges`: conversion numerica, 11 valores ausentes.
- Division estratificada: 5.634 train / 1.409 test.
- Semilla: 42.

La imputacion y OneHotEncoder estan dentro de
ColumnTransformer + Pipeline, ajustandose solo
con los datos disponibles en cada entrenamiento.

## Metrica principal y justificacion

**F2-score**, por dar mas peso al recall que a
precision. Detectar clientes en riesgo permite
intervenir antes de que cancelen, aunque genera
costos por contactos innecesarios.

Se reportan tambien precision, recall, accuracy y
average precision para transparentar el compromiso.

## Baseline

RandomForestClassifier con hiperparametros
predeterminados, excepto semilla y paralelismo.

- F2 medio CV: 0.502486
- Desviacion F2 CV: 0.029462

## RandomizedSearchCV

- 16 candidatos.
- 5 folds StratifiedKFold.
- 80 ajustes CV.
- F2 medio ganador: 0.722519
- Desviacion CV: 0.014097

## GridSearchCV

- 24 candidatos.
- 5 folds StratifiedKFold.
- 120 ajustes CV.
- F2 medio ganador: 0.722519
- Desviacion CV: 0.014097

GridSearchCV igualo el mejor F2 de RandomizedSearchCV,
sin una mejora adicional de rendimiento.

## Hiperparametros seleccionados

- `classifier__class_weight`: `balanced`
- `classifier__max_depth`: `6`
- `classifier__max_features`: `sqrt`
- `classifier__min_samples_leaf`: `8`
- `classifier__min_samples_split`: `8`
- `classifier__n_estimators`: `80`

El modelo se selecciono mediante validacion cruzada
sobre entrenamiento, antes de evaluar el test final.

## Evaluacion final en test

| Metrica | Baseline | Optimizado |
|---|---:|---:|
| Accuracy | 0.7779 | 0.7445 |
| Precision | 0.6034 | 0.5121 |
| Recall | 0.4759 | 0.7941 |
| F2-score | 0.4969 | 0.7153 |
| Average precision | 0.6057 | 0.6469 |
| Verdaderos positivos | 178 | 297 |
| Falsos negativos | 196 | 77 |
| Falsos positivos | 117 | 283 |
| Verdaderos negativos | 918 | 752 |

Mejora absoluta F2: 0.2184.
Mejora relativa F2: 43.95%.

El modelo identifica
119 clientes
adicionales que efectivamente cancelan, reduce
los falsos negativos en
119 y genera
166
falsos positivos adicionales.

## Seleccion y estabilidad

La desviacion CV desciende desde
0.029462 en baseline a
0.014097 en el modelo
seleccionado.

El modelo final se elige por F2 medio y estabilidad.
El conjunto de prueba no se utilizo para seleccionar
hiperparametros.

## Limitaciones y decision de negocio

La mejora de recall tiene como contrapartida menor
precision y accuracy. Antes de implementar acciones
de retencion masivas, StreamLoop debe estimar el
costo de contactar falsos positivos y el valor
economico de evitar cancelaciones.

No se optimizo el umbral de clasificacion usando
el conjunto de prueba. Un futuro experimento de
umbrales debe utilizar exclusivamente validacion
sobre entrenamiento.

No se afirma causalidad entre variables y churn.
El resultado corresponde a este dataset y split;
debe confirmarse con datos operacionales nuevos.

## Politica del conjunto de prueba

Se registro una evaluacion para el baseline y una
para el modelo optimizado. Los experimentos de
RandomizedSearchCV y GridSearchCV no usaron test.

El notebook de entrega lee resultados ya guardados
y no vuelve a evaluar el conjunto de prueba.

## Archivos relevantes

- `notebooks/streamloop_churn_tuning.ipynb`
- `scripts/churn_baseline.py`
- `scripts/churn_random_search.py`
- `scripts/churn_grid_search.py`
- `scripts/churn_final_evaluation.py`
- `reports/`: metricas y tablas de ambas busquedas.
- `models/`: modelos baseline y optimizados.

## Reproducibilidad

Desde la raiz del repositorio:

```bash
uv sync --python 3.12
uv run --frozen python -m pytest tests/ -q
```

No volver a ejecutar el script de evaluacion final:
el resultado del holdout ya esta guardado.

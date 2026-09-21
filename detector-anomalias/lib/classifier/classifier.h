#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <stddef.h>

// Número de features esperado pelo modelo treinado.
#define CLASSIFIER_N_FEATURES 8

// z-score: (bruto - media) / escala. Escala zero é tratada como 1.
void standardize(const float* raw, const float* mean, const float* scale,
                 float* out, size_t n);

// Regressão logística: sigmoid(pesos . features + bias).
float classifier_score(const float* features, const float* weights, float bias,
                       size_t n);

bool classifier_is_anomaly(float score, float threshold);

#endif

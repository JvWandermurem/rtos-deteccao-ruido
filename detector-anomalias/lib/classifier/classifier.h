#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <stddef.h>

// Número de features esperado pelo modelo treinado.
#define CLASSIFIER_N_FEATURES 8

bool classifier_is_anomaly(float score, float threshold);

#endif

#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

#include <stddef.h>

// Raiz quadrada da média dos quadrados do bloco. Retorna 0 se n == 0.
float compute_rms(const float* samples, size_t n);

#endif

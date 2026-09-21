#include "audio_features.h"
#include <math.h>

float compute_rms(const float* samples, size_t n) {
  if (n == 0) return 0.0f;
  float soma = 0.0f;
  for (size_t i = 0; i < n; i++) {
    soma += samples[i] * samples[i];
  }
  return sqrtf(soma / (float)n);
}

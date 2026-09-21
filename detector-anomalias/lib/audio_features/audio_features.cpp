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

float compute_spectral_centroid(const float* magnitudes, size_t n_bins,
                                float bin_hz) {
  float soma_pesos = 0.0f;
  float soma_ponderada = 0.0f;
  for (size_t k = 0; k < n_bins; k++) {
    soma_pesos += magnitudes[k];
    soma_ponderada += magnitudes[k] * (bin_hz * (float)k);
  }
  if (soma_pesos <= 0.0f) return 0.0f;
  return soma_ponderada / soma_pesos;
}

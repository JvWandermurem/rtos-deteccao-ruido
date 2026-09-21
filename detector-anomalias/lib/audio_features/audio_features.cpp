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

float hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void compute_mel_energies(const float* magnitudes, size_t n_bins, float bin_hz,
                          float* out_energies, size_t n_filters) {
  for (size_t f = 0; f < n_filters; f++) out_energies[f] = 0.0f;
  if (n_filters == 0 || n_bins < 2) return;

  const float freq_max = bin_hz * (float)(n_bins - 1);
  const float mel_max = hz_to_mel(freq_max);
  const float mel_passo = mel_max / (float)(n_filters + 1);

  for (size_t f = 0; f < n_filters; f++) {
    const float esquerda = mel_to_hz(mel_passo * (float)f);
    const float centro = mel_to_hz(mel_passo * (float)(f + 1));
    const float direita = mel_to_hz(mel_passo * (float)(f + 2));

    float soma = 0.0f;
    for (size_t k = 0; k < n_bins; k++) {
      const float freq = bin_hz * (float)k;
      if (freq <= esquerda || freq >= direita) continue;
      const float peso = (freq <= centro)
                             ? (freq - esquerda) / (centro - esquerda)
                             : (direita - freq) / (direita - centro);
      soma += magnitudes[k] * peso;
    }
    out_energies[f] = soma;
  }
}

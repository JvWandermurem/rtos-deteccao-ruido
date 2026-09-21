#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

#include <stddef.h>

// Raiz quadrada da média dos quadrados do bloco. Retorna 0 se n == 0.
float compute_rms(const float* samples, size_t n);

// Centro de massa do espectro, em Hz. bin_hz = sample_rate / fft_size.
// Retorna 0 se a soma das magnitudes for zero.
float compute_spectral_centroid(const float* magnitudes, size_t n_bins,
                                float bin_hz);

// Número máximo de filtros mel suportado pelos buffers internos.
#define MFCC_MAX_FILTROS 32

float hz_to_mel(float hz);
float mel_to_hz(float mel);

// Energia de cada filtro triangular mel, distribuídos entre 0 Hz e
// bin_hz * (n_bins - 1). out_energies precisa ter n_filters posições.
void compute_mel_energies(const float* magnitudes, size_t n_bins, float bin_hz,
                          float* out_energies, size_t n_filters);

#endif

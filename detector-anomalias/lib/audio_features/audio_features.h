#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

#include <stddef.h>

// Raiz quadrada da média dos quadrados do bloco. Retorna 0 se n == 0.
float compute_rms(const float* samples, size_t n);

// Centro de massa do espectro, em Hz. bin_hz = sample_rate / fft_size.
// Retorna 0 se a soma das magnitudes for zero.
float compute_spectral_centroid(const float* magnitudes, size_t n_bins,
                                float bin_hz);

#endif

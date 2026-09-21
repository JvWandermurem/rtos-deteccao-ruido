#include <unity.h>
#include "audio_features.h"

void setUp(void) {}
void tearDown(void) {}

void test_rms_de_sinal_constante(void) {
  const float amostras[4] = {3.0f, 3.0f, 3.0f, 3.0f};
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 3.0f, compute_rms(amostras, 4));
}

void test_rms_de_onda_quadrada(void) {
  const float amostras[4] = {1.0f, -1.0f, 1.0f, -1.0f};
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, compute_rms(amostras, 4));
}

void test_rms_de_bloco_vazio_e_zero(void) {
  const float amostras[1] = {5.0f};
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, compute_rms(amostras, 0));
}

void test_centroide_de_tom_puro(void) {
  const float magnitudes[4] = {0.0f, 1.0f, 0.0f, 0.0f};
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, 100.0f,
                           compute_spectral_centroid(magnitudes, 4, 100.0f));
}

void test_centroide_e_media_ponderada(void) {
  const float magnitudes[4] = {0.0f, 1.0f, 1.0f, 0.0f};
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, 150.0f,
                           compute_spectral_centroid(magnitudes, 4, 100.0f));
}

void test_centroide_de_espectro_silencioso_e_zero(void) {
  const float magnitudes[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f,
                           compute_spectral_centroid(magnitudes, 4, 100.0f));
}

void test_hz_to_mel_em_valor_conhecido(void) {
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, hz_to_mel(0.0f));
  TEST_ASSERT_FLOAT_WITHIN(1.0f, 1000.3f, hz_to_mel(1000.0f));
}

void test_mel_ida_e_volta(void) {
  TEST_ASSERT_FLOAT_WITHIN(0.1f, 3000.0f, mel_to_hz(hz_to_mel(3000.0f)));
}

void test_energias_mel_de_espectro_silencioso_sao_zero(void) {
  float magnitudes[257];
  for (size_t k = 0; k < 257; k++) magnitudes[k] = 0.0f;
  float energias[8];
  compute_mel_energies(magnitudes, 257, 31.25f, energias, 8);
  for (size_t f = 0; f < 8; f++) {
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, energias[f]);
  }
}

// Um tom grave deve ativar um filtro mais à esquerda que um tom agudo.
void test_energias_mel_respeitam_ordem_de_frequencia(void) {
  float magnitudes[257];
  float energias[8];

  for (size_t k = 0; k < 257; k++) magnitudes[k] = 0.0f;
  magnitudes[16] = 1.0f;  // 500 Hz com bin_hz = 31.25
  compute_mel_energies(magnitudes, 257, 31.25f, energias, 8);
  size_t argmax_grave = 0;
  for (size_t f = 1; f < 8; f++) {
    if (energias[f] > energias[argmax_grave]) argmax_grave = f;
  }

  for (size_t k = 0; k < 257; k++) magnitudes[k] = 0.0f;
  magnitudes[128] = 1.0f;  // 4000 Hz
  compute_mel_energies(magnitudes, 257, 31.25f, energias, 8);
  size_t argmax_agudo = 0;
  for (size_t f = 1; f < 8; f++) {
    if (energias[f] > energias[argmax_agudo]) argmax_agudo = f;
  }

  TEST_ASSERT_TRUE(argmax_grave < argmax_agudo);
}

int main(int argc, char** argv) {
  UNITY_BEGIN();
  RUN_TEST(test_rms_de_sinal_constante);
  RUN_TEST(test_rms_de_onda_quadrada);
  RUN_TEST(test_rms_de_bloco_vazio_e_zero);
  RUN_TEST(test_centroide_de_tom_puro);
  RUN_TEST(test_centroide_e_media_ponderada);
  RUN_TEST(test_centroide_de_espectro_silencioso_e_zero);
  RUN_TEST(test_hz_to_mel_em_valor_conhecido);
  RUN_TEST(test_mel_ida_e_volta);
  RUN_TEST(test_energias_mel_de_espectro_silencioso_sao_zero);
  RUN_TEST(test_energias_mel_respeitam_ordem_de_frequencia);
  return UNITY_END();
}

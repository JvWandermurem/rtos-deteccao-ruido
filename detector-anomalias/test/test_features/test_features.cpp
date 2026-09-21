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

int main(int argc, char** argv) {
  UNITY_BEGIN();
  RUN_TEST(test_rms_de_sinal_constante);
  RUN_TEST(test_rms_de_onda_quadrada);
  RUN_TEST(test_rms_de_bloco_vazio_e_zero);
  RUN_TEST(test_centroide_de_tom_puro);
  RUN_TEST(test_centroide_e_media_ponderada);
  RUN_TEST(test_centroide_de_espectro_silencioso_e_zero);
  return UNITY_END();
}

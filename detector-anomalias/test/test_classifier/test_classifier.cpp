#include <unity.h>
#include "classifier.h"

void setUp(void) {}
void tearDown(void) {}

void test_padroniza_subtraindo_media_e_dividindo_escala(void) {
  const float bruto[2] = {2.0f, 4.0f};
  const float media[2] = {1.0f, 2.0f};
  const float escala[2] = {1.0f, 2.0f};
  float saida[2];
  standardize(bruto, media, escala, saida, 2);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, saida[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, saida[1]);
}

// Escala zero apareceria se uma feature fosse constante no treino.
void test_padroniza_trata_escala_zero_como_um(void) {
  const float bruto[1] = {5.0f};
  const float media[1] = {2.0f};
  const float escala[1] = {0.0f};
  float saida[1];
  standardize(bruto, media, escala, saida, 1);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 3.0f, saida[0]);
}

void test_score_neutro_e_meio(void) {
  const float features[2] = {10.0f, -7.0f};
  const float pesos[2] = {0.0f, 0.0f};
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.5f,
                           classifier_score(features, pesos, 0.0f, 2));
}

void test_score_saturado_tende_a_um(void) {
  const float features[2] = {10.0f, 10.0f};
  const float pesos[2] = {1.0f, 1.0f};
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f,
                           classifier_score(features, pesos, 0.0f, 2));
}

void test_threshold_decide_anomalia(void) {
  TEST_ASSERT_TRUE(classifier_is_anomaly(0.7f, 0.5f));
  TEST_ASSERT_FALSE(classifier_is_anomaly(0.3f, 0.5f));
}

int main(int argc, char** argv) {
  UNITY_BEGIN();
  RUN_TEST(test_padroniza_subtraindo_media_e_dividindo_escala);
  RUN_TEST(test_padroniza_trata_escala_zero_como_um);
  RUN_TEST(test_score_neutro_e_meio);
  RUN_TEST(test_score_saturado_tende_a_um);
  RUN_TEST(test_threshold_decide_anomalia);
  return UNITY_END();
}

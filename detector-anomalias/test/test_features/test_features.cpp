#include <unity.h>
#include <math.h>
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

// DCT-II de um sinal constante concentra tudo no coeficiente 0.
void test_dct2_de_entrada_constante(void) {
  const float entrada[4] = {2.0f, 2.0f, 2.0f, 2.0f};
  float saida[4];
  dct2(entrada, 4, saida, 4);
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, 8.0f, saida[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, 0.0f, saida[1]);
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, 0.0f, saida[2]);
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, 0.0f, saida[3]);
}

// Mesmo com espectro zerado, o log tem piso: nada de -inf ou NaN.
void test_mfcc_de_espectro_silencioso_e_finito(void) {
  float magnitudes[257];
  for (size_t k = 0; k < 257; k++) magnitudes[k] = 0.0f;
  float coeficientes[6];
  compute_mfcc(magnitudes, 257, 31.25f, coeficientes, 6, 8);
  for (size_t c = 0; c < 6; c++) {
    TEST_ASSERT_TRUE(isfinite(coeficientes[c]));
  }
}

// O coeficiente 0 é proporcional à energia total em escala log.
void test_mfcc_primeiro_coeficiente_cresce_com_energia(void) {
  float magnitudes[257];
  float baixo[6];
  float alto[6];

  for (size_t k = 0; k < 257; k++) magnitudes[k] = 0.01f;
  compute_mfcc(magnitudes, 257, 31.25f, baixo, 6, 8);

  for (size_t k = 0; k < 257; k++) magnitudes[k] = 10.0f;
  compute_mfcc(magnitudes, 257, 31.25f, alto, 6, 8);

  TEST_ASSERT_TRUE(alto[0] > baixo[0]);
}

void test_remove_dc_centraliza_o_bloco(void) {
  float amostras[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  remove_dc(amostras, 4);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, -1.5f, amostras[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.5f, amostras[1]);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.5f, amostras[2]);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.5f, amostras[3]);
}

void test_remove_dc_preserva_sinal_ja_centrado(void) {
  float amostras[2] = {-1.0f, 1.0f};
  remove_dc(amostras, 2);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, -1.0f, amostras[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, amostras[1]);
}

void test_zero_low_bins_zera_abaixo_do_piso(void) {
  // bin_hz = 100 => bins em 0, 100, 200, 300 Hz. Piso de 250 Hz.
  float magnitudes[4] = {9.0f, 9.0f, 9.0f, 9.0f};
  zero_low_bins(magnitudes, 4, 100.0f, 250.0f);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, magnitudes[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, magnitudes[1]);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, magnitudes[2]);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 9.0f, magnitudes[3]);
}

void test_zero_low_bins_com_piso_zero_nao_altera_nada(void) {
  float magnitudes[3] = {1.0f, 2.0f, 3.0f};
  zero_low_bins(magnitudes, 3, 100.0f, 0.0f);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, magnitudes[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 2.0f, magnitudes[1]);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 3.0f, magnitudes[2]);
}

// O centroide precisa subir quando o rumble de baixa frequência é removido.
void test_piso_de_frequencia_levanta_o_centroide(void) {
  // Rumble enorme em 100 Hz, tom pequeno em 2000 Hz.
  float magnitudes[41];
  for (size_t k = 0; k < 41; k++) magnitudes[k] = 0.0f;
  magnitudes[2] = 100.0f;   // 100 Hz com bin_hz = 50
  magnitudes[40] = 5.0f;    // 2000 Hz
  const float antes = compute_spectral_centroid(magnitudes, 41, 50.0f);
  zero_low_bins(magnitudes, 41, 50.0f, 400.0f);
  const float depois = compute_spectral_centroid(magnitudes, 41, 50.0f);
  TEST_ASSERT_TRUE(antes < 300.0f);
  TEST_ASSERT_TRUE(depois > 1900.0f);
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
  RUN_TEST(test_dct2_de_entrada_constante);
  RUN_TEST(test_mfcc_de_espectro_silencioso_e_finito);
  RUN_TEST(test_mfcc_primeiro_coeficiente_cresce_com_energia);
  RUN_TEST(test_remove_dc_centraliza_o_bloco);
  RUN_TEST(test_remove_dc_preserva_sinal_ja_centrado);
  RUN_TEST(test_zero_low_bins_zera_abaixo_do_piso);
  RUN_TEST(test_zero_low_bins_com_piso_zero_nao_altera_nada);
  RUN_TEST(test_piso_de_frequencia_levanta_o_centroide);
  return UNITY_END();
}

#include <Arduino.h>
#include <driver/i2s.h>
#include <esp_timer.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#include "audio_features.h"
#include <arduinoFFT.h>

#define I2S_WS 25
#define I2S_SD 33
#define I2S_SCK 26
#define I2S_PORT I2S_NUM_0
// L/R do INMP441 fica sem conexão: ligá-lo ao GND travou a leitura
// neste módulo específico.

#define LED_VERMELHO 4
#define LED_VERDE 18

#define TAXA_AMOSTRAGEM 16000
#define BLOCO_AMOSTRAS 512
#define RING_BLOCOS 4
#define RING_AMOSTRAS (BLOCO_AMOSTRAS * RING_BLOCOS)
#define FILA_CAPACIDADE 8

LiquidCrystal_I2C lcd(0x27, 16, 2);

#define BINS_ESPECTRO (BLOCO_AMOSTRAS / 2 + 1)
#define BIN_HZ ((float)TAXA_AMOSTRAGEM / (float)BLOCO_AMOSTRAS)
#define MFCC_FILTROS 8
#define MFCC_COEFICIENTES 6
#define PISO_FREQUENCIA_HZ 200.0f

static float fft_real[BLOCO_AMOSTRAS];
static float fft_imag[BLOCO_AMOSTRAS];
static ArduinoFFT<float> fft(fft_real, fft_imag, BLOCO_AMOSTRAS,
                             (float)TAXA_AMOSTRAGEM);

struct VetorFeatures {
  float valores[8];
  int64_t ts_captura;
  int64_t ts_features;
};

static float ring_buffer[RING_AMOSTRAS];
static size_t escrita_idx = 0;
static int64_t ts_ultimo_bloco = 0;

static SemaphoreHandle_t mutex_buffer;
static SemaphoreHandle_t sem_bloco_pronto;
static QueueHandle_t fila_features;
static volatile uint32_t blocos_perdidos = 0;
static volatile uint32_t sinais_perdidos = 0;
static volatile uint32_t amostras_capturadas = 0;

static void configurarI2S() {
  i2s_config_t config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      // Medido com um contador de amostras (amostras_capturadas): com
      // sample_rate = TAXA_AMOSTRAGEM o I2S entrega ~16.000 amostras/s,
      // exatamente a taxa configurada — não há halving de canal ONLY_LEFT
      // aqui. A taxa real de LATENCY/s mais baixa que se observa (~15,7/s
      // em vez de ~31,3/s) vem do lado do consumidor: taskCaptura roda em
      // prioridade mais alta que taskFeatures no mesmo core, o DMA guarda
      // várias amostras em buffer, e i2s_read() drena esse backlog sem
      // bloquear várias vezes seguidas antes de taskFeatures conseguir
      // rodar — cada give() extra do semáforo binário nesse meio tempo é
      // descartado (ver sinais_perdidos). Não é um problema de hardware;
      // não dobre este valor.
      .sample_rate = TAXA_AMOSTRAGEM,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 4,
      .dma_buf_len = BLOCO_AMOSTRAS,
  };
  i2s_pin_config_t pinos = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_SD,
  };
  i2s_driver_install(I2S_PORT, &config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pinos);
}

// Prioridade alta: lê o I2S sem parar e alimenta o ring buffer.
static void taskCaptura(void* parametro) {
  static int32_t brutas[BLOCO_AMOSTRAS];
  for (;;) {
    size_t bytes_lidos = 0;
    i2s_read(I2S_PORT, brutas, sizeof(brutas), &bytes_lidos, portMAX_DELAY);
    const size_t n = bytes_lidos / sizeof(int32_t);
    const int64_t agora = esp_timer_get_time();
    amostras_capturadas += n;

    xSemaphoreTake(mutex_buffer, portMAX_DELAY);
    for (size_t i = 0; i < n; i++) {
      // INMP441 entrega 24 bits úteis alinhados à esquerda no word de 32.
      ring_buffer[escrita_idx] = (float)(brutas[i] >> 8);
      escrita_idx = (escrita_idx + 1) % RING_AMOSTRAS;
    }
    ts_ultimo_bloco = agora;
    xSemaphoreGive(mutex_buffer);

    // Semáforo binário: se a Task 2 ainda não consumiu o sinal anterior,
    // este give é descartado. É o comportamento desejado — a Task 2 sempre
    // processa o bloco mais recente, e a captura nunca bloqueia.
    if (xSemaphoreGive(sem_bloco_pronto) != pdTRUE) {
      sinais_perdidos++;
    }
  }
}

// Prioridade média: copia o bloco mais recente e extrai features.
static void taskFeatures(void* parametro) {
  static float trabalho[BLOCO_AMOSTRAS];
  for (;;) {
    xSemaphoreTake(sem_bloco_pronto, portMAX_DELAY);

    VetorFeatures vetor;
    xSemaphoreTake(mutex_buffer, portMAX_DELAY);
    const size_t inicio =
        (escrita_idx + RING_AMOSTRAS - BLOCO_AMOSTRAS) % RING_AMOSTRAS;
    for (size_t i = 0; i < BLOCO_AMOSTRAS; i++) {
      trabalho[i] = ring_buffer[(inicio + i) % RING_AMOSTRAS];
    }
    vetor.ts_captura = ts_ultimo_bloco;
    xSemaphoreGive(mutex_buffer);

    remove_dc(trabalho, BLOCO_AMOSTRAS);

    for (size_t i = 0; i < BLOCO_AMOSTRAS; i++) {
      fft_real[i] = trabalho[i];
      fft_imag[i] = 0.0f;
    }
    fft.windowing(FFTWindow::Hamming, FFTDirection::Forward);
    fft.compute(FFTDirection::Forward);
    fft.complexToMagnitude();

    zero_low_bins(fft_real, BINS_ESPECTRO, BIN_HZ, PISO_FREQUENCIA_HZ);

    vetor.valores[0] = compute_rms(trabalho, BLOCO_AMOSTRAS);
    vetor.valores[1] =
        compute_spectral_centroid(fft_real, BINS_ESPECTRO, BIN_HZ);
    compute_mfcc(fft_real, BINS_ESPECTRO, BIN_HZ, &vetor.valores[2],
                 MFCC_COEFICIENTES, MFCC_FILTROS);
    vetor.ts_features = esp_timer_get_time();

    if (xQueueSend(fila_features, &vetor, 0) != pdTRUE) {
      blocos_perdidos++;
    }
  }
}

// Prioridade baixa: consome a fila e decide.
static void taskDeteccao(void* parametro) {
  VetorFeatures vetor;
  for (;;) {
    if (xQueueReceive(fila_features, &vetor, pdMS_TO_TICKS(100)) != pdTRUE) {
      continue;
    }
    const int64_t ts_deteccao = esp_timer_get_time();
    Serial.printf("FEAT %.2f %.2f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                  vetor.valores[0], vetor.valores[1], vetor.valores[2],
                  vetor.valores[3], vetor.valores[4], vetor.valores[5],
                  vetor.valores[6], vetor.valores[7]);
    Serial.printf("LATENCY cap=%lld feat=%lld det=%lld total=%lld label=%s\n",
                  (long long)vetor.ts_captura,
                  (long long)(vetor.ts_features - vetor.ts_captura),
                  (long long)(ts_deteccao - vetor.ts_features),
                  (long long)(ts_deteccao - vetor.ts_captura), "NADA");
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(LED_VERMELHO, OUTPUT);
  pinMode(LED_VERDE, OUTPUT);
  digitalWrite(LED_VERMELHO, LOW);
  digitalWrite(LED_VERDE, HIGH);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Detector Ruido");
  lcd.setCursor(0, 1);
  lcd.print("Ativo           ");

  configurarI2S();

  mutex_buffer = xSemaphoreCreateMutex();
  sem_bloco_pronto = xSemaphoreCreateBinary();
  fila_features = xQueueCreate(FILA_CAPACIDADE, sizeof(VetorFeatures));

  xTaskCreatePinnedToCore(taskCaptura, "captura", 4096, NULL, 5, NULL, 1);
  xTaskCreatePinnedToCore(taskFeatures, "features", 8192, NULL, 3, NULL, 1);
  xTaskCreatePinnedToCore(taskDeteccao, "deteccao", 4096, NULL, 1, NULL, 0);
}

void loop() {
  // Métrica de saturação do pipeline: em operação normal fica em zero.
  Serial.printf("DROPS %u SINAIS_PERDIDOS %u AMOSTRAS %u\n",
                (unsigned)blocos_perdidos, (unsigned)sinais_perdidos,
                (unsigned)amostras_capturadas);
  vTaskDelay(pdMS_TO_TICKS(5000));
}

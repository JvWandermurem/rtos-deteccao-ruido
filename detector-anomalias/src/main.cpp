#include <Arduino.h>
#include <driver/i2s.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define I2S_WS   25
#define I2S_SD   33
#define I2S_SCK  26
#define I2S_PORT I2S_NUM_0
// L/R do INMP441: deixe sem conexão (flutuando). Neste módulo, ligar ao GND
// causou leitura travada em 0/1 — sem conexão, o canal esquerdo funciona bem.

#define LED_VERMELHO 4
#define LED_VERDE    18
#define BOTAO_DETECCAO 23

// LCD 16x2 I2C: SDA = GPIO21, SCL = GPIO22 (pinos padrão do ESP32)
// Endereço 0x27 é o mais comum nos backpacks PCF8574; se não aparecer nada
// na tela, rode um scanner I2C pra confirmar (0x3F é o outro endereço comum).
LiquidCrystal_I2C lcd(0x27, 16, 2);

bool deteccaoAtiva = true;
int estadoBotaoAnterior = HIGH;
unsigned long ultimoDebounce = 0;
const unsigned long DEBOUNCE_MS = 50;

void atualizarLcdStatus() {
  lcd.setCursor(0, 1);
  lcd.print(deteccaoAtiva ? "Ativo          " : "Pausado         ");
}

void setup() {
  Serial.begin(115200);

  pinMode(LED_VERMELHO, OUTPUT);
  pinMode(LED_VERDE, OUTPUT);
  pinMode(BOTAO_DETECCAO, INPUT_PULLUP); // botão liga o outro terminal ao GND

  digitalWrite(LED_VERMELHO, LOW);
  digitalWrite(LED_VERDE, HIGH);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Detector Ruido");
  atualizarLcdStatus();

  i2s_config_t config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
  };
  i2s_pin_config_t pins = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD,
  };

  i2s_driver_install(I2S_PORT, &config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pins);
}

void loop() {
  int leituraBotao = digitalRead(BOTAO_DETECCAO);
  if (leituraBotao == LOW && estadoBotaoAnterior == HIGH &&
      millis() - ultimoDebounce > DEBOUNCE_MS) {
    deteccaoAtiva = !deteccaoAtiva;
    ultimoDebounce = millis();
    digitalWrite(LED_VERDE, deteccaoAtiva ? HIGH : LOW);
    digitalWrite(LED_VERMELHO, LOW);
    atualizarLcdStatus();
  }
  estadoBotaoAnterior = leituraBotao;

  if (!deteccaoAtiva) {
    delay(10);
    return;
  }

  int32_t sample;
  size_t bytes_read;
  i2s_read(I2S_PORT, &sample, sizeof(sample), &bytes_read, portMAX_DELAY);
  Serial.println(sample);
}

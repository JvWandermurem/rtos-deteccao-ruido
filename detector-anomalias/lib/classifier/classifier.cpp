#include "classifier.h"
#include <math.h>

void standardize(const float* raw, const float* mean, const float* scale,
                 float* out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    const float divisor = (scale[i] == 0.0f) ? 1.0f : scale[i];
    out[i] = (raw[i] - mean[i]) / divisor;
  }
}

float classifier_score(const float* features, const float* weights, float bias,
                       size_t n) {
  float z = bias;
  for (size_t i = 0; i < n; i++) {
    z += weights[i] * features[i];
  }
  return 1.0f / (1.0f + expf(-z));
}

bool classifier_is_anomaly(float score, float threshold) {
  return score > threshold;
}

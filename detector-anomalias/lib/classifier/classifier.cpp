#include "classifier.h"

bool classifier_is_anomaly(float score, float threshold) {
  return score > threshold;
}

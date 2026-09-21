#include <unity.h>
#include "classifier.h"

void setUp(void) {}
void tearDown(void) {}

void test_threshold_decide_anomalia(void) {
  TEST_ASSERT_TRUE(classifier_is_anomaly(0.7f, 0.5f));
  TEST_ASSERT_FALSE(classifier_is_anomaly(0.3f, 0.5f));
}

int main(int argc, char** argv) {
  UNITY_BEGIN();
  RUN_TEST(test_threshold_decide_anomalia);
  return UNITY_END();
}

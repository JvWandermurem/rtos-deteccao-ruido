// Gerado por model/treino.py -- não editar à mão.
#ifndef CLASSIFIER_WEIGHTS_H
#define CLASSIFIER_WEIGHTS_H

// Árvore de decisão de profundidade 4, 6 folhas.
// Retorna a probabilidade de assobio na folha alcançada.
static inline float classifier_tree_score(const float* f) {
  if (f[0] <= 600495.281250f) {
    return 0.000000f;
  } else {
    if (f[5] <= 2.335250f) {
      if (f[2] <= 145.594299f) {
        if (f[1] <= 2348.574951f) {
          return 0.991255f;
        } else {
          return 0.000000f;
        }
      } else {
        if (f[6] <= -0.394150f) {
          return 0.000000f;
        } else {
          return 0.992779f;
        }
      }
    } else {
      return 0.000000f;
    }
  }
}

#endif

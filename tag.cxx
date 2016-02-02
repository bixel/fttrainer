#ifndef TAG_CXX
#define TAG_CXX

#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include "xgboost/c_api.h"

int main(int argc, char* argv[]){
  // TODO: read a root file here and build a xgboost compatible DMatrix
  DMatrixHandle mat;
  XGDMatrixCreateFromFile("../data/nnet_ele.xmat", 0, &mat);

  unsigned long num_rows;
  XGDMatrixNumRow(mat, &num_rows);

  std::cout << "File contains " << num_rows << " rows." << std::endl;

  DMatrixHandle mat_array[] = {mat};

  BoosterHandle booster;
  XGBoosterCreate(mat_array, 1, &booster);
  std::cout << "Loading model "
    << XGBoosterLoadModel(booster, "../models/ele_trained.xgb")
    << std::endl;

  // handle to store length and predictions
  unsigned long len;
  const float *predictions;

  XGBoosterPredict(booster, mat, 0, 0, &len, &predictions);

  for(unsigned long i=0; i<len; i++){
    std::cout << *predictions << std::endl;
    predictions++;
  }

  // use the model to predict something
  // auto *predictions = bst->Pred(*mat, 0x00, 0, len);
  return 0;
}

#endif // TAG_CXX

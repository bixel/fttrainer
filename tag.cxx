#ifndef TAG_CXX
#define TAG_CXX

#include <cmath>
#include <vector>
#include <iostream>
#include "io.h"
#include "xgboost_wrapper.cpp"

int main(int argc, char* argv[]){
  // TODO: read a root file here and build a xgboost compatible DMatrix
  auto *mat = xgboost::io::LoadDataMatrix("./data/nnet_ele.xmat",
                                          false,
                                          false,
                                          false);
  std::vector<DataMatrix*> mat_vector{mat};
  auto *bst = new xgboost::wrapper::Booster(mat_vector);

  // restore the model
  bst->LoadModel("./models/ele_trained.xgb");

  unsigned long *len = 0;

  // use the model to predict something
  // auto *predictions = bst->Pred(*mat, 0x00, 0, len);
  return 0;
}

#endif // TAG_CXX
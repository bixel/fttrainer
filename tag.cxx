#ifndef TAG_CXX
#define TAG_CXX

#include <cmath>
#include <vector>
#include <iostream>
#include "xgboost/c_api.h"

int main(int argc, char* argv[]){
  // TODO: read a root file here and build a xgboost compatible DMatrix
  auto *mat = DMatrixHandle();
  XGDMatrixCreateFromFile("../data/nnet_ele.xmat", 1, &mat);

  unsigned long num_rows;
  XGDMatrixNumRow(mat, &num_rows);

  std::cout << "File contains " << num_rows << " rows." << std::endl;

  // use the model to predict something
  // auto *predictions = bst->Pred(*mat, 0x00, 0, len);
  return 0;
}

#endif // TAG_CXX

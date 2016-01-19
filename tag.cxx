#ifndef TAG_CXX
#define TAG_CXX

#include <cmath>
#include <vector>
#include <iostream>
#include "io.h"
#include "xgboost_wrapper.cpp"

int main(int argc, char* argv[]){
  auto *mat = xgboost::io::LoadDataMatrix("./data/nnet_ele.xmat",
                                          false,
                                          false,
                                          false);
  std::vector<DataMatrix*> mat_vector{mat};
  auto b = xgboost::wrapper::Booster(mat_vector);
  return 0;
}

#endif // TAG_CXX

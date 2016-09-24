#ifndef TAG_CXX
#define TAG_CXX

#include <cmath>
#include <vector>
#include <array>
#include <iostream>

#include <boost/program_options.hpp>

#include "xgboost_wrapper.h"

namespace po = boost::program_options;
using namespace std;

int main(int argc, char* argv[]){
  string xgb_model_file;
  vector<double> data;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "print help message")
    ("model,m", po::value<string>(&xgb_model_file), "the trained XGBoost model")
    ("data,d", po::value<vector<double>>(&data)->multitoken(), "the data values which should be predicted")
    ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if(vm.count("help")){
    cout << desc << endl;
    return 0;
  }

  // TODO: read a root file here and build a xgboost compatible DMatrix
  DMatrixHandle data_mat,
                cache_mat;

  std::vector<float> data_(data.begin(), data.end());

  cout << "Data:\n";
  for(auto d : data_)
    cout << d << "\n";
  cout << endl;

  // fill the vector(!) into a csc matrix (ouch!)
  if(XGDMatrixCreateFromMat(&*data_.cbegin(),
                            1,
                            data.size(),
                            0.5,
                            &data_mat) != 0)
    cout << "Error creating DMatrix" << endl;

  unsigned long num_rows;
  XGDMatrixNumRow(data_mat, &num_rows);

  std::cout << "File contains " << num_rows << " rows." << std::endl;


  if(XGDMatrixCreateFromMat(&*std::vector<float>(2).cbegin(),
                            1,
                            2,
                            0.5,
                            &cache_mat) != 0)
    cout << "Error creating DMatrix" << endl;
  DMatrixHandle mat_array[] = {cache_mat};

  BoosterHandle booster;
  XGBoosterCreate(mat_array, 1, &booster);
  std::cout << "Loading model " << xgb_model_file << ": "
    << XGBoosterLoadModel(booster, xgb_model_file.c_str())
    << std::endl;

  // handle to store length and predictions
  unsigned long len;
  const float *predictions;

  XGBoosterPredict(booster, data_mat, 0, 0, &len, &predictions);

  for(unsigned long i=0; i<len; i++){
    std::cout << *predictions << std::endl;
    predictions++;
  }

  // use the model to predict something
  // auto *predictions = bst->Pred(*mat, 0x00, 0, len);
  return 0;
}

#endif // TAG_CXX

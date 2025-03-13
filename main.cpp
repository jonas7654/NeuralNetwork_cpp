#include "value.h"
#include "nn.h"
#include <iostream>


// This is a function which maps an array to a 2D-Matrix structure
Value* MAT_AT(Value** value_array, int n_cols, int i, int j) {
  return &(*value_array[n_cols * i + j]);
}

Value* cost(Value* y, Value* ypred){
  Value* diff = *y - ypred;
  Value* result = *diff * diff;
  return result;
}

int main() {
  // XOR Gate
  Value* train_data[12] = {
   new Value(0),new Value(0),new Value(0),
   new Value(0),new Value(1),new Value(1),
   new Value(1),new Value(0),new Value(1),
   new Value(1),new Value(1),new Value(0),
};
  // TEST LAYER 
  int epochs = 1000*100;
  constexpr int n_cols = 3;
  constexpr int n_rows = 4;
  int layer_sizes[3] = {2, 3, 1};
  constexpr int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

  nn nn(layer_sizes, num_layers);
  nn.train(train_data, 0.1, epochs, n_cols, n_rows);

  ///// forward
  Value* input[2] = {MAT_AT(train_data, n_cols, 0, 0), MAT_AT(train_data, n_cols, 0, 1)}; // x1 = 0, x2 = 0
  Value** output = nn.forward(input);
  for (int i = 0; i < 1; i++) {
    output[i]->printValue();
  }

  return 0;
}

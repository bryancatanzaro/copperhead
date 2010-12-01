#pragma once

int ceil_int_div(int x, int y) {
  return (x - 1)/y + 1;
}

template<typename Seq>
void update_grid_size(int block, Seq input, int &grid) {
  int input_size = input.size();
  int proposed_grid = ceil_int_div(input_size, block);
  if (proposed_grid > grid) {
    grid = proposed_grid;
  }
}

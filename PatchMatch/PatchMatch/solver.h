#pragma once

#ifndef SOLVER_H
#define SOLVER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

namespace solver {

using namespace cv;
using namespace std;

class Solver {
public:
    Mat3b image;
    Mat3b edit_layer;
    Mat3b A;
    Mat3b B;

    Mat1d curr_d;

    Mat2d nnf;  // Maps coordinates from A to patch in B
    Mat1d hole;

    Solver(Mat3b image, Mat3b edit_layer) : image(image), edit_layer(edit_layer) { Initialize(); }
    void Edit();

private:
    int ROWS;
    int COLS;

    int PATCH_SIZE = 7; // Must be odd
    int ITERATIONS = 5;
    Vec3b EDIT_COLOR = Vec3b(0, 0, 0);

    void Initialize();
    void InitializeNNF();
    void Propagation(bool top);
    void RandomSearch();

    double PatchDistance(int a_row, int a_col, int d_row, int d_col);
    bool IsValidCell(int r, int c);

    bool IsForEdit(Vec3b v);
    int Random(int a, int b);
};
} // namespace solver
#endif // !SOLVER_H

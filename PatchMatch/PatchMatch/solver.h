#pragma once

#ifndef SOLVER_H
#define SOLVER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

namespace solver {

using namespace cv;
using namespace std;

class Solver {
public:
    Mat3b image;
    Mat3b edit_layer;
    Mat3b A;

    //Mat1d curr_d;

    Mat2i nnf;  // Maps coordinates from A to patch in B
    Mat1i hole;

    Solver(Mat3b i, Mat3b e);
    void Edit();
    void DisplayNNF(int iteration);
    void DisplayImage(int iteration);

private:
    int ROWS;
    int COLS;
    int PATCH_SIZE = 7; // Must be odd
    int ITERATIONS = 5;
    int RANDOM_ITERATIONS = 20;
    Vec3b EDIT_COLOR = Vec3b(0, 0, 0);

    void InitializeNNF();
    void Interleave(bool top);
    void Propagation(int row, int col, int next_row, int next_col);
    void RandomSearch(int row, int col);
    void UpdateA(int row, int col);

    double PatchDistance(int a_row, int a_col, Vec2i offset);
    double PatchDistance(int a_row, int a_col, int d_row, int d_col);
    bool IsValidCell(int r, int c);

    void SquashNNF(int row, int col);
    bool IsForEdit(Vec3b v);
    bool IsHole(int r, int c);
    int Random(int a, int b);
};
} // namespace solver
#endif // !SOLVER_H

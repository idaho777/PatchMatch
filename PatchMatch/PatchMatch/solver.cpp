#include "solver.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <sstream>
#include <random>

namespace solver {

Solver::Solver(Mat3b i, Mat3b e) {
    image = i;
    edit_layer = e;
    ROWS = image.rows;
    COLS = image.cols;
}


void Solver::Edit() {
    InitializeNNF();
    A = image;

    for (int i = 0; i < ITERATIONS; ++i) {
        cout << "Iteration: " << i << endl;
        Interleave(i % 2 == 0);
        DisplayImage(i);
    }
}


void Solver::DisplayNNF(int iteration) {
    Mat3b image_nnf = Mat3b(ROWS, COLS);
    
    MatIterator_<Vec3b> it = image_nnf.begin();
    MatIterator_<Vec2i> nnf_it = nnf.begin();
    while (nnf_it != nnf.end()) {
        *it = Vec3b(0, 0, 0);
        if (*nnf_it != Vec2i(0, 0)) {
            int r = ((*nnf_it)[1] + ROWS) / (2.0 * ROWS) * 255;
            int g = ((*nnf_it)[0] + COLS) / (2.0 * COLS) * 255;
            int b = 0;

            *it = Vec3b(b, g, r);
            //cout << nnf_it.pos() << " " << *nnf_it << " " << *it << endl;
        }

        ++it;
        ++nnf_it;
    }

    stringstream ss;
    ss << "NNF: " << iteration;
    string name = ss.str();
    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, image_nnf); // Show our image inside it.
}


void Solver::DisplayImage(int iteration) {
    stringstream ss;
    ss << "Image: " << iteration;
    string name = ss.str();
    namedWindow(name, WINDOW_AUTOSIZE);

    Mat3b A_bgr;
    cvtColor(A, A_bgr, COLOR_Lab2BGR);
    imshow(name, A_bgr); // Show our image inside it.
}


void Solver::Interleave(bool top) {
    int start_row = 1;
    int start_col = 1;
    int end_row = ROWS;
    int end_col = COLS;
    int next_row = 1;
    int next_col = 1;

    if (!top) {
        start_row = ROWS - 2;
        start_col = COLS - 2;
        end_row = -1;
        end_col = -1;
        next_row = -1;
        next_col = -1;
    }

    for (int row = start_row; row != end_row; row += next_row) {
        for (int col = start_col; col != end_col; col += next_col) {
            if (!IsHole(row, col)) continue;
            Propagation(row, col, next_row, next_col);
            RandomSearch(row, col);
            UpdateA(row, col);
        }
    }
}


// ====================================================================================================================
void Solver::Propagation(int row, int col, int next_row, int next_col) {
    Vec2i hori_offset = nnf(row, col - next_col);
    Vec2i curr_offset = nnf(row, col);
    Vec2i vert_offset = nnf(row - next_row, col);

    double hori_d = PatchDistance(row, col, hori_offset);
    double curr_d = PatchDistance(row, col, curr_offset);
    double vert_d = PatchDistance(row, col, vert_offset);

    Vec2i arg_min = curr_offset;
    double min_d = curr_d;

    // Only consider solved holes.  Non-holes are perfect 0,0
    if (IsHole(row, col - next_col) && hori_d < min_d) {
        arg_min = hori_offset;
        min_d = hori_d;
    }

    if (IsHole(row - next_row, col) && vert_d < min_d) {
        arg_min = vert_offset;
        min_d = vert_d;
    }

    nnf(row, col) = arg_min;
    SquashNNF(row, col);
}


bool Solver::IsValidCell(int r, int c) {
    return r >= 0 && r < ROWS && c >= 0 && c < COLS;
}


// ====================================================================================================================
void Solver::RandomSearch(int row, int col) {
    Vec2i min_offset = nnf(row, col);
    double min_d = PatchDistance(row, col, min_offset);

    int i = 0;
    while (i < RANDOM_ITERATIONS) {
        Vec2i b = Vec2i(Random(0, ROWS), Random(0, COLS));
        if (IsHole(b[0], b[1])) continue;

        double cand_d = PatchDistance(row, col, b[0], b[1]);
        if (cand_d < min_d) {
            min_offset = Vec2i(b[0] - row, b[1] - col);
            min_d = cand_d;
        }

        ++i;
    }

    nnf(row, col) = min_offset;
    SquashNNF(row, col);
}


// ====================================================================================================================
void Solver::UpdateA(int row, int col) {
    SquashNNF(row, col);
    Vec2i d = nnf(row, col);
    A(row, col) = A(row + d[0], col + d[1]);
}


// ====================================================================================================================
void Solver::InitializeNNF() {
    Mat2i ret(ROWS, COLS);
    Mat1i ret_hole(ROWS, COLS);

    MatIterator_<Vec3b> it = edit_layer.begin();
    MatIterator_<Vec2i> ret_it = ret.begin();
    MatIterator_<int> ret_hole_it = ret_hole.begin();

    while (it != edit_layer.end()) {
        Vec3b curr_edit = *it;
        Vec2i curr_d_vec = Vec2b(0, 0);
        *ret_hole_it = 0;
        Vec2i pos = Vec2i(it.pos().y, it.pos().x);

        if (IsForEdit(curr_edit)) {
            curr_d_vec[0] = Random(0, ROWS) - pos[0];
            curr_d_vec[1] = Random(0, COLS) - pos[1];
            *ret_hole_it = 1;
        }

        int rp = pos[0] + curr_d_vec[0];
        int cp = pos[1] + curr_d_vec[1];
        if (IsForEdit(edit_layer(rp, cp))) continue;

        *ret_it = curr_d_vec;
        it++;
        ret_it++;
        ret_hole_it++;
    }

    nnf = ret;
    hole = ret_hole;
}


// ====================================================================================================================
double Solver::PatchDistance(int a_row, int a_col, Vec2i offset) {
    int b_row = a_row + offset[0];
    int b_col = a_col + offset[1];

    return PatchDistance(a_row, a_col, b_row, b_col);
}


double Solver::PatchDistance(int a_row, int a_col, int b_row, int b_col) {
    double ret = 0;
    int half = PATCH_SIZE / 2;

    for (int ro = -half; ro <= half; ++ro) {
        for (int co = -half; co <= half; ++co) {
            if (IsValidCell(a_row + ro, a_col + co) && IsValidCell(b_row + ro, b_col + co)) {
                Vec3d A_cell = A(a_row + ro, a_col + co);
                Vec3d B_cell = A(b_row + ro, b_col + co);
                double rd = pow(A_cell[0] - B_cell[0], 2);
                double gd = pow(A_cell[1] - B_cell[1], 2);
                double bd = pow(A_cell[2] - B_cell[2], 2);

                ret += rd + gd + bd;
            }
        }
    }

    return ret;
}


void Solver::SquashNNF(int row, int col) {
    Vec2i offset = nnf(row, col);
    if (!IsValidCell(row + offset[0], col + offset[1])) {
        if (row + offset[0] >= ROWS) --offset[0];
        if (row + offset[0] < 0)     ++offset[0];
        if (col + offset[1] >= COLS) --offset[1];
        if (col + offset[1] < 0)     ++offset[1];
    }
}


bool Solver::IsForEdit(Vec3b v) {
    return v == EDIT_COLOR;
}


bool Solver::IsHole(int r, int c) {
    return hole(r, c) == 1;
}

int Solver::Random(int a, int b) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(a, b - 1);

    return dis(gen);
}

} // namespace solver
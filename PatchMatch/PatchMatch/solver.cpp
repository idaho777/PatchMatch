#include "solver.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <random>
#include <iostream>

namespace solver {

void Solver::Initialize() {
    ROWS = image.rows;
    COLS = image.cols;
}


void Solver::Edit() {
    InitializeNNF();
    A = image;


    for (int i = 0; i < ITERATIONS; ++i) {
        DisplayNNF(i);
        cout << "Iteration: " << i << endl;
        curr_d = -1 * Mat1d::ones(ROWS, COLS);
        Propagation(i % 2 == 0);
        RandomSearch();

        /*
            Update A from B
            Display A
            Replace B with A
        */
    }

    DisplayNNF(ITERATIONS);
    /*
    Section 3:
        - Nearest Neighbor Field (NNF) f: A -> R2 defined over all patch coordinates(patch centers) in image A.
            In this case, f(a) = b - a, or a vector indicating the coordinate offset.
            We need to create an offset array and then use this offset array to color our image.

        - Initialization:
            Initialize all offsets as zero except for those colored: be random vector

        -Iterations:
            Improving the mapping.  Alternate between propogation and random search.
            - Propogation:
                for (x,y) colored, check left and top.  argmin(Distance of {left, center, top})
                Examine iterations in reverse for even iteration numbers.
            - Random Search:
                Find a random vector ui = v0 + w a^i R, where v0 is f(x,y), w = maximum image dimension, a is ratio of window sizes, R is unit window.
            Do this iteraition 5 times.

    */



    // Create NNF
}


void Solver::DisplayNNF(int iteration) {
    /*
    Mat3b image_nnf = Mat3b(ROWS, COLS);
    
    MatIterator_<Vec3b> it = image_nnf.begin();
    MatIterator_<Vec3b> image_it = image.begin();
    MatIterator_<Vec2i> nnf_it = nnf.begin();


    while (nnf_it != nnf.end()) {
        int l = 100;
        int a = (*nnf_it)[1] / ROWS;
        int b = (*nnf_it)[0] / COLS;

        //*it = Vec3b(l, a, b);
        *it = *image_it;

        it++;
        image_it++;
        nnf_it++;
    }
    */

    //Mat3b image_nnf_bgr = Mat3b(ROWS, COLS);
    //cvtColor(image, image_nnf_bgr, COLOR_Lab2BGR);

    namedWindow("NNF: " + iteration, WINDOW_AUTOSIZE);
    imshow("Display window", image); // Show our image inside it.
}


void Solver::DisplayImage() {

}


// ====================================================================================================================
void Solver::Propagation(bool top) {
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
            Vec2i hori_nnf = nnf(row, col - next_col);
            Vec2i curr_nnf = nnf(row, col);
            Vec2i vert_nnf = nnf(row - next_row, col);

            double hori_d = PatchDistance(row, col, hori_nnf[1], hori_nnf[0]);
            double curr_d = PatchDistance(row, col, curr_nnf[1], curr_nnf[0]);
            double vert_d = PatchDistance(row, col, vert_nnf[1], vert_nnf[0]);

            Vec2i arg_min = nnf(row, col);
            double min_d = curr_d;

            if (hori_d < min_d) {
                arg_min = hori_nnf;
                min_d = hori_d;
            }

            if (vert_d < min_d) {
                arg_min = vert_nnf;
                min_d = vert_d;
            }
            
            nnf(row, col) = arg_min;   
        }
    }
}


bool Solver::IsValidCell(int r, int c) {
    return r >= 0 && r < ROWS && c >= 0 && c < COLS;
}


// ====================================================================================================================
void Solver::RandomSearch() {

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
        Vec2i pos = Vec2i(it.pos().x, it.pos().y);

        if (IsForEdit(curr_edit)) {
            curr_d_vec[0] = Random(0, COLS) - pos[0];
            curr_d_vec[1] = Random(0, ROWS) - pos[1];
            *ret_hole_it = 1;
        }

        int xp = pos[0] + curr_d_vec[0];
        int yp = pos[1] + curr_d_vec[1];
        if (IsForEdit(edit_layer(yp, xp))) continue;

        *ret_it = curr_d_vec;
        it++;
        ret_it++;
        ret_hole_it++;
    }

    nnf = ret;
    hole = ret_hole;
}


// ====================================================================================================================
double Solver::PatchDistance(int a_row, int a_col, int d_row, int d_col) {
    int b_row = a_row + d_row;
    int b_col = a_col + d_col;

    double ret = 0;
    int half = PATCH_SIZE / 2;

    for (int ro = -half; ro <= half; ++ro) {
        for (int co = -half; co <= half; ++co) {
            if (IsValidCell(a_row + ro, a_col + co) && IsValidCell(b_row + ro, b_col + co)) {
                Vec3d A_cell = A(a_row + ro, a_col + co);
                Vec3d B_cell = A(b_row + ro, b_col + co);
                double rd = sqrt(A_cell[0] - B_cell[0]);
                double gd = sqrt(A_cell[1] - B_cell[1]);
                double bd = sqrt(A_cell[2] - B_cell[2]);

                ret += rd + gd + bd;
            }
        }
    }

    return ret;
}



bool Solver::IsForEdit(Vec3b v) {
    return v == EDIT_COLOR;
}


int Solver::Random(int a, int b) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(a, b - 1);

    return dis(gen);
}

} // namespace solver
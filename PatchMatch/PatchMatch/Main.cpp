
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "solver.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << " Usage: ImageToEdit Edits" << endl;
        return -1;
    }

    // Load Images
    Mat3b image_bgr, image_lab, edit_layer;
    image_bgr = imread(argv[1], IMREAD_COLOR); // Read the file
    cvtColor(image_bgr, image_lab, COLOR_BGR2Lab);
    edit_layer = imread(argv[2], IMREAD_COLOR);

    if (!image_bgr.data || !image_lab.data || !edit_layer.data) { // Check for invalid input
        cout << "Could not open or find the image or edit" << std::endl;
        return -1;
    }

    // Solver
    solver::Solver solver(image_lab, edit_layer);
    solver.Edit();

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}

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
    Mat3b image_rgb, edit_layer;
    image_rgb  = imread(argv[1], IMREAD_COLOR); // Read the file
    edit_layer = imread(argv[2], IMREAD_COLOR);

    if (!image_rgb.data || !edit_layer.data) { // Check for invalid input
        cout << "Could not open or find the image or edit" << std::endl;
        return -1;
    }

    solver::Solver solver(image_rgb, edit_layer);
    solver.Edit();

    /*
        Goal: Filling in hole
        Stretch Goal: Filling in hole with constraints
    */


    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", image_rgb); // Show our image inside it.

    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Edit window", edit_layer); // Show our image inside it.


    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
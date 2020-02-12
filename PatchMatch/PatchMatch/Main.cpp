
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
    
    solver::Solver solver(image_bgr, edit_layer);
    solver.Edit();

    /*for (auto it = image_lab.begin(); it != image_lab.end(); ++it) {
        cout << *it << endl;
    }*/

    Mat3b edited_image_lab = image_lab;
    Mat3b edited_image_bgr;
    cvtColor(edited_image_lab, edited_image_bgr, COLOR_Lab2BGR);

    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", edited_image_bgr); // Show our image inside it.
//    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
//    imshow("Edit window", edit_layer); // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
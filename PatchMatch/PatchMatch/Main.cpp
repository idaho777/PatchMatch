
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    Mat image;
    image = imread(argv[1], IMREAD_COLOR); // Read the file


    MatIterator_<Vec3b> it;
    for (it = image.begin<Vec3b>(); it != image.end<Vec3b>(); ++it) {
        //cout << (*it) << endl;
    }
    /*
        Goal: Filling in hole
        Stretch Goal: Filling in hole with constraints

        Get Image to Fix
        Color Image in another program
        Read image in

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

        With the mapping, create new image with mappings.

        
        Iterative Process:
            - Propagation:


        Get Image
        Color Image

    
    
    
    */


    if (!image.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", image); // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
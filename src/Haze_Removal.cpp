#include "hazeremoval.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **args) {
        const char * img_path = args[1];
        Mat in_img = imread("images/Haze_Singapore.jpg", IMREAD_UNCHANGED);
        Mat out_img(in_img.rows, in_img.cols, CV_8UC3);
        unsigned char * indata = in_img.data;
        unsigned char * outdata = out_img.data;

        CHazeRemoval hr;
        cout << hr.InitProc(in_img.cols, in_img.rows, in_img.channels()) << endl;
        //auto start = std::chrono::high_resolution_clock::now();
        cout << hr.Process(indata, outdata, in_img.cols, in_img.rows, in_img.channels()) << endl;
        //auto stop = std::chrono::high_resolution_clock::now();
        //float duration = std::chrono::duration_cast<microseconds>(stop - start).count();
        //cout << "Processing time: " << to_string(duration) << " us ~ " << to_string(1/(duration/1000000)) << " FPS" << endl;
        //cout << "Expected       : 16667 us ~ 60 FPS" << endl;
        namedWindow("out_img", WINDOW_AUTOSIZE);
        imshow("out_img", out_img);
        waitKey(0);
        // system("pause");
        return 0;
}

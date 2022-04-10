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
  Mat in_img;
  if (args[1])
    in_img = imread(img_path, IMREAD_UNCHANGED);
  else
    in_img = imread("images/Haze_Singapore.jpg", IMREAD_UNCHANGED);
  Mat out_img(in_img.rows, in_img.cols, CV_8UC3);
  unsigned char * indata = in_img.data;
  unsigned char * outdata = out_img.data;

  CHazeRemoval hr;
  cout << hr.InitProc(in_img.cols, in_img.rows, in_img.channels()) << endl;
  cout << hr.Process(indata, outdata, in_img.cols, in_img.rows, in_img.channels()) << endl;

  Mat dst;
  cv::hconcat(in_img, out_img, dst);
  namedWindow("IDCP", WINDOW_AUTOSIZE);
  imshow("IDCP", dst);
  waitKey(0);
  return 0;
}

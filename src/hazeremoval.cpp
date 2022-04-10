#include "hazeremoval.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <chrono>

using namespace cv;
using namespace std;

CHazeRemoval::CHazeRemoval() {
  rows = 0;
  cols = 0;
  channels = 0;
}

CHazeRemoval::~CHazeRemoval() {

}

bool CHazeRemoval::InitProc(int width, int height, int nChannels) {
  bool ret = false;
  rows = height;
  cols = width;
  channels = nChannels;

  if (width > 0 && height > 0 && nChannels == 3) ret = true;
  return ret;
}

bool CHazeRemoval::Process(const unsigned char* indata, unsigned char* outdata, int width, int height, int nChannels) {
  bool ret = true;
  if (!indata || !outdata) {
    ret = false;
  }
  rows = height;
  cols = width;
  channels = nChannels;

  int radius = 7;
  double omega = 0.95;
  double t0 = 0.1;
  int r = 60;
  double eps = 0.001;
  vector<Pixel> tmp_vec;
  Mat * p_src = new Mat(rows, cols, CV_8UC3, (void *)indata);
  Mat * p_dst = new Mat(rows, cols, CV_64FC3);
  Mat * p_tran = new Mat(rows, cols, CV_64FC1);
  Mat * p_gtran = new Mat(rows, cols, CV_64FC1);
  Vec3d * p_Alight = new Vec3d();

#if _MEASURE_RUNTIME_
  auto start0 = std::chrono::high_resolution_clock::now();
  auto start = start0;
#endif
  get_dark_channel(p_src, tmp_vec, rows, cols, channels, radius);
#if _MEASURE_RUNTIME_
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> fp_ms = stop - start;
  std::cout << "get_dark_channel() took " << fp_ms.count() << " ms.\n";
#endif

#if _MEASURE_RUNTIME_
  start = std::chrono::high_resolution_clock::now();
#endif
  get_air_light(p_src, tmp_vec, p_Alight, rows, cols, channels);
#if _MEASURE_RUNTIME_
  stop = std::chrono::high_resolution_clock::now();
  fp_ms = stop - start;
  std::cout << "get_air_light() took " << fp_ms.count() << " ms.\n";
#endif

#if _MEASURE_RUNTIME_
  start = std::chrono::high_resolution_clock::now();
#endif
  get_transmission(p_src, p_tran, p_Alight, rows, cols, channels, radius = 7, omega);
#if _MEASURE_RUNTIME_
  stop = std::chrono::high_resolution_clock::now();
  fp_ms = stop - start;
  std::cout << "get_transmission() took " << fp_ms.count() << " ms.\n";
#endif

#if _MEASURE_RUNTIME_
  start = std::chrono::high_resolution_clock::now();
#endif
  guided_filter(p_src, p_tran, p_gtran, r, eps);
#if _MEASURE_RUNTIME_
  stop = std::chrono::high_resolution_clock::now();
  fp_ms = stop - start;
  std::cout << "guided_filter() took " << fp_ms.count() << " ms.\n";
#endif

#if _MEASURE_RUNTIME_
  start = std::chrono::high_resolution_clock::now();
#endif
  recover(p_src, p_gtran, p_dst, p_Alight, rows, cols, channels, t0);
#if _MEASURE_RUNTIME_
  stop = std::chrono::high_resolution_clock::now();
  fp_ms = stop - start;
  std::cout << "recover() took " << fp_ms.count() << " ms.\n";
#endif

#if _MEASURE_RUNTIME_
  start = std::chrono::high_resolution_clock::now();
#endif
  assign_data(outdata, p_dst, rows, cols, channels);
#if _MEASURE_RUNTIME_
  auto stop0 = std::chrono::high_resolution_clock::now();
  stop = stop0;
  fp_ms = stop - start;
  std::cout << "assign_data() took " << fp_ms.count() << " ms.\n";
  fp_ms = stop0 - start0;
  std::cout << "\nTotal runtime: " << fp_ms.count() << " ms.\n";
#endif

  return ret;
}

bool sort_fun(const Pixel&a, const Pixel&b) {
  return a.val > b.val;
}

void get_dark_channel(const cv::Mat *p_src, std::vector<Pixel> &tmp_vec, int rows, int cols, int channels, int radius) {
  int rmin;
  int rmax;
  int cmin;
  int cmax;
  double min_val;
  cv::Vec3b tmp;
  uchar b, g, r;
  std::vector<uchar> tmp_value(3);
  uchar median;
  uchar threshold_lo;
  uchar minpixel;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      rmin = cv::max(0, i - radius);
      rmax = cv::min(i + radius, rows - 1);
      cmin = cv::max(0, j - radius);
      cmax = cv::min(j + radius, cols - 1);
      min_val = 255;
      for (int x = rmin; x <= rmax; x++) {
        for (int y = cmin; y <= cmax; y++) {
          tmp = p_src->ptr<cv::Vec3b>(x)[y];
          b = tmp[0];
          g = tmp[1];
          r = tmp[2];
          // find median value
          tmp_value[0] = b;
          tmp_value[0] = g;
          tmp_value[0] = r;
          //cout << to_string(b) << " - " << to_string(g) << " - " << to_string(r) << "\n";
          std::sort(tmp_value.begin(), tmp_value.end());
          //cout << to_string(tmp_value[0]) << " - " << to_string(tmp_value[1]) << " - " << to_string(tmp_value[2]) << "\n";
          median = tmp_value[1]/2;
          threshold_lo = median/1.5;
          //uchar threshold_hi = median*1.5;
          if (b < threshold_lo)
            minpixel = (g>r) ? r : g;
          else
            minpixel = b > g ? ((g>r) ? r : g) : ((b > r) ? r : b);
          //cout << to_string(median) << " - " << to_string(threshold_lo) << "\n----------\n";
          //if (y==100)  exit(0);
          min_val = cv::min((double)minpixel, min_val);
        }
      }
      tmp_vec.push_back(Pixel(i, j, uchar(min_val)));
    }
  }
  std::sort(tmp_vec.begin(), tmp_vec.end(), sort_fun);
}

void get_air_light(const cv::Mat *p_src, std::vector<Pixel> &tmp_vec, cv::Vec3d *p_Alight, int rows, int cols, int channels) {
  int num = int(rows*cols*0.001);
    double A_sum[3] = { 0, };
    std::vector<Pixel>::iterator it = tmp_vec.begin();
    for (int cnt = 0; cnt<num; cnt++) {
        cv::Vec3b tmp = p_src->ptr<cv::Vec3b>(it->i)[it->j];
        A_sum[0] += tmp[0];
        A_sum[1] += tmp[1];
        A_sum[2] += tmp[2];
        it++;
    }
    for (int i = 0; i < 3; i++) {
        (*p_Alight)[i] = A_sum[i] / num;
    }
}

void get_transmission(const cv::Mat *p_src, cv::Mat *p_tran, cv::Vec3d *p_Alight, int rows, int cols, int channels, int radius, double omega) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int rmin = cv::max(0, i - radius);
      int rmax = cv::min(i + radius, rows - 1);
      int cmin = cv::max(0, j - radius);
      int cmax = cv::min(j + radius, cols - 1);
      double min_val = 255.0;
      for (int x = rmin; x <= rmax; x++) {
        for (int y = cmin; y <= cmax; y++) {
          cv::Vec3b tmp = p_src->ptr<cv::Vec3b>(x)[y];
          double b = (double)tmp[0] / (*p_Alight)[0];
          double g = (double)tmp[1] / (*p_Alight)[1];
          double r = (double)tmp[2] / (*p_Alight)[2];
          double minpixel = b > g ? ((g>r) ? r : g) : ((b > r) ? r : b);
          min_val = cv::min(minpixel, min_val);
        }
      }
      p_tran->ptr<double>(i)[j] = 1 - omega*min_val;
    }
  }
}

void guided_filter(const cv::Mat *p_src, const cv::Mat *p_tran, cv::Mat *p_gtran, int r, double eps) {
  *p_gtran = guidedFilter(*p_src, *p_tran, r, eps);
}

void recover(const cv::Mat *p_src, const cv::Mat *p_gtran, cv::Mat *p_dst, cv::Vec3d *p_Alight, int rows, int cols, int channels, double t0) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      for (int c = 0; c < channels; c++) {
        double val = (double(p_src->ptr<cv::Vec3b>(i)[j][c]) - (*p_Alight)[c]) / cv::max(t0, p_gtran->ptr<double>(i)[j]) + (*p_Alight)[c];
        p_dst->ptr<cv::Vec3d>(i)[j][c] = cv::max(0.0, cv::min(255.0, val));
      }
    }
  }
}

void assign_data(unsigned char *outdata, const cv::Mat *p_dst, int rows, int cols, int channels) {
  for (int i = 0; i < rows*cols*channels; i++) {
    *(outdata + i) = (unsigned char)(*((double*)(p_dst->data) + i));
  }
}

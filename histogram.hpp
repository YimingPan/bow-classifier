#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include <opencv2/opencv.hpp>

void computeHistogram(cv::Mat& wordMap, cv::Mat& h, int dictionarySize);

cv::Mat distance(cv::Mat& sample, cv::Mat& observations);

#endif

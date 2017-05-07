#include "histogram.hpp"

/*
 * Extracts the histogram of visual words within the given image.
 * The resulting histogram h is L1 normalized.
 * h[i]: the occurence of the i-th visual word.
 */
void computeHistogram(cv::Mat& wordMap, cv::Mat& h, int dictionarySize)
{
    h = cv::Mat::zeros(1, dictionarySize, CV_64F);
    double* h_ptr = h.ptr<double>(0);
    for (int i = 0; i < wordMap.rows; i++)
    {
        for (int j = 0; j < wordMap.cols; j++)
        {
            int word = wordMap.at<int>(i,j);
            h_ptr[word] = h_ptr[word] + 1.0;
        }
    }

    // L1 normalize histogram h.
    double sumh = cv::sum(h)[0];
    h = h / sumh;
}

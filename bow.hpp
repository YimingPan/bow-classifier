#ifndef BOW_H_
#define BOW_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

class FilterBank
{
private:
    vector<Mat> filters; // filter list

    void initialize(vector<double>& scales,
                    vector<double>& gaussianSigmas,
                    vector<double>& logSigmas,
                    vector<double>& dGaussianSigmas);

    /*
     * Gaussian and LoG kernel generator.
     */
    Mat getGaussianFilter(int ksize, double sigma);
    Mat getLOGFilter(int ksize, double sigma);

public:
    FilterBank();
    FilterBank(vector<double>& scales, vector<double>& gaussianSigmas,
               vector<double>& logSigmas, vector<double>& dGaussianSigmas);
    ~FilterBank();

    /*
     * Get the filter response of image.
     * response is a numPixels * (numFilters*3) matrix.
     */
    void filter(Mat& image, Mat& response);
};

class Dictionary
{
private:
    Mat dictionary;
    vector<Mat> vec_allFilterResponses;

public:
    Dictionary();
    ~Dictionary();

    void create(int alpha, int K, FilterBank& filterbank,
                vector<string>& trainingImagesPath, string& imagesDir);

    /*
     * Saves the dictionary to a local file.
     */
    void save(const string& path);

    /*
     * Loads a dictionary stored in a local file.
     */
    void load(const string& path);

    /*
     * Returns the number of visual words contained in dictionary.
     */
    int getWordsNum();

    Mat getWordmap(const Mat& image, FilterBank& filterbank);

private:
    void randAlpha(vector<int> &randomIndex, int N, int alpha);
    void dbg_initialize(vector<Mat>& vec_allFilterResponses);
    int nearestWord(Mat& oneResponse); 

};

#endif

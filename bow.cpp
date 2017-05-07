#include "bow.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;
using namespace cv;

#define NUM_CHANNEL 3

/*
 * Default constructor of filterbank.
 */
FilterBank::FilterBank()
{
    // Default parameters.
    vector<double> s = {1, 2, 3};      // scales
    vector<double> gs = {1, 2, 4};     // gaussianSigmas
    vector<double> ls = {1, 2, 4, 8};  // logSigmas
    vector<double> ds = {2, 4};        // dGaussianSigmas

    initialize(s, gs, ls, ds);
}

/*
 * Constructs the filterbank with given parameters.
 */
FilterBank::FilterBank(vector<double>& scales,
                       vector<double>& gaussianSigmas,
                       vector<double>& logSigmas,
                       vector<double>& dGaussianSigmas)
{
    initialize(scales, gaussianSigmas, logSigmas, dGaussianSigmas);
}

FilterBank::~FilterBank() {}

/*
 * Creates filters.
 */
void FilterBank::initialize(vector<double>& scales,
                    vector<double>& gaussianSigmas,
                    vector<double>& logSigmas,
                    vector<double>& dGaussianSigmas)
{
    for (double scale : scales)
    {
        double scaleMultiply = pow(sqrt(2), scale);

        // Creates gaussian filters.
        for (double s : gaussianSigmas)
        {
            double sigma = s*scaleMultiply;
            int ksize = ceil(sigma*6+1);
            Mat kernel = getGaussianFilter(ksize, sigma);
            filters.push_back(kernel);
        }

        // Create d/dx, d/dy gaussians.
        Mat dx, dy;
        getDerivKernels(dx, dy, 1, 1, 3);
        dy = dy.t();
        for (double s : dGaussianSigmas)
        {
            double sigma = s*scaleMultiply;
            int ksize = ceil(sigma*6+1);
            Mat kernel = getGaussianFilter(ksize, sigma);
            Mat dk;
            filter2D(kernel, dk, -1, dx);
            filters.push_back(dk);
            filter2D(kernel, dk, -1, dy);
            filters.push_back(dk);
        }

        // Creates LoG filters.
        for (double s : logSigmas)
        {
            double sigma = s*scaleMultiply;
            int ksize = ceil(sigma*6+1);
            Mat kernel = getLOGFilter(ksize, sigma);
            filters.push_back(kernel);
        }
    }
}

/*
 * Get the filter response of image.
 * response is a numPixels * (numFilters*3) matrix.
 */
void FilterBank::filter(Mat& image, Mat& response)
{
    int numPixels = image.rows * image.cols;
    int numFilters = filters.size();
    int ddepth = -1;
    Mat tmp;

    cvtColor(image, image, CV_BGR2Lab); // Convert to Lab
    response.create(numPixels, numFilters*3, CV_64F);

    int idx = 0;
    for (Mat filter : filters)
    {
        filter2D(image, tmp, ddepth, filter); // store filtered image in tmp
        tmp = tmp.reshape(1, numPixels); // make numPixels*3 1-channel matrix
        tmp.copyTo(response.colRange(idx, idx+3));
        idx += 3;
    }
}

/*
 * Gaussian kernel generator.
 */
Mat FilterBank::getGaussianFilter(int ksize, double sigma)
{
    const double EPS = 1e-6;
    const double C = 2*sigma*sigma;

    Mat kernel(ksize, ksize, CV_64F);
    int m = ksize / 2;
    for (int i=0; i < kernel.rows; i++)
    {
        int x = i - m;
        double* pData = kernel.ptr<double>(i);
        for (int j = 0; j < kernel.cols; j++)
        {
            int y = j - m;
            pData[j] = exp( -(x*x+y*y) /  C);
            if (fabs(pData[j]) < EPS)
                pData[j] = 0.0;
        }
    }
    return kernel;
}

/*
 * LoG kernel generator.
 */
Mat FilterBank::getLOGFilter(int ksize, double sigma)
{
    // Calculates Gaussian kernel first.
    Mat kernel = getGaussianFilter(ksize, sigma);
    int m = ksize / 2;

    // Then calculates the Laplacian.
    const double C1 = 2*sigma*sigma;
    const double C2 = sigma*sigma*sigma*sigma;
    for (int i = 0; i < kernel.rows; i++)
    {
        int x = i - m;
        double* pData = kernel.ptr<double>(i);
        for (int j = 0; j < kernel.cols; j++)
        {
            int y = j - m;
            pData[j] *= (x*x + y*y - C1) / C2;
        }
    }

    double sumv = sum(kernel)[0];
    kernel -= sumv / (ksize*ksize);

    return kernel;
}


Dictionary::Dictionary(){
    
}
Dictionary::~Dictionary() {

}
/*
 * parameters
 * alpha: for each picture, randomly choose alpha pixels, between 50 and 150
 * K: There are totally K words in the dictionary, between 100 and 300
 * filterbank: all the kernels are in the filterbank
 * trainingImagesPath: path of the image files
 */
void Dictionary::create(int alpha, int K, FilterBank& filterbank,
            vector<string>& trainingImagesPath, string& imagesDir) {
   
    int numImg = (int)trainingImagesPath.size();
    int numRes;

    //for each the images, get the filter response
    //select alpha responses from each image and put all in a Mat
    Mat Response; 
    Mat allSelectedFilterResponses;
    vector<int> randomIndex;
    int N = 0;
    for(int i = 0; i < numImg; i++) {
        //cout << "Processing image " << i+1 << "/" << numImg << endl;

       //read the image
        Mat Img = imread(imagesDir + trainingImagesPath[i], 1);

       //filter response
        filterbank.filter(Img, Response);

        numRes = Response.cols;

        randomIndex.clear();
        //decide which alpha pixels to take
        N = Response.rows;
        //decide which alpha pixels to take        
        randAlpha(randomIndex, N, alpha);

        Mat newMat = Mat::zeros(alpha, numRes, CV_64F);
        for(int j = 0; j < alpha; j++) {
            for(int k = 0; k < numRes; k++)
            newMat.at<double>(j,k) = Response.at<double>(randomIndex[j],k);
        }

        for(int j = 0; j < alpha; j++) {
            allSelectedFilterResponses.push_back(newMat.row(j));
        }

       
    }
    

    //kmeans to get K clusters
    int allResponsesRows = allSelectedFilterResponses.rows;
    Mat floatAllResponses(allResponsesRows, numRes, CV_32F);
    for(int i = 0; i < allResponsesRows; i++)
        for(int j = 0; j < numRes; j++)
            floatAllResponses.at<float>(i,j) = (float) allSelectedFilterResponses.at<double>(i,j);
    //cout<<"float result :"<< endl << floatAllResponses << endl;
    Mat kmeansResultCenters;
    Mat labels;
    TermCriteria criteria;
    criteria.epsilon = 0.01;
    double compactness = kmeans(floatAllResponses, K, labels, 
        criteria, 3, KMEANS_RANDOM_CENTERS, kmeansResultCenters );
    //cout<<"cluster after kmeans" << endl << kmeansResultCenters << endl;
    dictionary = Mat::zeros(K, numRes, CV_64F);
    for(int i = 0; i < K; i++)
        for(int j = 0; j < numRes; j++)
            dictionary.at<double>(i,j) = (double) kmeansResultCenters.at<float>(i,j);
}


void Dictionary::randAlpha(vector<int> &randomIndex, int N, int alpha) {
    for(int i = 0; i < N; i++)
        randomIndex.push_back(i);
    random_shuffle(randomIndex.begin(), randomIndex.end());

}
 
//defination of path input, for example: ../data/
//then the file will save as dictionary.xml
void Dictionary::save(const string& path) {
    String dictName = "dictionary.xml";
    String fullPath = path + dictName;
    FileStorage fs(fullPath, FileStorage::WRITE);  
    fs << "dictionary" << dictionary;  
    fs.release();
    return;

}

void Dictionary::load(const string& path) {
    FileStorage fs(path, FileStorage::READ);    
    fs["dictionary"] >> dictionary;
    fs.release();
    return;
}

int Dictionary::getWordsNum() {
    return dictionary.rows;
}

Mat Dictionary::getWordmap(const Mat& image, FilterBank& filterbank) {
    int numRows = image.rows;
    int numCols = image.cols;
    int allPixel = numRows * numCols; 
    Mat imageBackup = image;

    Mat Response; 
    filterbank.filter(imageBackup, Response);
    int numRes = Response.cols;

    Mat wordMap = Mat::zeros(numRows, numCols, CV_32S);

    for(int i = 0; i < numRows; i++)
        for(int j = 0; j < numCols; j++)
        {
            Mat oneResponse = Mat::zeros(1, numRes,CV_64F);

            for(int k = 0; k < numRes; k++)
                oneResponse.at<double>(0,k) = Response.at<double>(i*numCols+j,k);

            int mapping = nearestWord(oneResponse);

            wordMap.at<int>(i,j) = mapping;
        }
    
    return wordMap;
}

int Dictionary::nearestWord(Mat& oneResponse) {
    int K = dictionary.rows;
    int numCols = dictionary.cols;

    //initialize
    double distance = 0;
    int word = 0;
    for(int i = 0; i < numCols; i++)
    {
        double diff = oneResponse.at<double>(0,i) - dictionary.at<double>(0,i);
        distance += diff*diff;
    }

    for(int i = 1; i < K; i++)
    {
        double temp = 0;
        for(int j = 0; j < numCols; j++)
        {
            double diff = oneResponse.at<double>(0,j) - dictionary.at<double>(i,j);
            temp += diff*diff;
        }
        if(temp < distance) 
        {
            distance = temp;
            word = i;
        }

    }
    return word;
}

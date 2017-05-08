#include "bow.hpp"
#include "histogram.hpp"

#include <iostream>
#include <fstream>


/* Declaration of functions */
void help();
void readTestImagePaths(vector<string>& testImagesPath,
        const char *filename);
void readRealLabels(vector<int>& readLabels, const char *filename);
void readTrainingLabels(vector<int>& trainingLabels, const char *filename);
void readHistograms(Mat& H, const char *filename);
int knnClassify(Mat& testH, Mat& histograms, vector<int>& trainingLabels,
        int K);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        help();
        return -1;
    }

    string imageDir = "images/";

    vector<string> testImagesPath;
    vector<int> realLabels;
    vector<int> trainingLabels;
    Mat histograms;

    readTestImagePaths(testImagesPath, argv[1]);
    readRealLabels(realLabels, "test_label.txt");
    readTrainingLabels(trainingLabels, "training_label.txt");
    readHistograms(histograms, "histograms.xml");

    // Initializes filterbank.
    // The parameter must be the same as what was used in training phase.
    FilterBank filterbank;

    // Loads dictionary
    Dictionary dict;
    dict.load("dictionary/dictionary.xml");

    Mat cm = Mat::zeros(9, 9, CV_32S); // confusion matrix

    /* Evaluates classifier. Computes confusion matrix. */
    for (int i = 0; i < testImagesPath.size(); i++)
    {
        cout << "Testing image " << i+1 << "/" << testImagesPath.size();
        cout << endl;

        string& imagePath = testImagesPath[i];
        Mat image = imread(imageDir + imagePath);
        Mat wordmap = dict.getWordmap(image, filterbank);
        
        Mat h;
        computeHistogram(wordmap, h, dict.getWordsNum());

        // Predicts the label of the test image using knn. k = 5.
        int predictedLabel = knnClassify(h, histograms, trainingLabels, 5);
        int realLabel = realLabels[i];
        int &res = cm.at<int>(realLabel, predictedLabel);
        res = res + 1;
    }

    double tr = trace(cm)[0];
    double sumv = sum(cm)[0];
    cout << "Evaluation result (k = 5)\n";
    cout << "Confusion matrix:\n" << cm << endl;
    cout << "Accuracy: " << tr/sumv << endl;

    return 0;
}

void help()
{
    cout << "Usage: ./evaluate <test_set>\n";
    cout << "\t<test_set> is a txt file that contains the relative paths ";
    cout << "of all testing images.\n";
}

void readTestImagePaths(vector<string>& testImagesPath, const char *filename)
{
    ifstream in(filename);
    if (!in.is_open())
    {
        cout << "Error opening file\n";
        cout << "File " << filename << " may not exist.\n";
    }

    // Reads the paths of test images.
    string str;
    while (in >> str)
        testImagesPath.push_back(str);
    in.close();
}

void readRealLabels(vector<int>& realLabels, const char *filename)
{
    ifstream in(filename);
    if (!in.is_open())
    {
        cout << "Error opening file\n";
        cout << "File " << filename << "may not exist.\n";
    }
    int label;
    while (in >> label)
        realLabels.push_back(label);
    in.close();
}

void readTrainingLabels(vector<int>& trainingLabels, const char *filename)
{
    ifstream in(filename);
    if (!in.is_open())
    {
        cout << "Error opening file\n";
        cout << "File " << filename << "may not exist.\n";
    }
    int label;
    while (in >> label)
        trainingLabels.push_back(label);
    in.close();
}

void readHistograms(Mat& H, const char *filename)
{
    FileStorage fs(filename, FileStorage::READ);
    fs["histograms"] >> H;
}

int knnClassify(Mat& testH, Mat& histograms, vector<int>& trainingLabels,
        int K)
{
    Mat dist = distance(testH, histograms);
    // Sorts the distance array and gets the indices of the observations
    // that is closest to sample.
    Mat indices;
    cv::sortIdx(dist, indices, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);

    Mat counter = Mat::zeros(1, histograms.rows, CV_32S);
    for (int i = 0; i < K; i++)
    {
        int labelId = indices.at<int>(0,i);
        counter.at<int>(0,labelId) = counter.at<int>(0,labelId) + 1;
    }
    
    Point maxLoc;
    cv::minMaxLoc(counter, NULL, NULL, NULL, &maxLoc);
    return maxLoc.x;
}

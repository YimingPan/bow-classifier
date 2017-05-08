#include "bow.hpp"
#include "histogram.hpp"

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <omp.h>


/* Global variables. */
struct timeval tStart, cTime;

/* Declaration of functions. */
void help();

void tic();
time_t toc();

void computeWordmaps(vector<string>& trainingImagesPath, string& imageDir,
        string& targetDir, Dictionary& dictionary, FilterBank& filterbank);
void createHistograms(vector<string>& trainingImagesPath, int dictionarySize,
        string& targetDir);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        help();
        return -1;
    }

    ifstream in(argv[1]);
    string imageDir = "images/";
    string targetDir = "wordmaps/";
    vector<string> trainingImagesPath;

    if (!in.is_open())
    {
        cout << "Error opening file\n";
        cout << "File " << argv[1] << " may not exist.\n";
    }
    // Reads image paths.
    string str;
    while (in >> str)
    {
        trainingImagesPath.push_back(str);
    }

    cout << "Initializing filterbank ...\n";
    FilterBank filterbank;

    cout << "Computing dictionary ...\n";
    tic();
    Dictionary dict;
    int alpha = 50;
    int K = 150;
    dict.create(alpha, K, filterbank, trainingImagesPath, imageDir);
    cout << "Elapsed time(ms): " << toc() << endl;
    dict.save("dictionary/");

    //Dictionary dict;
    //dict.load("dictionary/dictionary.xml");
    cout << "Build word maps ...\n";
    tic();
    computeWordmaps(trainingImagesPath, imageDir, targetDir, dict, filterbank);
    cout << "Elapsed time(ms): " << toc() << endl;

    cout << "Create histograms ...\n";
    tic();
    createHistograms(trainingImagesPath, dict.getWordsNum(), targetDir);
    cout << "Elapsed time(ms): " << toc() << endl;

    return 0;
}

void help()
{
    cout << "Usage: ./train <training_set>\n";
    cout << "\t<training_set> is a txt file that contains the relative paths ";
    cout << "of all training images.\n";
}

/*
 * Starts a timer.
 */
void tic()
{
    gettimeofday(&tStart, NULL);
}

/*
 * Returns how many miliseconds has passed since last time tic() is called.
 */
time_t toc()
{
    gettimeofday(&cTime, NULL);
    cTime.tv_sec -= tStart.tv_sec;
    cTime.tv_usec -= tStart.tv_usec;
    return cTime.tv_sec*1000 + cTime.tv_usec/1000;
}

void computeWordmaps(vector<string>& trainingImagesPath, string& imageDir,
        string& targetDir, Dictionary& dictionary, FilterBank& filterbank)
{
    int N = trainingImagesPath.size(); // debug info

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        //cout << "Processing image " << i+1 << "/" << N << endl; // debug info

        string& imagePath = trainingImagesPath[i];
        Mat image = imread(imageDir + imagePath);
        Mat wordmap = dictionary.getWordmap(image, filterbank);

        // Constructs path for the xml file that stores the word map.
        string savepath = targetDir;
        savepath += imagePath.substr(0, imagePath.size()-3) + "xml";

        // Save wordmap to ./wordmaps/<category>/<imageName>.xml
        FileStorage fs(savepath, FileStorage::WRITE);
        fs << "wordmap" << wordmap;
        fs.release();
    }
}

/*
 * Computes the feature histograms of training images, puts them together to
 * form a big matrix, and saves it.
 * Each row of the matrix is the histogram of one image, so the number of row
 * equals to that of training images.
 */
void createHistograms(vector<string>& trainingImagesPath, int dictionarySize,
        string& targetDir)
{
    int numImages = trainingImagesPath.size();
    Mat histograms(numImages, dictionarySize, CV_64F);
    for (int i = 0; i < numImages; i++)
    {
        string& imagePath = trainingImagesPath[i];
        string savepath = targetDir + imagePath.substr(0, imagePath.size()-3)
                          + "xml";
        FileStorage fs(savepath, FileStorage::READ);
        Mat wordmap;
        fs["wordmap"] >> wordmap;

        Mat h;
        computeHistogram(wordmap, h, dictionarySize);
        h.copyTo(histograms.row(i));
    }
    FileStorage fs("histograms.xml", FileStorage::WRITE);
    fs << "histograms" << histograms;
}

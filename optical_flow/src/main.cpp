#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>
#include <dirent.h>                     // open dir

using namespace cv;
using namespace std;

int main()
{
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size winSize(31,31);


    string img_filepath = "../imgs/static_hopper/RGB/";

    if(!img_filepath.empty())
    {
        DIR *dir_imgs = opendir(img_filepath.c_str());
        std::vector<std::string> imageNames;
        struct dirent *ptr;

        while((ptr = readdir(dir_imgs)) != NULL)
        {
            if(ptr->d_name[0] == '.')
                continue;
            imageNames.push_back(ptr->d_name);
        }
        std::sort(imageNames.begin(),imageNames.end());

        namedWindow( "Hopper", 1 );
        cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();


        vector<Point2f> points[2];
        Mat gray, prevGray;
        for(int i = 0; i < imageNames.size(); i++)
        {
            std::vector<cv::KeyPoint> keypoints;

            Mat image = imread(img_filepath + imageNames[i], cv::IMREAD_COLOR);
            cvtColor(image, gray, COLOR_BGR2GRAY);
            detector->detect(gray, keypoints);

            if( !points[0].empty() )
            {
                vector<uchar> status;
                vector<float> err;
                if(prevGray.empty())
                    gray.copyTo(prevGray);

                calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                        7, termcrit, 0, 0.001);
                size_t j, k;
                for( j = k = 0; j < points[1].size(); j++ )
                {
                    if( !status[j] )
                        continue;

                    // if found, k is the real index of found points
                    points[1][k++] = points[1][j];
                    circle( image, points[0][j], 2, Scalar(0,0,255), -1, 8);
                    line( image, points[0][j], points[1][j], Scalar(0,255,0), 1, 8, 0);
                }
                points[1].resize(k);
            }
            else
            {
                for(int m = 0; m < keypoints.size(); m++)
                {
                    Point2f keypoint(keypoints[m].pt);
                    points[1].push_back(keypoint);
                }
            }
            for(int n = 0; n < points[1].size(); n++)
                circle( image, points[1][n], 2, Scalar(0,255,0), -1, 8);

            imshow("Hopper", image);
            waitKey(30);
            std::swap(points[1], points[0]);
            cv::swap(prevGray, gray);
        }
    }

    return 0;
}

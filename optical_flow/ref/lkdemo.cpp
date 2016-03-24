#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/features2d/features2d.hpp"

#include <iostream>
#include <ctype.h>
#include <dirent.h>                     // open dir

using namespace cv;
using namespace std;

static void help()
{
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
            "Using OpenCV version " << CV_VERSION << endl;
    cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
    cout << "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tr - auto-initialize tracking\n"
            "\tc - delete all the points\n"
            "\tn - switch the \"night\" mode on/off\n"
            "To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
    if( event == EVENT_LBUTTONDOWN )
    {
        point = Point2f((float)x, (float)y);
        addRemovePt = true;
    }
}

int main( int argc, char** argv )
{

    //  TermCriteria ( 	int  	type,
    //                  int  	maxCount,
    //                  double  	epsilon)
    // to fulfill one of the Criteria -- iteration count or accuracy
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 500;
    bool needToInit = false;
    bool nightMode = false;

    // "@input" for path of input video
    cv::CommandLineParser parser(argc, argv, "{@input||}{help h||}");
    string input = parser.get<string>("@input");
    if (parser.has("help"))
    {
        help();
        return 0;
    }

    // added by me, read image series from files
    string img_filepath = "../imgs/static_hopper/RGB/";

    if(img_filepath.empty())
    {

        VideoCapture cap;
        if( input.empty() )
            cap.open(0);
        else if( input.size() == 1 && isdigit(input[0]) )
            cap.open(input[0] - '0');
        else
            cap.open(input);

        if( !cap.isOpened() )
        {
            cout << "Could not initialize capturing...\n";
            return 0;
        }

        namedWindow( "LK Demo", 1 );
        setMouseCallback( "LK Demo", onMouse, 0 );

        // prevGray -- previous frame in gray
        Mat gray, prevGray, image, frame;
        vector<Point2f> points[2];          // 2 gourps of 2D points

        for(;;)
        {
            cap >> frame;
            if( frame.empty() )
                break;

            frame.copyTo(image);
            cvtColor(image, gray, COLOR_BGR2GRAY);

            if( nightMode ) // show a black scene instead
                image = Scalar::all(0);

            // automatically find some points to track
            if( needToInit )
            {
                // automatic initialization
                goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
                cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
                addRemovePt = false;
            }
            // if grounp 0 is not empty (the points from previous frame)
            else if( !points[0].empty() )
            {
                vector<uchar> status;
                vector<float> err;
                if(prevGray.empty())
                    gray.copyTo(prevGray);

                // Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.

                //            calcOpticalFlowPyrLK(InputArray prevImg,
                //                    InputArray nextImg,
                //                    InputArray prevPts,       // points need to be found
                //                    InputOutputArray nextPts, // found new position
                //                    OutputArray status,       //-- each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0
                //                    OutputArray err,
                //                    Size winSize=Size(21,21),
                //                    int maxLevel=3,
                //                    TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                //                    int flags=0,
                //                    double minEigThreshold=1e-4 )
                calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                        7, termcrit, 0, 0.001);
                size_t i, k;
                for( i = k = 0; i < points[1].size(); i++ )
                {
                    if( addRemovePt )
                    {
                        // already added, don't add it again!
                        if( norm(point - points[1][i]) <= 5 )
                        {
                            addRemovePt = false;
                            continue;
                        }
                    }

                    if( !status[i] )
                        continue;

                    // if found, k is the real index of found points
                    points[1][k++] = points[1][i];
                    circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
                }
                points[1].resize(k);
            }

            // if mouse was clicked and the group 1 is not full
            if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
            {
                vector<Point2f> tmp;
                tmp.push_back(point);   // point is global, assigned in onMouse

                // Refines the corner locations (which means that user gives a point, and this trying to find a corner point around the giving point, then return its location)
                // cornerSubPix(InputArray image, InputOutputArray corners, Size winSize, Size zeroZone, TermCriteria criteria)
                // corners – Initial coordinates of the input corners and refined coordinates provided for output.
                // 11x11 search window, no zero zone
                cornerSubPix( gray, tmp, winSize, Size(-1,-1), termcrit);
                points[1].push_back(tmp[0]);
                addRemovePt = false;
            }

            needToInit = false;
            imshow("LK Demo", image);

            char c = (char)waitKey(10);
            if( c == 27 )
                break;
            switch( c )
            {
            case 'r':
                needToInit = true;
                break;
            case 'c':
                points[0].clear();
                points[1].clear();
                break;
            case 'n':
                nightMode = !nightMode;
                break;
            }

            // put all points from group 1 to group 0
            std::swap(points[1], points[0]);
            // keep current gray frame as preGray for next frame
            cv::swap(prevGray, gray);
        }
    }
    else
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

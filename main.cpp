#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;
int main()
{
    VideoCapture capture("/home/yusuf/CLionProjects/opencv/EK000044.AVI");
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }
    Mat frame1, prvs;
    capture >> frame1;
    cvtColor(frame1, prvs, COLOR_BGR2GRAY);
    while(true){
        Mat frame2, next;
        capture >> frame2;
        Mat OrigFrame = frame2;
        if (frame2.empty())
            break;
        cvtColor(frame2, next, COLOR_BGR2GRAY);
        Mat flow(prvs.size(), CV_32FC2);
        calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        // visualization
        Mat flow_parts[2];
        split(flow, flow_parts);
        Mat magnitude, angle, magn_norm;
        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));
        //build hsv image
        Mat _hsv[3], hsv, hsv8, bgr;
        _hsv[0] = angle;
        _hsv[1] = Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magn_norm;
        merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);
        cvtColor(hsv8, bgr, COLOR_HSV2BGR);
//        ============================================================
        Mat canvas = Mat::zeros(bgr.rows, bgr.cols*2+10, bgr.type());

        bgr.copyTo(canvas(Range::all(), Range(0, frame2.cols)));
        frame2.copyTo(canvas(Range::all(), Range(frame2.cols+10, frame2.cols*2+10)));

// if it is too big to fit on the screen, then scale it down by 2, hopefully it'll fit :-)
        if(canvas.cols > 1920)
        {
            resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
        }

        imshow("canvas", canvas);
//         ============================================================
//        imshow("frame2", bgr);
        int keyboard = waitKey(3);
        if (keyboard == 'q' || keyboard == 27)
            break;
        prvs = next;
    }
}
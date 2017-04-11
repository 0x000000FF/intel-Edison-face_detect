#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <unistd.h>
// #include "mraa.hpp"

using namespace std;
using namespace cv;
vector<Rect> faces;
string face_cascade_name = "./haarcascade_frontalface_alt2.xml";
string cascadeName = "./haarcascade_frontalface_alt.xml";
string nestedCascadeName = "./haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier cascade, nestedCascade;

int detectAndDisplay(Mat frame);

int detectAndDraw( Mat& img, CascadeClassifier& cascade,CascadeClassifier& nestedCascade);

int open_camera();

int main(int argc, char** argv)
{
    // d_pin = new mraa::Gpio(13, true, false);
    // if (d_pin == NULL)
    // {
    //     cout << "Can't create mraa::Gpio object, exiting" << endl;
    //     return mraa::ERROR_UNSPECIFIED;
    // }

    // if (d_pin->dir(mraa::DIR_OUT) != mraa::SUCCESS)
    // {
    //     cout << "Can't set digital pin as output, exiting" << endl;
    //     return MRAA_ERROR_UNSPECIFIED;
    // }

    // Mat image;
    // image = imread("./mm.png", 1);

    // if (image.empty())
    // {
    //     cout<<"no image found!";
    // }

    if ( !nestedCascade.load( nestedCascadeName ) )
    {
        cout << "WARNING: Could not load classifier cascade for nested objects" << endl;
    }
    if( !cascade.load( cascadeName ) )
    {
        cout << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    if (!face_cascade.load(face_cascade_name))
    {
        cout<<"no hraa file found\n";
        return -1;
    }
    // detectAndDisplay(image);
    // detectAndDraw(image,cascade,nestedCascade);
    open_camera();
    return 1;
}

int open_camera()
{
    IplImage *frame = NULL;
    int num = 0;
    CvCapture *input_camera = cvCaptureFromCAM(-1);
    frame = cvQueryFrame(input_camera);
    Mat image(frame,0);
    int faces_num = 0;

    while(frame != NULL)
    {
        num++;
        cout<<num<<endl;
        frame = cvQueryFrame(input_camera);
        // if(num ==5)
        // {
        //     cvSaveImage("mm.jpg",frame);
        //     break;
        // }
        faces_num = detectAndDraw(image,cascade,nestedCascade);
        cout << "faces: " << faces_num << endl;
        if (faces_num > 0)
        {
            cvReleaseCapture(&input_camera);
            return faces_num;
        }
        if(num > 20)
        {
            cvReleaseCapture(&input_camera);
            return -1;
        }
    }
    cvReleaseCapture(&input_camera);
    return 0;
}
  
int detectAndDisplay(Mat face)
{
    Mat face_gray;
    cvtColor(face, face_gray, CV_BGR2GRAY);
    equalizeHist(face_gray, face_gray);
  
    face_cascade.detectMultiScale(face_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));  
    for (int i = 0; i <3; i++)
    {
        Point center(int(faces[i].x + faces[i].width*0.5), int(faces[i].y + faces[i].height*0.5));  
        ellipse(face, center, Size(int(faces[i].width*0.5), int(faces[i].height*0.5)), 0, 0, 360, Scalar(255, 0, 0), 4, 8, 0);  
    }
    imwrite("test.png",face);
    // d_pin->write(1);
    return 0;
}

int detectAndDraw( Mat& img, CascadeClassifier& cascade,CascadeClassifier& nestedCascade)
{
    double scale = 1.0;
    bool tryflip = false;
    double t = 0;

    vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );
    double fx = 1 / scale;
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)getTickCount();
    cascade.detectMultiScale( smallImg, faces,1.1, 2, 0|CASCADE_SCALE_IMAGE,Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,1.1, 2, 0|CASCADE_SCALE_IMAGE,Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    cout <<  "detection time = " << t*1000/getTickFrequency() << "ms \n";
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                       cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
        // if( nestedCascade.empty() )
        // {
        //     continue;
        // }
        // smallImgROI = smallImg( r );
        // nestedCascade.detectMultiScale( smallImgROI, nestedObjects,1.1, 2, 0|CASCADE_SCALE_IMAGE,Size(30, 30) );
        // for ( size_t j = 0; j < nestedObjects.size(); j++ )
        // {
        //     Rect nr = nestedObjects[j];
        //     center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
        //     center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
        //     radius = cvRound((nr.width + nr.height)*0.25*scale);
        //     circle( img, center, radius, color, 3, 8, 0 );
        // }
    }

    imwrite("test.png",img);
    return faces.size();
}
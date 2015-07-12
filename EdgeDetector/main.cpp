#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;
Mat faceROI;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

String face_cascade_name = "../haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "../haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
RNG rng(12345);
std::vector<std::vector<Point> > contours;
std::vector<Vec4i> hierarchy;

Rect face, eye_1, eye_2;

Mat imagenOriginal, imageResized;

int mousex, mousey;

Mat zoomIn(int x, int y)
{
	int width = imagenOriginal.size().width / 1.5, height = imagenOriginal.size().height / 1.5;
	int ptoX = x - (width / 2), ptoY = y - (height / 2);

	//Check if zoom inside boundaries
	if ((x + (width / 2)) > dst.size().width)
		ptoX = dst.size().width-width;

	if ((y + (height / 2)) > dst.size().height)
		ptoY = dst.size().height-height;

	if ((x - (width / 2)) < 0)
		ptoX = 0;

	if ((y - (height / 2)) < 0)
		ptoY = 0;

	Rect roi = Rect(ptoX, ptoY, width, height);
	Mat imagen_roi = dst(roi);
	resize(imagen_roi, imagen_roi, Size(imagenOriginal.size().width, imagenOriginal.size().height), 0, 0, CV_INTER_AREA);

	return imagen_roi;
}

Mat zoomOut(int x, int y)
{
	return imagenOriginal;
}

static void onMouse(int event, int x, int y, int , void*)
{
	mousex = x;
	mousey = y;

	if (event == CV_EVENT_LBUTTONDOWN)
		dst = zoomIn(x, y);
	else if (event == CV_EVENT_RBUTTONDOWN)
		dst = zoomOut(x, y);
}

bool detectFaceAndEyes(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	if (faces.size()){
		std::cout << "HEYHEY " << faces.size() << std::endl;
		for (int i = 0; i < faces.size(); i++){
			faceROI = frame_gray(faces[i]);
			std::vector<Rect> eyes;

			eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

			if (eyes.size() > 1){
				std::cout << "HEYHO " << eyes.size() << std::endl;

				face = faces[0];
				eye_1 = eyes[0];
				eye_2 = eyes[1];
			}
		}
	}

	return false;
}

void drawFaceAndEyes(Mat frame){
	if (face.width > 0 && eye_1.width > 0 && eye_2.width > 0){
		Point center(face.x + face.width*0.5, face.y + face.height*0.5);
		ellipse(frame, center, Size(face.width*0.5, face.height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		Point center_face(face.x + face.width*0.5, face.y + face.height*0.5);
		ellipse(frame, center_face, Size(face.width*0.02, face.height*0.02), 0, 0, 360, Scalar(255, 0, 255), -1, 8, 0);

		Point center_1(face.x + eye_1.x + eye_1.width*0.5, face.y + eye_1.y + eye_1.height*0.5);
		int radius = cvRound((eye_1.width + eye_1.height)*0.25);
		circle(frame, center_1, radius, Scalar(255, 0, 0), 4, 8, 0);

		Point center_2(face.x + eye_2.x + eye_2.width*0.5, face.y + eye_2.y + eye_2.height*0.5);
		int radius_2 = cvRound((eye_2.width + eye_2.height)*0.25);
		circle(frame, center_2, radius, Scalar(255, 0, 0), 4, 8, 0);

		Point eye_center = (center_2 + center_1)/2.f;
		ellipse(frame, eye_center, Size(face.width*0.02, face.height*0.02), 0, 0, 360, Scalar(255, 0, 255), -1, 8, 0);

		//std::cout << "Coef: " << (eye_center.x - center_face.x)/(eye_center.y - center_face.y) << std::endl;

		line(frame, center_face, eye_center, Scalar(255, 0, 0), 4, 8, 0);
	}
}

void drawScene(){
	dst.copyTo(imageResized);
	
	Mat drawing = Mat::zeros(imageResized.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point(face.x, face.y));
	}

	drawFaceAndEyes(drawing);
	imshow(window_name, drawing);
}

/**
* @function CannyThreshold
* @brief Trackbar callback - Canny thresholds input with a ratio 1:3
*/
void CannyThreshold(int, void*)
{
	/// Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3, 3));

	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	imagenOriginal = dst = Scalar::all(0);

	src.copyTo(dst, detected_edges);

	detected_edges = detected_edges(face);

	findContours(detected_edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	drawScene();
	//imshow(window_name, dst);
}

/** @function main */
int main(int argc, char** argv)
{
	char* img_name = "../glasses_model_13.jpg";

	CvCapture* capture;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading\n"); return -1; };

	/// Load an image
	//src = imread(argv[1]);
	imagenOriginal = src = imread(img_name);

	if (!src.data)
	{
		std::cout << "No image found!" << std::endl;
		return -1;
	}

	/// Create a matrix of the same type and size as src (for dst)
	dst.create(src.size(), src.type());

	/// Convert the image to grayscale
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Create a window
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create a Trackbar for user to enter threshold
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

	detectFaceAndEyes(imagenOriginal);

	/// Show the image
	CannyThreshold(0, 0);

	/*setMouseCallback(window_name, onMouse, 0);
	do
	{

		dst.copyTo(imageResized);

		rectangle(imageResized,
			Point(mousex - (imagenOriginal.size().width / 1.5 / 2), mousey - (imagenOriginal.size().height / 1.5 / 2)),
			Point(mousex + (imagenOriginal.size().width / 1.5 / 2), mousey + (imagenOriginal.size().height / 1.5 / 2)),
			cv::Scalar(0, 255, 0), 1, 8, 0);
		Mat drawing = Mat::zeros(imageResized.size(), CV_8UC3);
		for (int i = 0; i< contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		}
		drawFaceAndEyes(drawing);
		imshow(window_name, drawing);

		char c = (char)waitKey(10);
		if (c == 27)
			break;
	} while (true);*/

	/// Wait until user exit program by pressing a key
	waitKey(0);

	return 0;
}
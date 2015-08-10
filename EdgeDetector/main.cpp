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
int lowThreshold = 100, lowSThreshold = 80;
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
std::vector<std::vector<Point> >hull;
std::vector<std::vector<Point> >potential_hull;
std::vector<std::vector<Point> >big_hull;
std::vector<Rect> bd_rects;
std::vector<Vec4i> hierarchy;

char* img_name = "../glasses_model_15.jpg";

Rect face, eye_1, eye_2, glasses_bb;

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

void clearSmallContours(float max_width, float max_height, float eye_height){
	if (contours.size() < 1)
		return;

	int last_index = contours.size() - 1;

	bd_rects.clear();
	Rect bd_rect;

	Point center_eye_1(face.x + eye_1.x + eye_1.width*0.5f, face.y + eye_1.y + eye_1.height*0.5f);
	Point center_eye_2(face.x + eye_2.x + eye_2.width*0.5f, face.y + eye_2.y + eye_2.height*0.5f);

	for (int i = 0; i < contours.size(); i++){
		if (i == last_index){
			break;
		}

		bd_rect = boundingRect(Mat(contours[i]));
		//bd_rect.x += glasses_bb.x;
		//bd_rect.y += glasses_bb.y;

		if (bd_rect.height > max_height*0.75 || (bd_rect.width < max_width)){
			std::swap(contours[i], contours[last_index]);
			last_index--;
			i--;
		}else{
			bd_rects.push_back(bd_rect);
		}
	}

	int nb_of_pops = contours.size() - last_index;

	for (int i = 0; i < nb_of_pops; i++){
		contours.pop_back();
	}
}

void selectBiggestContour(){
	if (contours.size() < 2)
		return;
	
	Rect bd_rect;

	bd_rect = boundingRect(Mat(contours[0]));

	float max_width = bd_rect.width, max_height = bd_rect.height, big_index = 0;

	for (int i = 1; i < contours.size(); i++){

		bd_rect = boundingRect(Mat(contours[i]));

		if (bd_rect.height > max_height && bd_rect.width > max_width){
			big_index = i;
			max_height = bd_rect.height;
			max_width = bd_rect.width;
		}
	}

	std::swap(contours[big_index], contours[0]);

	while (contours.size() > 1)
		contours.pop_back();

}

void selectBiggestConvex(){
	if (potential_hull.size() < 2){
		if (potential_hull.size() > 0){
			big_hull.push_back(potential_hull[0]);
		}
		return;
	}

	big_hull.clear();
	double max_area = 0;
	int big_index = 0;

	std::cout << "Nb of hulls: " << potential_hull.size() << std::endl;

	for (int j = 0; j < potential_hull.size(); j++){
		double area = 0;
		for (int i = 0; i < potential_hull[j].size(); i++){
			int next_i = (i + 1) % (potential_hull[j].size());
			double dX = potential_hull[j][next_i].x - potential_hull[j][i].x;
			double avgY = (potential_hull[j][next_i].y + potential_hull[j][i].y) / 2.f;
			area += dX*avgY;  // This is the integration step.
		}
		area = abs(area);
		std::cout << area << std::endl;
		if (area > max_area){
			big_index = j;
			max_area = area;
		}
	}

	std::cout << "Big Index: " << big_index << std::endl;
	big_hull.push_back(potential_hull[big_index]);
	//if (big_index != 0)
	//	std::swap(hull[big_index], hull[0]);*/
}

void selectPotentialConvex(){
	std::cout << hull.size() << std::endl;
	if (hull.size() < 2){
		if (hull.size() > 0){
			potential_hull.push_back(hull[0]);
		}
		return;
	}
	double max_area = 0;
	int big_index = 0;

	for (int j = 0; j < hull.size(); j++){
		double area = 0;
		for (int i = 0; i < hull[j].size(); i++){
			int next_i = (i + 1) % (hull[j].size());
			double dX = hull[j][next_i].x - hull[j][i].x;
			double avgY = (hull[j][next_i].y + hull[j][i].y) / 2.f;
			area += dX*avgY;  // This is the integration step.
		}
		area = abs(area);
		std::cout << area << std::endl;
		if (area > max_area){
			big_index = j;
			max_area = area;
		}
	}

	std::cout << "Big Index: " << big_index << std::endl;
	potential_hull.push_back(hull[big_index]);
	//if (big_index != 0)
	//	std::swap(hull[big_index], hull[0]);*/
}

void clearContoursNotContainingEyes(){
	if (contours.size() < 1)
		return;

	int last_index = contours.size() - 1;

	bd_rects.clear();
	Rect bd_rect;

	Point center_eye_1(face.x + eye_1.x + eye_1.width*0.5f, face.y + eye_1.y + eye_1.height*0.5f);
	Point center_eye_2(face.x + eye_2.x + eye_2.width*0.5f, face.y + eye_2.y + eye_2.height*0.5f);

	for (int i = 0; i < contours.size(); i++){
		bd_rect = boundingRect(Mat(contours[i]));

		if ((bd_rect.contains(center_eye_1) || bd_rect.contains(center_eye_2)) && !(bd_rect.contains(center_eye_1) && bd_rect.contains(center_eye_2))){
			bd_rects.push_back(bd_rect);
		}else{
			std::swap(contours[i], contours.back());
			contours.pop_back();
			i--;
		}
	}
}

Point projectPoint(Point point, Point vecOrigin){
	Vec2f v1(point.x - vecOrigin.x, point.y - vecOrigin.y);
	//Vec2f vecdiff = 2.f*axis.dot(v1.dot(axis));

	Point result = Point(vecOrigin.x * 2 - point.x, point.y);

	return result;
}

bool hasClosePoint(Point point, float threshold){
	for (int i = 0; i < contours.size(); i++){
		for (int j = 0; j < contours[i].size(); j++){
			//std::cout << norm(contours[i][j] - point)<<std::endl;
			if (norm(contours[i][j] - point) < threshold)
				return true;
		}
	}
	return false;
}


void clearUnreflectedPoints(float threshold){
	Point reflectedPoint;

	Point axisEnd(face.x + face.width*0.5, face.y + face.height);
	Point axisOrigin(face.x + face.width*0.5, face.y);

	Vec2f axis(axisEnd.x - axisOrigin.x, axisEnd.y - axisOrigin.y);

	for (int i = 0; i < contours.size(); i++){
		for (int j = 0; j < contours[i].size(); j++){
			reflectedPoint = projectPoint(contours[i][j], axisOrigin);
			if (!hasClosePoint(reflectedPoint, threshold)){
				std::swap(contours[i][j], contours[i].back());
				contours[i].pop_back();
				j--;
			}
		}
	}
}

void clearUnreflectedContours(float threshold, float minSimilarity){
	Point reflectedPoint;

	Point axisEnd(face.x + face.width*0.5, face.y + face.height);
	Point axisOrigin(face.x + face.width*0.5, face.y);

	Vec2f axis(axisEnd.x - axisOrigin.x, axisEnd.y - axisOrigin.y);

	float numOfPoints = 0, numOfSharedPoints = 0;
	float similarity = 0;

	int last_index = contours.size();

	for (int i = 0; i < last_index; i++){
		numOfPoints = contours[i].size();
		numOfSharedPoints = 0;
		for (int j = 0; j < contours[i].size(); j++){
			reflectedPoint = projectPoint(contours[i][j], axisOrigin);
			if (hasClosePoint(reflectedPoint, threshold)){
				numOfSharedPoints++;
			}
		}

		similarity = numOfSharedPoints / numOfPoints;
		if (similarity < minSimilarity){
			std::swap(contours[i], contours[last_index-1]);
			last_index--;
			i--;
		}
	}

	int nbOfPop = contours.size() - last_index;

	for (int i = 0; i < nbOfPop; i++){
		contours.pop_back();
	}

	/*Point reflectedPoint;

	Point axisEnd(face.x + face.width*0.5, face.y + face.height);
	Point axisOrigin(face.x + face.width*0.5, face.y);

	Vec2f axis(axisEnd.x - axisOrigin.x, axisEnd.y - axisOrigin.y);

	float numOfPoints = 0, numOfSharedPoints = 0;
	float similarity = 0;

	for (int i = 0; i < contours.size(); i++){
		numOfPoints = contours[i].size();
		numOfSharedPoints = 0;
		for (int j = 0; j < contours[i].size(); j++){
			reflectedPoint = projectPoint(contours[i][j], axisOrigin);
			if (hasClosePoint(reflectedPoint, threshold)){
				numOfSharedPoints++;
			}
		}

		similarity = numOfSharedPoints / numOfPoints;
		if (similarity < minSimilarity){
			std::swap(contours[i], contours.back());
			contours.pop_back();
			i--;
		}
	}*/
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

				glasses_bb = Rect(face.x, face.y + eye_1.y - eye_1.y*0.1, face.width, eye_1.height*1.9);

				return true;
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

		/*Point center_1(face.x + eye_1.x + eye_1.width*0.5, face.y + eye_1.y + eye_1.height*0.5);
		int radius = cvRound((eye_1.width + eye_1.height)*0.4);
		circle(frame, center_1, radius, Scalar(255, 0, 0), 2, 8, 0);

		Point center_2(face.x + eye_2.x + eye_2.width*0.5, face.y + eye_2.y + eye_2.height*0.5);
		int radius_2 = cvRound((eye_2.width + eye_2.height)*0.4);
		circle(frame, center_2, radius, Scalar(255, 0, 0), 2, 8, 0);*/

		Rect eye_1_rect(face.x + eye_1.x, face.y + eye_1.y, eye_1.width, eye_1.height);
		Rect eye_2_rect(face.x + eye_2.x, face.y + eye_2.y, eye_2.width, eye_2.height);

		//rectangle(frame, eye_1_rect, Scalar(0, 0, 255), 2, 8, 0);
		//rectangle(frame, eye_2_rect, Scalar(0, 0, 255), 2, 8, 0);

		rectangle(frame, glasses_bb, Scalar(0, 255, 0), 2, 8, 0);

		//Point eye_center = (center_2 + center_1)/2.f;
		//ellipse(frame, eye_center, Size(face.width*0.02, face.height*0.02), 0, 0, 360, Scalar(255, 0, 255), -1, 8, 0);

		//std::cout << "Coef: " << (eye_center.x - center_face.x)/(eye_center.y - center_face.y) << std::endl;

		Point pt1(face.x + face.width*0.5, face.y + face.height);
		Point pt2(face.x + face.width*0.5, face.y);

		line(frame, pt1, pt2, Scalar(255, 0, 0), 2, 8, 0);
	}
}

void drawScene(){
	dst.copyTo(imageResized);
	
	//Mat drawing = Mat::zeros(imageResized.size(), CV_8UC3);
	Mat drawing = imread(img_name);
	///Mat drawing(src);

	if (big_hull.size() > 0){
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, big_hull, 0, color, 2, 8, std::vector<Vec4i>(), 0, Point());
	}

	/*for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point(0,0));
		drawContours(drawing, hull, i, color, 2, 8, std::vector<Vec4i>(), 0, Point());
		//for (int j = 0; j < contours[i].size(); j++){
		//	ellipse(drawing, contours[i][j], Size(face.width*0.005, face.height*0.005), 0, 0, 360, Scalar(255, 255, 0), -1, 8, 0);
		//}
	}*/


	/*for (int i = 0; i < bd_rects.size(); i++){
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		rectangle(drawing, bd_rects[i], color, 2, 8, 0);
	}*/

	drawFaceAndEyes(drawing);
	imshow(window_name, drawing);
}

void getConvexHulls(){
	hull.clear();
	hull.resize(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		convexHull(Mat(contours[i]), hull[i], false);
	}
}

void setupScene(){
	findContours(detected_edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(glasses_bb.x, glasses_bb.y));
	clearUnreflectedContours(5.f, lowSThreshold / 100.f);
	clearContoursNotContainingEyes();
	//clearSmallContours(eye_1.width, glasses_bb.height, eye_1.height);
	//selectBiggestContour();
	getConvexHulls();
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

	detected_edges = detected_edges(glasses_bb);

	setupScene();
	drawScene();
}

void SimilarityThreshold(int, void*)
{
	setupScene();
	drawScene();
}

void searchForGlassesFrame(){
	bool foundFrame = false;
	
	while (lowThreshold > 1){
		//lowSThreshold -= 5;
		lowThreshold -= 5;
		CannyThreshold(0, 0);
		//setupScene();
		selectPotentialConvex();
		//std::cout << lowSThreshold << std::endl;
		std::cout << lowThreshold << std::endl;
	}
	selectBiggestConvex();
	drawScene();
}

/** @function main */
int main(int argc, char** argv)
{
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
	createTrackbar("Canny Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
	createTrackbar("Similarity Threshold:", window_name, &lowSThreshold, max_lowThreshold, SimilarityThreshold);
	//createButton("Tick", CheckBox, NULL, CV_CHECKBOX, 0);

	detectFaceAndEyes(imagenOriginal);

	/// Show the image
	CannyThreshold(0, 0);

	searchForGlassesFrame();

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
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <conio.h>
#include <ctype.h>
#include <Windows.h>
#include <sstream>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>



#include "opencv2/videoio.hpp"

typedef std::vector<std::string> stringvec;

using namespace cv;
using namespace std;

Mat mask2;




















/* 코드 추가(거리 도출) */

/* 차 브레이크 등만 남기기 위한 필터 생성부 */
// 입력은 원본 이미지, 반환값은 필터 영역
cv::Mat make_filter(cv::Mat img)
{
	cv::cvtColor(img, img, COLOR_BGR2GRAY);
	cv::Mat filter = cv::Mat::zeros(img.size(), CV_8UC1);
	int width = filter.cols;
	int height = filter.rows;
	for (int y = 0; y < height; y++)
	{
		uchar* filter_data = filter.data;
		for (int x = 0; x < width; x++)
		{
			//필터 위치 조작부
			if ((((y >= (int)(height / 4) && y <= (int)((height) / 2)))) && // 위에서 1/4~절반 되는 지점이면서
				(((x >= (int)(width / 10)) && (x <= (int)(width * 1 / 4))) || // 폭이 왼쪽에서 1/10~1/4인 지점이거나
				(x >= (int)((width * 3) / 4) && (x <= ((int)(width * 9) / 10))))) // 폭이 왼쪽에서 3/4~9/10인 지점만 필터링하는 필터
			{
				filter_data[(y * width) + x] = (uchar)255;
			}
		}
	}
	cv::cvtColor(filter, filter, COLOR_GRAY2BGR);
	//cv::imshow("필터", filter);
	return filter;
}

/* 차 후면 필터링 */
// 입력은 가우시안 블러링 된 차 사진의 컬러 이미지, 반환값은 필터링된 자동차 브레이크 등 부분 이미지
cv::Mat filtering(cv::Mat img, cv::Mat filter)
{
	img = img & filter;
	return img;
}

/* 브레이크 등의 좌우측을 나누기 위한 함수부 */
// 좌측면은 flag = 1, 우측면은 flag = 2
// 입력은 색공간으로 걸러진 전체 브레이크 등 이미지, 반환은 좌우로 분할된 브레이크 이미지
cv::Mat saperate_brake(cv::Mat img, int flag)
{
	cv::Mat saperated_img = cv::Mat::zeros(img.size(), CV_8UC1);
	int width = saperated_img.cols;
	int height = saperated_img.rows;

	for (int y = 0; y < height; y++)
	{
		uchar* filtered_data = img.data;
		uchar* saperated_data = saperated_img.data;
		for (int x = 0; x < width; x++)
		{
			if (flag == 1) // 좌측면 분할이면-
			{
				if (x <= (int)(width / 2))
				{
					saperated_data[(y * width) + x] = filtered_data[(y * width) + x];
				}
			}

			if (flag == 2) // 우측면 분할이면-
			{
				if (x >= (int)(width / 2))
				{
					saperated_data[(y * width) + x] = filtered_data[(y * width) + x];
				}
			}

		}
	}
	return saperated_img;
}


/* 색공간으로 2차로 걸러진 브레이크등에서 모폴로지 처리해, 추측되는 최종 브레이크 영역만 최대한 남기는 함수부 */
// 입력은 좌우측으로 나눠진 브레이크 이미지, 반환은 추측된 브레이크 영역 이미지

cv::Mat last_filtering(cv::Mat left_or_right)
{
	//모폴로지 연산으로 열림, 닫힘 연산해 빈 부분 메꿈
	cv::Mat element5(5, 5, CV_8U, cv::Scalar(1)); // 모폴로지 커널
	cv::morphologyEx(left_or_right, left_or_right, cv::MORPH_CLOSE, element5); // 닫고
	cv::morphologyEx(left_or_right, left_or_right, cv::MORPH_OPEN, element5); // 열음
																			  //cv::imshow("모폴로지 연산처리 한 브레이크 영역", left_or_right);
																			  //cv::waitKey(0);

																			  // 아랫부분은 컨투어 따 처리하려 했으나, 모폴로지만으로도 거리 산출 충분할 것 같아 진행하지 않았음
																			  // 필요한 경우 이 부분 추가/수정
																			  /*
																			  std::vector<std::vector<cv::Point>> contours;
																			  cv::findContours(left_or_right, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

																			  cv::drawContours(left_or_right, left_or_right, -1, cv::Scalar(255), 2);
																			  cv::imshow("컨투어 처리한 브레이크 영역", left_or_right);
																			  */
	return left_or_right;
}

/* 좌우 브레이크 등 이미지에서 평균 좌표값 찾아내는 함수부 */
// 입력 영상은 좌측 또는 우측 브레이크 등 이미지, 반환은 영상의 y 픽셀 평균값
// 만약 입력 영상 내 브레이크로 판단되는 부분이 없다면, 없다는 flag 정수값을 반환
int cal_average_pixel(cv::Mat left_or_right)
{
	int count = 0;
	int total_x = 0;
	int total_y = 0;
	int avg_x, avg_y; // x는 좌표 확인용
	int width = left_or_right.cols;
	int height = left_or_right.rows;

	for (int y = 0; y < height; y++)
	{
		uchar* data = left_or_right.data;
		for (int x = 0; x < width; x++)
		{
			if (data[(y * width) + x] != ((uchar)0))
			{
				total_x += x;
				total_y += y;
				count += 1;
			}

		}
	}

	if (count != 0)
	{
		avg_x = (int)(total_x / count);
		avg_y = (int)(total_y / count);
		std::cout << "각각의 평균 x,y값 : " << "( " << avg_x << ", " << avg_y << " )" << std::endl;
		return avg_y;
	}
	if (count == 0)
	{
		std::cout << "브레이크로 판단되는 픽셀이 없습니다. 차의 중심값으로 설정합니다." << std::endl;
		return 0;
	}
}

/* 차 영상에서 브레이크 정중앙 픽셀위치를 찾아내는 함수부 */
// 입력은 좌우 브레이크 등의 최종 걸러진 이미지, 반환값은 브레이크 등의 전체 평균 y좌표
int cal_center_pixel(cv::Mat left, cv::Mat right)
{
	int left_avg_y_point = cal_average_pixel(left);
	int right_avg_y_point = cal_average_pixel(right);

	if (left_avg_y_point != 0 && right_avg_y_point != 0)
	{
		int total_y_center = (int)((left_avg_y_point + right_avg_y_point) / 2);
		std::cout << "전체의 평균 y값 : " << total_y_center << std::endl;
		return total_y_center;
	}
	else
	{
		return 0;
	}
}

/* 입력받은 이미지에서 주어진 좌표값에 따른 차만 골라내는 함수부 */
// 입력은 컬러 원본 영상/차량 x1/차량 x2/차량 y1/차량 y2, 반환은 걸러진 차량 컬러이미지
cv::Mat filtering_car(cv::Mat Origin, int car_x1, int car_x2, int car_y1, int car_y2)
{
	int car_width = car_x2 - car_x1;
	int car_height = car_y2 - car_y1;
	cv::Rect ROI(car_x1, car_y1, car_width, car_height);
	cv::Mat car_img = Origin(ROI).clone();

	return car_img;
}

/* 거리 도출부 끝 */









/* 차선 검출을 위한 생성자 클래스 */
class LaneDetector {
private:
	double img_size;
	double img_center;
	bool left_flag = false;  // Tells us if there's left boundary of lane detected
	bool right_flag = false;  // Tells us if there's right boundary of lane detected
	cv::Point right_b;  // Members of both line equations of the lane boundaries:
	double right_m;  // y = m*x + b
	cv::Point left_b;  //
	double left_m;  //

public:
	cv::Mat deNoise(cv::Mat inputImage);  // Apply Gaussian blurring to the input Image
	cv::Mat edgeDetector(cv::Mat img_noise);  // Filter the image to obtain only edges
	cv::Mat mask(cv::Mat img_edges);  // Mask the edges image to only care about ROI
	std::vector<cv::Vec4i> houghLines(cv::Mat img_mask);  // Detect Hough lines in masked edges image
	std::vector<std::vector<cv::Vec4i> > lineSeparation(std::vector<cv::Vec4i> lines, cv::Mat img_edges);  // Sprt detected lines by their slope into right and left lines
	std::vector<cv::Point> regression(std::vector<std::vector<cv::Vec4i> > left_right_lines, cv::Mat inputImage);  // Get only one line for each side of the lane
	std::string predictTurn();  // Determine if the lane is turning or not by calculating the position of the vanishing point
	int plotLane(cv::Mat inputImage, std::vector<cv::Point> lane, std::string turn);  // Plot the resultant lane and turn prediction in the frame.
};


/* 이미지 블러처리 및 잡음 제거 함수 */
// 이미지 가우시안 필터처리
// 블러처리되고 잡음 제거된 영상 반환
cv::Mat LaneDetector::deNoise(cv::Mat inputImage) {
	cv::Mat output;

	cv::GaussianBlur(inputImage, output, cv::Size(3, 3), 0, 0);

	return output;
}

/* 엣지 검출 함수 */
//이미지 내 모든 엣지 검출
// 흑백처리 후 엣지 찾아낸 이진화사진 반환
cv::Mat LaneDetector::edgeDetector(cv::Mat img_noise) {
	cv::Mat output;
	cv::Mat kernel;
	cv::Point anchor;

	// 컬러이미지를 흑백처리
	cv::cvtColor(img_noise, output, cv::COLOR_RGB2GRAY);
	// 이미지 이진화
	/*cv::threshold(output, output, 140, 255, cv::THRESH_BINARY);

	// 다음의 커널을 생성 : [-1 0 1]
	// This kernel is based on the one found in the
	// Lane Departure Warning System by Mathworks
	anchor = cv::Point(-1, -1);
	kernel = cv::Mat(1, 3, CV_32F);
	kernel.at<float>(0, 0) = -1;
	kernel.at<float>(0, 1) = 0;
	kernel.at<float>(0, 2) = 1;

	// 커널(필터) 적용해서 엣지찾아냄
	cv::filter2D(output, output, -1, kernel, anchor, 0, cv::BORDER_DEFAULT);
	*/
	cv::Mat cannyimg(img_noise.size(), CV_8UC1, cv::Scalar(0, 0, 0));
	cv::Canny(img_noise, cannyimg, 50, 150, 3);
	cv::imshow("cannyimg", cannyimg);
	return cannyimg;
}

/* 엣지 찾은 이미지 마스킹 */
// ROI 설정해 차선 엣지만 걸러냄
// 원하는 엣지(차선)만 나타낸 영상 반환 
cv::Mat LaneDetector::mask(cv::Mat img_edges) {
	cv::Mat output;
	cv::Mat mask = cv::Mat::zeros(img_edges.size(), img_edges.type());
	cv::Point pts[4] = {		//영상 ROI
								//	cv::Point(500, mask.rows),
								//	cv::Point(mask.cols-500, mask.rows-100),
								//	cv::Point((int)(mask.cols / 2) + 50, (int)(mask.rows / 2) + 230),
								//	cv::Point((int)(mask.cols / 2) - 50, (int)(mask.rows / 2) + 230)
		cv::Point((int)(mask.cols*0.15), (int)(mask.rows*0.9)), // 좌측 하단
		cv::Point((int)(mask.cols*0.85), (int)(mask.rows*0.9)), // 우측 하단
		cv::Point((int)((mask.cols / 2) + (mask.cols*0.1)), (int)((mask.rows / 2) + (mask.rows*0.05))), // 우측 상단
		cv::Point((int)((mask.cols / 2) - (mask.cols*0.1)), (int)((mask.rows / 2) + (mask.rows*0.05))) //좌측 상단
	};

	// 이진 폴리곤 마스크 생성
	cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255, 0, 0));
	// ROI와 찾아낸 모든 엣지를 서로 AND 비트마스킹
	mask2 = mask;
	cv::bitwise_and(img_edges, mask, output);

	return output;
}

/* 허프라인 검출 함수 */
// 허프 검출방식을 이용해, 차선 라인이라고 생각되는 부분의 좌표를 반환
std::vector<cv::Vec4i> LaneDetector::houghLines(cv::Mat img_mask) {
	std::vector<cv::Vec4i> line;

	// rho and theta are selected by trial and error
	HoughLinesP(img_mask, line, 1, CV_PI / 180, 20, 20, 30);

	return line;
}

// SORT RIGHT AND LEFT LINES
/**
*@brief Sort all the detected Hough lines by slope.
*@brief The lines are classified into right or left depending
*@brief on the sign of their slope and their approximate location
*@param lines is the vector that contains all the detected lines
*@param img_edges is used for determining the image center
*@return The output is a vector(2) that contains all the classified lines
*/
std::vector<std::vector<cv::Vec4i> > LaneDetector::lineSeparation(std::vector<cv::Vec4i> lines, cv::Mat img_edges) {
	std::vector<std::vector<cv::Vec4i> > output(2);
	size_t j = 0;
	cv::Point ini;
	cv::Point fini;
	double slope_thresh = 0.3;
	std::vector<double> slopes;
	std::vector<cv::Vec4i> selected_lines;
	std::vector<cv::Vec4i> right_lines, left_lines;

	// Calculate the slope of all the detected lines
	for (auto i : lines) { // 이건 라인하나다. 라인하나에 좌표값이 두개. 즉 그냥 상수값은 4개가 있는 것. 계속 햇갈리지말고하
		ini = cv::Point(i[0], i[1]);
		fini = cv::Point(i[2], i[3]);

		// Basic algebra: slope = (y1 - y0)/(x1 - x0)
		double slope = (static_cast<double>(fini.y) - static_cast<double>(ini.y)) / (static_cast<double>(fini.x) - static_cast<double>(ini.x) + 0.00001);

		// If the slope is too horizontal, discard the line
		// If not, save them  and their respective slope
		if (std::abs(slope) > slope_thresh) {
			slopes.push_back(slope);
			selected_lines.push_back(i);
		}
	}

	// Split the lines into right and left lines
	img_center = static_cast<double>((img_edges.cols / 2));
	while (j < selected_lines.size()) {
		ini = cv::Point(selected_lines[j][0], selected_lines[j][1]);
		fini = cv::Point(selected_lines[j][2], selected_lines[j][3]);

		// Condition to classify line as left side or right side
		if (slopes[j] > 0 && fini.x > img_center && ini.x > img_center) {
			right_lines.push_back(selected_lines[j]);
			right_flag = true;
		}
		else if (slopes[j] < 0 && fini.x < img_center && ini.x < img_center) {
			left_lines.push_back(selected_lines[j]);
			left_flag = true;
		}
		j++;
	}

	output[0] = right_lines;
	output[1] = left_lines;

	return output;
}

// REGRESSION FOR LEFT AND RIGHT LINES
/**
*@brief Regression takes all the classified line segments initial and final points and fits a new lines out of them using the method of least squares.
*@brief This is done for both sides, left and right.
*@param left_right_lines is the output of the lineSeparation function
*@param inputImage is used to select where do the lines will end
*@return output contains the initial and final points of both lane boundary lines
*/
std::vector<cv::Point> LaneDetector::regression(std::vector<std::vector<cv::Vec4i> > left_right_lines, cv::Mat inputImage) {
	std::vector<cv::Point> output(4);
	cv::Point ini;
	cv::Point fini;
	cv::Point ini2;
	cv::Point fini2;
	cv::Vec4d right_line;
	cv::Vec4d left_line;
	std::vector<cv::Point> right_pts;
	std::vector<cv::Point> left_pts;

	// If right lines are being detected, fit a line using all the init and final points of the lines
	if (right_flag == true) {
		for (auto i : left_right_lines[0]) {
			ini = cv::Point(i[0], i[1]);
			fini = cv::Point(i[2], i[3]);

			right_pts.push_back(ini);
			right_pts.push_back(fini);
		}

		if (right_pts.size() > 0) {
			// The right line is formed here
			cv::fitLine(right_pts, right_line, DIST_L2, 0, 0.01, 0.01);
			right_m = right_line[1] / right_line[0];
			right_b = cv::Point(right_line[2], right_line[3]);
		}
	}

	// If left lines are being detected, fit a line using all the init and final points of the lines
	if (left_flag == true) {
		for (auto j : left_right_lines[1]) {
			ini2 = cv::Point(j[0], j[1]);
			fini2 = cv::Point(j[2], j[3]);

			left_pts.push_back(ini2);
			left_pts.push_back(fini2);
		}

		if (left_pts.size() > 0) {
			// The left line is formed here
			cv::fitLine(left_pts, left_line, DIST_L2, 0, 0.01, 0.01);
			left_m = left_line[1] / left_line[0];
			left_b = cv::Point(left_line[2], left_line[3]);
		}
	}

	// One the slope and offset points have been obtained, apply the line equation to obtain the line points
	int ini_y = inputImage.rows;
	int fin_y = (int)(inputImage.rows*0.6); // 영상의 추측 차선 끝 위치

	double right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
	double right_fin_x = ((fin_y - right_b.y) / right_m) + right_b.x;

	double left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
	double left_fin_x = ((fin_y - left_b.y) / left_m) + left_b.x;

	output[0] = cv::Point(right_ini_x, ini_y);
	output[1] = cv::Point(right_fin_x, fin_y);
	output[2] = cv::Point(left_ini_x, ini_y);
	output[3] = cv::Point(left_fin_x, fin_y);

	return output;
}

/* 차 회전 추측 함수 */
// 좌,우,직진 여부를 추측
// 반환값은 좌 우 직진 여부를 말하는 문자열
std::string LaneDetector::predictTurn() {
	std::string output;
	double vanish_x;
	double thr_vp = 10;

	// The vanishing point is the point where both lane boundary lines intersect
	vanish_x = static_cast<double>(((right_m*right_b.x) - (left_m*left_b.x) - right_b.y + left_b.y) / (right_m - left_m));

	// The vanishing points location determines where is the road turning
	if (vanish_x < (img_center - thr_vp))
		output = "Left Turn";
	else if (vanish_x > (img_center + thr_vp))
		output = "Right Turn";
	else if (vanish_x >= (img_center - thr_vp) && vanish_x <= (img_center + thr_vp))
		output = "Straight";

	return output;
}

// PLOT RESULTS
/**
*@brief This function plots both sides of the lane, the turn prediction message and a transparent polygon that covers the area inside the lane boundaries
*@param inputImage is the original captured frame
*@param lane is the vector containing the information of both lines
*@param turn is the output string containing the turn information
*@return The function returns a 0
*/
int LaneDetector::plotLane(cv::Mat inputImage, std::vector<cv::Point> lane, std::string turn) {
	std::vector<cv::Point> poly_points;
	cv::Mat output;

	// Create the transparent polygon for a better visualization of the lane
	inputImage.copyTo(output);
	poly_points.push_back(lane[2]);
	poly_points.push_back(lane[0]);
	poly_points.push_back(lane[1]);
	poly_points.push_back(lane[3]);
	cv::fillConvexPoly(output, poly_points, cv::Scalar(0, 0, 255), CV_AA, 0);
	cv::addWeighted(output, 0.3, inputImage, 1.0 - 0.3, 0, inputImage);

	// Plot both lines of the lane boundary
	cv::line(inputImage, lane[0], lane[1], cv::Scalar(0, 255, 255), 5, CV_AA);
	cv::line(inputImage, lane[2], lane[3], cv::Scalar(0, 255, 255), 5, CV_AA);

	// Plot the turn message
	cv::putText(inputImage, turn, cv::Point((int)(inputImage.rows*0.1), (int)(inputImage.cols*0.1)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 255, 0), 1, CV_AA);

	// Show the final output image
	cv::namedWindow("결과", WINDOW_AUTOSIZE);
	cv::imshow("결과", inputImage);
	return 0;
}




string returnfilepath(string a, int num, int num2)
{
	int number = 1 + num2;
	string strings = to_string(number);
	string ww = "\\";
	string jpg = ".jpg";
	string coordinate = "coordinate.txt";
	if (num == 1) // load image file
	{
		a.append(ww);
		a.append(strings);
		a.append(jpg);
	}
	if (num == 2)// load text file
	{
		a.append(ww);
		a.append(strings);
		a.append(coordinate);
	}
	return a;
}




int main(void)
{
	string image_directory = "C:\\Users\\ne_mu\\Desktop\\Dental - 복사본 (2) - 복사본 - 복사본\\Dental\\'May 02th Thursday 09-33\\image_directory";
	string imagetext_directory = "C:\\Users\\ne_mu\\Desktop\\Dental - 복사본 (2) - 복사본 - 복사본\\Dental\\'May 02th Thursday 09-33\\imagetext_directory";
	std::vector<cv::Mat> frames;
	std::vector<int>frame_num;
	int acc_flag = 0;//'
	int here = 0;

	/*// write File
	ofstream writeFile(filePath.data());
	if (writeFile.is_open())
	{
	writeFile << "Hello World!\n";
	writeFile << "This is C++ File Contents.\n";
	writeFile.close();
	}
	*/
	// read File
	int repeat = 0;;
	int path = 0;

	while (repeat < 10000) // 프레임별 처리 알고리즘 시작
	{
		string filePath = returnfilepath(imagetext_directory, 2, path);
		string imagePath = returnfilepath(image_directory, 1, path);
		path = path + 1;
		ifstream openFile(filePath.data());
		int num = 0; // 찾은 클래스의 숫자.
		Mat image = cv::imread(imagePath);
		if (image.empty())
			break;
		cv::imshow("image", image);
		vector<string> stringvector2; // 좌표랑 클래스를 담은 배열
		if (openFile.is_open()) {
			string line;
			vector<vector<string>> stringvector; // 좌표랑 클래스를 담은 배열의 배열
			while (getline(openFile, line))
			{

				stringstream ss(line);
				string a;
				while (ss >> a)
				{
					stringvector2.push_back(a);
				}
				num++;
			}
			openFile.close();
		}
		double x1 = 0, y1 = 0, x2 = 0, y2 = 0, label = 0, count = 0;
		int iol = 0;
		vector<double> coordinates[10];
		vector<int> car_pos_x, car_pos_y; //
		LaneDetector lanedetector;  // Create the class object
		cv::Mat img_denoise;
		cv::Mat img_edges;
		cv::Mat img_mask;
		cv::Mat img_lines;
		std::vector<cv::Vec4i> lines;
		std::vector<std::vector<cv::Vec4i> > left_right_lines;
		std::vector<cv::Point> lane;
		std::string turn;
		int flag_plot = -1;
		int i = 0;
		vector<int> car_pos_x, car_pos_y, person_pos_x, person_pos_y; //
		while (num)
		{
			coordinates[iol].push_back(atof(stringvector2[count].c_str())); //y1
			coordinates[iol].push_back(atof(stringvector2[count + 1].c_str())); //x1
			coordinates[iol].push_back(atof(stringvector2[count + 2].c_str())); //y2
			coordinates[iol].push_back(atof(stringvector2[count + 3].c_str())); //x2
			coordinates[iol].push_back(atof(stringvector2[count + 4].c_str())); //레이블 // 1 - person , 3 - car , 6 - bus , 8 - truck , 10 - traffic light
			car_pos_x.push_back(coordinates[iol][3] - coordinates[iol][1]);//
			car_pos_y.push_back(coordinates[iol][2] - coordinates[iol][0]);//

			if (coordinates[iol][4] == 10)// 신호등인경우
			{

				Vec3f params;
				int Color = 0, temp = 0, CNT = 0;
				int cx, cy, r;

					Mat img = imread("Pic.jpg");

					Mat Light_gray(img.size(), CV_8UC1);
					//resize(img, img, Size(300, 300));
					Mat max_mask(img.size(), CV_8UC3), Light(img.size(), CV_8UC3), max_mask_black(img.size(), CV_8UC3, Scalar(0, 0, 0));
					Mat contourMat(img.size(), CV_8UC3, Scalar(0, 0, 0)), Arrow(img.size(), CV_8UC3), max2_mask_black(img.size(), CV_8UC3, Scalar(0, 0, 0));
					//Mat img = imread("TrafficLights.jpg");
					//resize(img2, img2, Size(300, 300));
					//Mat img3=imread("LeftLight.PNG",0);
					//Mat img4=img;

					//YCrCb이미지로 변경
					Mat YCrCb, mask(img.size(), CV_8U, Scalar(0));
					cvtColor(img, YCrCb, COLOR_BGR2YCrCb);
					Mat YCrCb2, mask2(img.size(), CV_8U, Scalar(0));
					cvtColor(img, YCrCb2, COLOR_BGR2YCrCb);


					//YCrCb 이미지를 각 채널별로 분리 (Red)
					vector<Mat> planes;
					split(YCrCb, planes);
					int nr = img.rows;
					int nc = img.cols;

					//YCrCb 이미지를 각 채널별로 분리 (Green)
					vector<Mat> planes2;
					split(YCrCb2, planes2);
					int nr2 = img.rows;
					int nc2 = img.cols;

					// Red : 200 < Cr <255, 150 <Cb < 255인 영역만 255로 표시해서 mask 만들기
					for (int i = 0; i < nr; i++) {
						uchar* Cr = planes[1].ptr<uchar>(i);
						uchar* Cb = planes[2].ptr<uchar>(i);
						for (int j = 0; j < nc; j++) {
							if ((150 < Cr[j] && Cr[j] < 250) && (75 <= Cb[j] && Cb[j] < 250) || ((150 < Cr[j] && Cr[j] < 250) && (75 <= Cb[j] && Cb[j] < 250)))
								mask.at<uchar>(i, j) = 255;
						}
					}

					// Green : 150 < Cr <200, 70 <Cb < 130인 영역만 255로 표시해서 mask 만들기
					for (int i = 0; i < nr2; i++) {
						uchar* Cr = planes2[1].ptr<uchar>(i);
						uchar* Cb = planes2[2].ptr<uchar>(i);
						for (int j = 0; j < nc2; j++) {
							if ((55 < Cr[j] && Cr[j] < 84) && (75 < Cb[j] && Cb[j] < 156))
								mask2.at<uchar>(i, j) = 255;
						}
					}


					//허프 변환으로 Red원 찾기
					vector<Vec3f> circles;
					HoughCircles(mask, circles, HOUGH_GRADIENT, 2, 100);

					if (circles.size() <= 1) {

						//허프 변환으로 Green원 찾기
						HoughCircles(mask2, circles, HOUGH_GRADIENT, 2, 100);

						if (circles.size() < 1) {

							//컬러원을 하나도 찾지 못했을 때 각 Mask의 영역을 비교해 더 큰 영역이 있는 색으로 신호 구분하기
							cout << "no circles" << endl;
							vector<vector<Point>> contours, contours2, contours3;
							int max = 0;
							int max2 = 0;
							findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);   //red영역 contour
							findContours(mask2.clone(), contours3, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);   //green영역 contour
							drawContours(contourMat, contours, -1, Scalar(255, 255, 255), -1);
							drawContours(contourMat, contours3, -1, Scalar(255, 255, 255), -1);
							uchar data[] = { 0,1,0,
							 1,1,1,
							 0,1,0 };
							Mat mask(3, 3, CV_8UC1, data);
							dilate(contourMat, contourMat, mask);
							erode(contourMat, contourMat, mask);
							erode(contourMat, contourMat, mask);
							erode(contourMat, contourMat, mask);
							cvtColor(contourMat, contourMat, COLOR_RGB2GRAY);
							findContours(contourMat, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


							contours2 = contours;

							//contour영역을 크기순으로 정렬하고 가장 큰 것과 두번째로 큰 것을 찾음
							for (int j = 0; j < contours.size(); j++) {
								if (cv::contourArea(contours[max]) <= cv::contourArea(contours[j]))
								{
									max = j;
								}
							}


							if (contours.size() > 1) {
								for (int j = 0; j < contours.size(); j++) { //contour의 수가 1개보다 많을 때 2번째 큰 contour를 구함
									if (j == max) continue;
									if (cv::contourArea(contours[max2]) <= cv::contourArea(contours[j])) {
										max2 = j;
									}
								}
							}


							//max가 원인지 화살표인지 판단
							//영역을 4등분, 각각의 영역이 일정크기임을 만족하면 원이고, 좌우의 크기차이가 있고 상하의 크기차이가 적으면 화살표로 간주

							//max그리기
							vector<vector<Point>> maxcontour;
							maxcontour.push_back(contours2[max]);
							drawContours(max_mask_black, maxcontour, -1, Scalar(255, 255, 255), -1);
							Light = max_mask_black & img;
							cvtColor(Light, Light_gray, COLOR_RGB2GRAY);

							bool maxArrow = true;

							//영역 나누기
							Rect reL = boundingRect(Light_gray);
							Mat rect2L = Light_gray(reL);
							imshow("Light rect", rect2L);
							Point centerL(reL.x + (reL.width / 2), reL.y + (reL.height / 2));
							imshow("light", Light_gray);
							int L1 = 0, L2 = 0, L3 = 0, L4 = 0;
							Mat Light2 = Light_gray.clone();

							uchar* redscenes = (uchar*)Light2.data;


							for (int i = 0; i < rect2L.cols; i++)
							{
								for (int j = 0; j < rect2L.rows; j++)
								{
									uchar pixel2L = rect2L.at<uchar>(j, i);

									if (pixel2L)
									{
										if ((i < (rect2L.cols / 2)) && (j < (rect2L.rows / 2)))
											L1++; // 좌상
										if ((i > (rect2L.cols / 2)) && (j < (rect2L.rows / 2)))
											L2++; // 우상
										if ((i < (rect2L.cols / 2)) && (j > (rect2L.rows / 2)))
											L3++; // 좌하
										if ((i > (rect2L.cols / 2)) && (j > (rect2L.rows / 2)))
											L4++; // 우하
									}
								}
							}


							String Dir;

							//max의 모양 판단
							int averL = (L1 + L2 + L3 + L4) / 4;
							if ((averL*0.8 < L1) && (averL*1.2 > L1) && (averL*0.8 < L2) && (averL*1.2 > L2) &&	//원?
								(averL*0.8 < L3) && (averL*1.2 > L3) && (averL*0.8 < L4) && (averL*1.2 > L4)) {

								cout << "circle" << endl;
								maxArrow = false;


								//색 판단

								Mat YCrCb_r, mask_r(Light.size(), CV_8U, Scalar(0));
								cvtColor(Light, YCrCb_r, COLOR_BGR2YCrCb);
								Mat YCrCb_o, mask_o(Light.size(), CV_8U, Scalar(0));
								cvtColor(Light, YCrCb_o, COLOR_BGR2YCrCb);
								Mat YCrCb_g, mask_g(Light.size(), CV_8U, Scalar(0));
								cvtColor(Light, YCrCb_g, COLOR_BGR2YCrCb);
								Mat contourR(Light.size(), CV_8UC3, Scalar(0, 0, 0));
								Mat contourO(Light.size(), CV_8UC3, Scalar(0, 0, 0));
								Mat contourG(Light.size(), CV_8UC3, Scalar(0, 0, 0));
								vector<vector<Point>> contoursR, contoursO, contoursG;

								vector<Mat> planes_red;
								split(YCrCb_r, planes_red);
								int nr_r = Light.rows;
								int nc_r = Light.cols;

								vector<Mat> planes_orange;
								split(YCrCb_o, planes_orange);
								int nr_o = Light.rows;
								int nc_o = Light.cols;

								vector<Mat> planes_green;
								split(YCrCb_g, planes_green);
								int nr_g = Light.rows;
								int nc_g = Light.cols;

								int r = 0, o = 0, g = 0;


								// Red : 200 < Cr <255, 150 <Cb < 255인 영역만 255로 표시해서 mask 만들기
								for (int i = 0; i < nr_r; i++) {
									uchar* Cr = planes_red[1].ptr<uchar>(i);
									uchar* Cb = planes_red[2].ptr<uchar>(i);
									for (int j = 0; j < nc_r; j++) {
										if ((170 < Cr[j] && Cr[j] < 250) && (70 <= Cb[j] && Cb[j] < 250) || ((150 < Cr[j] && Cr[j] < 250) && (75 <= Cb[j] && Cb[j] < 250)))
											mask_r.at<uchar>(i, j) = 255;
									}
								}
								findContours(mask_r, contoursR, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
								drawContours(contourR, contoursR, -1, Scalar(255, 255, 255), -1);

								for (int i = 0; i < contourR.cols; i++)
								{
									for (int j = 0; j < contourR.rows; j++)
									{
										uchar pixel2 = contourR.at<uchar>(j, i);

										if (pixel2)
										{
											if ((i < contourR.cols) && (j < contourR.rows))
												r++;
										}
									}
								}

								// Orange : 150 < Cr <200, 70 <Cb < 130인 영역만 255로 표시해서 mask 만들기
								for (int i = 0; i < nr_o; i++) {
									uchar* Cr = planes_orange[1].ptr<uchar>(i);
									uchar* Cb = planes_orange[2].ptr<uchar>(i);
									for (int j = 0; j < nc_o; j++) {
										if ((50 < Cr[j] && Cr[j] < 120) && (60 < Cb[j] && Cb[j] < 90))
											mask_o.at<uchar>(i, j) = 255;
									}
								}
								findContours(mask_o, contoursO, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
								drawContours(contourO, contoursO, -1, Scalar(255, 255, 255), -1);

								for (int i = 0; i < contourO.cols; i++)
								{
									for (int j = 0; j < contourO.rows; j++)
									{
										uchar pixel2 = contourO.at<uchar>(j, i);

										if (pixel2)
										{
											if ((i < contourO.cols) && (j < contourO.rows))
												o++;
										}
									}
								}

								// Green : 150 < Cr <200, 70 <Cb < 130인 영역만 255로 표시해서 mask 만들기
								for (int i = 0; i < nr_g; i++) {
									uchar* Cr = planes_green[1].ptr<uchar>(i);
									uchar* Cb = planes_green[2].ptr<uchar>(i);
									for (int j = 0; j < nc_g; j++) {
										if ((50 < Cr[j] && Cr[j] < 120) && (100 < Cb[j] && Cb[j] < 135))
											mask_g.at<uchar>(i, j) = 255;
									}
								}
								findContours(mask_g, contoursG, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
								drawContours(contourG, contoursG, -1, Scalar(255, 255, 255), -1);

								for (int i = 0; i < contourG.cols; i++)
								{
									for (int j = 0; j < contourG.rows; j++)
									{
										uchar pixel2 = contourG.at<uchar>(j, i);

										if (pixel2)
										{
											if ((i < contourG.cols) && (j < contourG.rows))
												g++;
										}
									}
								}

								String Light_Color;

								if ((r > o) && (r > g)) Light_Color = "RED";
								if ((o > r) && (o > g)) Light_Color = "ORANGE";
								if ((g > r) && (g > o)) Light_Color = "GREEN";

								cout << "It is " << Light_Color << " Light" << endl;

							}

							else { //화살표
								cout << "arrow" << endl;
								if ((L1 + L3) > (L2 + L4)) Dir = "LEFT";
								if ((L1 + L3) < (L2 + L4)) Dir = "RIGHT";
								cout << "There is a " << Dir << " arrow sign" << endl;

								maxArrow = true;
							}


							//max가 화살표라면 max2는 판단할 필요없으므로 넘어간다.

							//max2영역이 max영역의 절반보다 크고 max가 화살표가 아니라고 판단된 경우 max2를 화살표라고 판단
							if ((maxArrow == false)) {
								//max2 그리기
								vector<vector<Point>> max2contour;
								max2contour.push_back(contours2[max2]);
								drawContours(max2_mask_black, max2contour, -1, Scalar(255, 255, 255), -1);
								Arrow = max2_mask_black & img;
								cvtColor(Arrow, Arrow, COLOR_RGB2GRAY);

								//영역 나누기
								Rect re = boundingRect(Arrow);
								Mat rect2 = Arrow(re);
								imshow("rect2", rect2);
								rectangle(Arrow, re, Scalar(255, 255, 255));
								Point center(re.x + (re.width / 2), re.y + (re.height / 2));
								circle(Arrow, center, 5, Scalar(255, 100, 150));
								imshow("arrow", Arrow);
								int a1 = 0, a2 = 0, a3 = 0, a4 = 0;
								Mat Arrow2 = Arrow.clone();
								uchar* redscenes = (uchar*)Arrow2.data;


								for (int i = 0; i < rect2.cols; i++)
								{
									for (int j = 0; j < rect2.rows; j++)
									{
										uchar pixel2 = rect2.at<uchar>(j, i);

										if (pixel2)
										{
											if ((i < (rect2.cols / 2)) && (j < (rect2.rows / 2)))
												a1++; // 좌상
											if ((i > (rect2.cols / 2)) && (j < (rect2.rows / 2)))
												a2++; // 우상
											if ((i < (rect2.cols / 2)) && (j > (rect2.rows / 2)))
												a3++; // 좌하
											if ((i > (rect2.cols / 2)) && (j > (rect2.rows / 2)))
												a4++; // 우하
										}
									}
								}

								int up, down, right, left;
								up = (a1 + a2) / 2;
								down = (a3 + a4) / 2;
								right = (a2 + a4) / 2;
								left = (a1 + a3) / 2;
								//영역 크기비교 -> 화살표인지 판단
								//상하크기가 비슷하고 좌우간 크기차이가 나면 화살표
									//방향결정
								if ((a1 + a3) > (a2 + a4)) Dir = "LEFT";
								if ((a1 + a3) < (a2 + a4)) Dir = "RIGHT";
								cout << "There is a " << Dir << " arrow sign" << endl;

							}

						}
						else {
							cout << "green light" << endl;
							for (int i = 0; i < circles.size(); i++) {
								Point center((int)(circles[i][0] + 0.5), (int)(circles[i][1] + 0.5));
								int radius = (int)(circles[i][2]);
								circle(img, center, radius, Scalar(0, 255, 0), 3);
								if (circles.size() == 1) { Color = 2; }
								cout << Color << endl;
								params = circles[i];
								cx = cvRound(params[0]);
								cy = cvRound(params[1]);
								r = cvRound(params[2]);
								cout << "centerPoint = " << cx << "," << cy << "," << r << endl;
							}
						}
					}
					else {
						cout << "red light" << endl;
						for (int i = 0; i < circles.size(); i++) {
							Point center((int)(circles[i][0] + 0.5), (int)(circles[i][1] + 0.5));
							int radius = (int)(circles[i][2]);
							params = circles[i];
							cx = cvRound(params[0]);
							cy = cvRound(params[1]);
							r = cvRound(params[2]);
							cout << "centerPoint = " << cx << "," << cy << "," << r << endl;

						}

					}

					//Show
					imshow("IMAGE", img);
					imshow("R_MASK", mask);
					imshow("G_MASK", mask2);
					imshow("MAX", max_mask_black);
					imshow("Light", Light_gray);
					imshow("Arrow", Arrow);


					waitKey();
				
			}

			if (coordinates[iol][4] == 1) // 사람인 경우
			{
				person_pos_x.push_back(coordinates[iol][3] - coordinates[iol][1]);//
				person_pos_y.push_back(coordinates[iol][2] - coordinates[iol][0]);//

				// 상대 차 중앙부 도출

				cv::Point person_center_point_for_cal_dist = cv::Point((int)((coordinates[iol][3] + coordinates[iol][1]) / 2), (int)((coordinates[iol][2] + coordinates[iol][0]) / 2));

				// 거리 계산 후 나타내기
				float person_dist = std::roundf((((280 * 0.47) / (coordinates[iol][3] - coordinates[iol][1]))) * 100) / 100; // 사람 어깨넓이 평균 47cm
				std::string dist = std::to_string(person_dist);
				cv::line(image, cv::Point(((int)(image.cols / 2)), (int)(image.rows * 0.95)), person_center_point_for_cal_dist, cv::Scalar(0, 0, 255), 1);
				cv::putText(image, dist, cv::Point(coordinates[iol][1], coordinates[iol][0]), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 0, 255), 2);
				if (person_dist <= 10 && ((person_center_point_for_cal_dist.x >= (image.cols * 0.15)) && (person_center_point_for_cal_dist.x <= (int)(image.cols * 0.85))))
					cv::rectangle(image, cv::Point(coordinates[iol][1], coordinates[iol][0]), cv::Point(coordinates[iol][3], coordinates[iol][2]), cv::Scalar(0, 0, 255), 4, 0, 0);
				else
					cv::rectangle(image, cv::Point(coordinates[iol][1], coordinates[iol][0]), cv::Point(coordinates[iol][3], coordinates[iol][2]), cv::Scalar(0, 255, 0), 4, 0, 0);

				if (person_dist <= 1.3)
					acc_flag = 1;
			}

			else if (coordinates[iol][4] == 3 || coordinates[iol][4] == 6 || coordinates[iol][4] == 8)// 신호등 포함 X
			{
				car_pos_x.push_back(coordinates[iol][3] - coordinates[iol][1]);//
				car_pos_y.push_back(coordinates[iol][2] - coordinates[iol][0]);//


				/* 추가부*/
				cv::Mat car_img = filtering_car(image, coordinates[iol][1], coordinates[iol][3], coordinates[iol][0], coordinates[iol][2]);// 
				cv::resize(car_img, car_img, cv::Size(car_img.cols * 3, car_img.rows * 3));//
				cv::GaussianBlur(car_img, car_img, cv::Size(5, 5), 0);//
				cv::Mat filter = make_filter(car_img); //
				car_img = filtering(car_img, filter); //
				std::vector<cv::Mat> hsv;  //
				cv::Mat cvt_hsv; //
				cv::cvtColor(car_img, cvt_hsv, COLOR_RGB2HSV);//
				cv::split(cvt_hsv, hsv);//
				hsv[2] = ~hsv[2];//
				cv::threshold(hsv[1], hsv[1], 100, 255, THRESH_BINARY);//
				cv::Mat filter2 = hsv[2] & hsv[1];//
				cv::threshold(filter2, filter2, 10, 255, THRESH_BINARY);//
				cv::Mat left_brake = saperate_brake(filter2, 1);//
				cv::Mat right_brake = saperate_brake(filter2, 2);//
				left_brake = last_filtering(left_brake);//
				right_brake = last_filtering(right_brake);//
				int brakes_y_point = (int)((cal_center_pixel(left_brake, right_brake)) / 3);// 흑백 이미지상의 자동차 
				std::cout << "자동차 y값 연산 끝";//

				// 상대 차 중앙부 도출
				cv::Point car_center_point_for_cal_dist;
				if (brakes_y_point != 0)
					car_center_point_for_cal_dist = cv::Point((int)((coordinates[iol][3] + coordinates[iol][1]) / 2), (int)((coordinates[iol][2] + coordinates[iol][0]) / 2) + brakes_y_point);
				else
					car_center_point_for_cal_dist = cv::Point((int)((coordinates[iol][3] + coordinates[iol][1]) / 2), (int)((coordinates[iol][2] + coordinates[iol][0]) / 2));


				float car_dist = std::roundf((((280 * 1.77) / (coordinates[iol][3] - coordinates[iol][1]))) * 100) / 100; // 중형 자동차 평균 전폭 177cm
				if ((car_center_point_for_cal_dist.x >= (int)((image.cols / 2) - (image.cols * 0.1))) && (car_center_point_for_cal_dist.x <= (int)((image.cols / 2) + (image.cols * 0.1))))
				{
					// 거리 계산 후 나타내기
					std::string dist = std::to_string(car_dist);
					cv::line(image, cv::Point(((int)(image.cols / 2)), (int)(image.rows * 0.95)), car_center_point_for_cal_dist, cv::Scalar(0, 0, 255), 1);
					cv::putText(image, dist, cv::Point(coordinates[iol][1], coordinates[iol][0]), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 0, 255), 2);

					if (car_dist >= 15)
					{
						cv::rectangle(image, cv::Point(coordinates[iol][1], coordinates[iol][0]), cv::Point(coordinates[iol][3], coordinates[iol][2]), cv::Scalar(255, 0, 0), 4, 0, 0);

					}
					else if (car_dist < 15 && car_dist >= 7)
					{
						cv::rectangle(image, cv::Point(coordinates[iol][1], coordinates[iol][0]), cv::Point(coordinates[iol][3], coordinates[iol][2]), cv::Scalar(0, 255, 255), 4, 0, 0);

					}
					else
					{
						cv::rectangle(image, cv::Point(coordinates[iol][1], coordinates[iol][0]), cv::Point(coordinates[iol][3], coordinates[iol][2]), cv::Scalar(0, 0, 255), 4, 0, 0);

					}
				}

				if (((car_center_point_for_cal_dist.x <= (int)((image.cols / 2) - (image.cols * 0.1))) || (car_center_point_for_cal_dist.x >= (int)((image.cols / 2) + (image.cols * 0.1)))) && (car_center_point_for_cal_dist.y >= (int)((image.rows * 4) / 5)))
				{
					std::string dist = std::to_string(car_dist);
					cv::line(image, cv::Point(((int)(image.cols / 2)), (int)(image.rows* 0.95)), car_center_point_for_cal_dist, cv::Scalar(0, 0, 255), 1);
					cv::putText(image, dist, cv::Point(coordinates[iol][1], coordinates[iol][0]), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 0, 255), 2);

					if (car_dist >= 4)
					{
						cv::rectangle(image, cv::Point(coordinates[iol][1], coordinates[iol][0]), cv::Point(coordinates[iol][3], coordinates[iol][2]), cv::Scalar(0, 255, 0), 4, 0, 0);
					}
					else
					{
						cv::rectangle(image, cv::Point(coordinates[iol][1], coordinates[iol][0]), cv::Point(coordinates[iol][3], coordinates[iol][2]), cv::Scalar(0, 0, 255), 4, 0, 0);
					}
				}
				if (car_dist <= 1.1)
					acc_flag = 1;
			}

			if (coordinates[iol][4] == 10) // 신호등
			{
				count = count + 5;
				num = num - 1;
				iol = iol + 1;
				continue;
			}



			// 가우시안 필터로 잡음 제거
			img_denoise = lanedetector.deNoise(image);

			// 프레임 내 엣지 검출
			img_edges = lanedetector.edgeDetector(img_denoise);
			Mat eg2 = img_edges;

			// 엣지 중 ROI 내 엣지 거름
			img_mask = lanedetector.mask(img_edges);
			bitwise_and(mask2, eg2, eg2);
			imshow("eg2", eg2);


			// ROI 내 엣지로 허프라인 검출
			lines = lanedetector.houghLines(img_mask);

			if (!lines.empty()) {
				// 허프라인으로 얻은 차선 왼쪽, 오른쪽 차선 분리
				left_right_lines = lanedetector.lineSeparation(lines, img_edges);

				// 분리된 차선 각각 얻음
				lane = lanedetector.regression(left_right_lines, image);

				// 좌, 우, 직진 판단
				turn = lanedetector.predictTurn();

				// 라인 검출
				flag_plot = lanedetector.plotLane(image, lane, turn);



				i += 1;
				cv::waitKey(25);
			}
			else {
				flag_plot = -1;
			}
			// 위험판단코드 
			Rect re((int)coordinates[iol][1], (int)coordinates[iol][0], (int)coordinates[iol][3] - (int)coordinates[iol][1] + 1, (int)coordinates[iol][2] - (int)coordinates[iol][0] + 1);
			Mat rectangles(image.size(), CV_8UC1, Scalar(0, 0, 0));
			rectangle(rectangles, re, Scalar(255, 255, 255), -1);
			Mat comingregion;
			bitwise_and(rectangles, mask2, comingregion);
			imshow("rec", rectangles);
			imshow("comingregion", comingregion);

			frame_num.push_back(acc_flag);
			frames.push_back(image);
			count = count + 5;
			num = num - 1;
			iol = iol + 1;
			repeat += 1;


			cv::waitKey(25);
		}
	}



	for (int i = 0; i < frame_num.size(); i++)
	{
		if (frame_num[i] == 0)
			continue;
		if (frame_num[i] == 1 && here == 0)
		{
			here = i;
		}
	}
	if (here == 0)
	{
		std::cout << "사고가 발생하지 않은 것으로 보입니다" << std::endl;
		return 0;
	}

	if (here != 0 && frames.size() <= 339)
	{

		cv::VideoWriter oVideoWriter("C:\\Users\\USER\\Desktop\\Result.avi", CV_FOURCC('X', 'V', 'I', 'D'), 15.0, cv::Size(frames[0].cols, frames[0].rows), true); //initialize the VideoWriter object 
		if (!oVideoWriter.isOpened()) //if not initialize the VideoWriter successfully, exit the program
		{
			cout << "ERROR: Failed to write the video" << endl;
			return -1;
		}

		for (int i = 0; i < frames.size(); i++)
		{
			cv::Mat c = frames[i];
			c.convertTo(c, CV_8UC3);
			oVideoWriter.write(c);
		}
	}

	if (here != 0 && frames.size() > 339)
	{
		if (here <= 169)
		{
			cv::VideoWriter oVideoWriter("C:\\Users\\USER\\Desktop\\Result.avi", CV_FOURCC('X', 'V', 'I', 'D'), 15.0, cv::Size(frames[0].cols, frames[0].rows), true); //initialize the VideoWriter object 
			if (!oVideoWriter.isOpened()) //if not initialize the VideoWriter successfully, exit the program
			{
				cout << "ERROR: Failed to write the video" << endl;
				return -1;
			}

			for (int i = 0; i < here + 170; i++)
			{
				cv::Mat c = frames[i];
				c.convertTo(c, CV_8UC3);
				oVideoWriter.write(c);
			}
		}

		if (here > 169 && frames.size() < (here + 170))
		{
			cv::VideoWriter oVideoWriter("C:\\Users\\USER\\Desktop\\Result.avi", CV_FOURCC('X', 'V', 'I', 'D'), 15.0, cv::Size(frames[0].cols, frames[0].rows), true); //initialize the VideoWriter object 
			if (!oVideoWriter.isOpened()) //if not initialize the VideoWriter successfully, exit the program
			{
				cout << "ERROR: Failed to write the video" << endl;
				return -1;
			}

			for (int i = here - 170; i < frames.size(); i++)
			{
				cv::Mat c = frames[i];
				c.convertTo(c, CV_8UC3);
				oVideoWriter.write(c);
			}
		}

		if (here > 169 && frames.size() > (here + 170))
		{
			cv::VideoWriter oVideoWriter("C:\\Users\\USER\\Desktop\\Result.avi", CV_FOURCC('X', 'V', 'I', 'D'), 15.0, cv::Size(frames[0].cols, frames[0].rows), true); //initialize the VideoWriter object 
			if (!oVideoWriter.isOpened()) //if not initialize the VideoWriter successfully, exit the program
			{
				cout << "ERROR: Failed to write the video" << endl;
				return -1;
			}

			for (int i = here - 170; i < here + 170 - 1; i++)
			{
				cv::Mat c = frames[i];
				c.convertTo(c, CV_8UC3);
				oVideoWriter.write(c);
			}
		}
	}
	return 0;
}


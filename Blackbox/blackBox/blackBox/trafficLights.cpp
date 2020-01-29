#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat preprocessing(Mat img) {	//영상 전처리
	Mat gray, th_img;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(7, 7), 2, 2); 

	threshold(gray, th_img, 130, 255, THRESH_BINARY | THRESH_OTSU);
	morphologyEx(th_img, th_img, MORPH_OPEN, Mat(), Point(-1, -1), 1);

	return th_img;
}

vector<Point2f> find_center(Mat img) {	//원의 중심 찾기
	vector<vector<Point>> contours;
	findContours(img.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);	//객체의 외곽선들을 찾아서 외곽선 좌표(vector<Point>)로 변환한다.

	vector<Point2f> circles;						//원의 중심좌표를 저장할 벡터행렬
	for (int i = 0; i < contours.size(); i++) {
		RotatedRect mr = minAreaRect(contours[i]);	//findContours에서 얻은 좌표들을 적용해 해당 좌표들을 포함하는 RotatedRect(회전사각형)객체로 변환
		circles.push_back(mr.center);
	}
	return circles;
}

int main() {
	String filename = "TrafficLights.jpg";
	Mat image = imread(filename, 1);
	CV_Assert(image.data);

	Mat th_img = preprocessing(image);
	vector<Point2f> circles = find_center(image);

	for (int i = 0; i < circles.size(); i++) {
		circle(image, circles[i], 1, Scalar(0, 255, 0), 2);
		cout << circles[i] << endl;
	}

	imshow("전처리영상", th_img);
	imshow("신호등영상", image);

	waitKey();
	return 0;
}
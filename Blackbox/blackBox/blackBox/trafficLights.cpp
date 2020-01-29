#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat preprocessing(Mat img) {	//���� ��ó��
	Mat gray, th_img;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(7, 7), 2, 2); 

	threshold(gray, th_img, 130, 255, THRESH_BINARY | THRESH_OTSU);
	morphologyEx(th_img, th_img, MORPH_OPEN, Mat(), Point(-1, -1), 1);

	return th_img;
}

vector<Point2f> find_center(Mat img) {	//���� �߽� ã��
	vector<vector<Point>> contours;
	findContours(img.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);	//��ü�� �ܰ������� ã�Ƽ� �ܰ��� ��ǥ(vector<Point>)�� ��ȯ�Ѵ�.

	vector<Point2f> circles;						//���� �߽���ǥ�� ������ �������
	for (int i = 0; i < contours.size(); i++) {
		RotatedRect mr = minAreaRect(contours[i]);	//findContours���� ���� ��ǥ���� ������ �ش� ��ǥ���� �����ϴ� RotatedRect(ȸ���簢��)��ü�� ��ȯ
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

	imshow("��ó������", th_img);
	imshow("��ȣ���", image);

	waitKey();
	return 0;
}
#include <tuple>
#include <opencv2/opencv.hpp>

class FireDetector {
public:
	FireDetector();

	// detection fire in an image frame
	std::tuple<bool, std::vector<cv::KeyPoint>> detectFire(cv::Mat &image);

private:
	cv::Ptr<cv::SimpleBlobDetector> detector;

	const int threshold = 110;
	const int area = 1;
	const float max_value = 32124.0;
	const float min_value = 28896.0;
};


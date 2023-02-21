#include <tuple>
#include <opencv2/opencv.hpp>

class FireDetector {
public:
	FireDetector();

	// detection fire in an image frame
	std::tuple<bool, std::vector<cv::KeyPoint>> detect_fire(cv::Mat &image);

private:
	cv::Ptr<cv::SimpleBlobDetector> detector;

	const int threshold = 211; //90
	const int area = 1; //6
	const float max_value = 32124.0; //30200.0;
	const float min_value = 28896.0; //28160.0;
};


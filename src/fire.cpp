#include "fire.hpp"

FireDetector::FireDetector() {

	// setup blob parameters
	cv::SimpleBlobDetector::Params params;

	// filter by thresholding
	params.minThreshold = 0;
	params.maxThreshold = threshold;

	// filter by area
	params.filterByArea = true;
	params.minArea = area;

	// filter by circularity (not used)
	params.filterByCircularity = false;
	params.minCircularity = 0.1;

	// filter by convexity (not used)
	params.filterByConvexity = false;
	params.minConvexity = 0.87;

	// filter by inertia (not used)
	params.filterByInertia = false;
	params.minInertiaRatio = 0.01;

	detector = cv::SimpleBlobDetector::create(params);
}

std::tuple<bool, std::vector<cv::KeyPoint>> FireDetector::detectFire(cv::Mat &image){

	// thresholding
	cv::Mat mask;
	cv::threshold(image, mask, max_value, max_value, cv::THRESH_TRUNC);
	cv::bitwise_not(mask, image);
	cv::threshold(image, mask, 65536.0 - min_value, 65536.0 - min_value, cv::THRESH_TRUNC);

	// apply normalization (convert to 8-bit)
	cv::bitwise_not(mask, image);
	image.convertTo(image, CV_8UC1, 255.0 / (max_value - min_value), - 255.0 * min_value / (max_value - min_value));

	// detect blobs
	std::vector<cv::KeyPoint> keypoints;
	cv::bitwise_not(image,image);
	detector->detect(image, keypoints);

	bool flame = (keypoints.size() != 0);

	return {flame, keypoints};
}

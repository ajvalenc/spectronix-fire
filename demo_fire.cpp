#include "fire.hpp"
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {

  std::string directory{"/home/ajvalenc/Datasets/spectronix/thermal/fire/raw/flames/*.png"};
	//std::string directory{"/home/ajvalenc/Datasets/spectronix/thermal/fire/new/blood_fire_test_02_MicroCalibir_M0000334/*.png"};
	std::vector<cv::String> filenames;
	cv::glob(directory, filenames, false);

	// create fire detector instance
	FireDetector fdet;

	int i = 0;
	int detected = 0;
	int runtime = 0;
	while (i < filenames.size()) {
		
		// read image (16-bit)
		cv::Mat img = cv::imread(filenames[i], cv::IMREAD_ANYDEPTH);

		auto start = std::chrono::high_resolution_clock::now();
		// detect
		auto [flame, keypoints] = fdet.detectFire(img);
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
		runtime += duration.count();
		std::cout << "\nFire detection\n"
			<< "frame: " << i << ", duration: " << duration.count() << " Fps: " << 1000.0f / duration.count() << "\n";
		
		// process output
		if (flame) {
			detected += 1;
		}
		i += 1;
		
		// plotting
		cv::drawKeypoints(img, keypoints, img, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		cv::imshow("Thermal Camera", img);
		if ((char)cv::waitKey(5) > 0 ) break;
	}

	std::cout << "number of flames detected = " << detected << "\n";
	std::cout << "detection percentage = " << 100.*((float)detected/filenames.size()) << "\n";
	std::cout << "duration average = " << (float)runtime/filenames.size() << "\n";
	
}


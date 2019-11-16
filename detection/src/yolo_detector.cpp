#include <iostream>
#include <experimental/filesystem>
#include <fstream>
#include <string>
#include <sys/time.h>

#include "ros/ros.h"
#include "sensor_msgs/RegionOfInterest.h"
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"

#include "ds_image.h"
#include "trt_utils.h"
#include "yolo.h"
#include "yolo_config_parser.h"
#include "yolov3.h"


std::unique_ptr<Yolo> inferNet{nullptr};
std::vector<DsImage> dsImages;


void detectorCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    std::cout << "Receive image" << std::endl;

    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    DsImage curImage(cv_ptr->image, inferNet->getInputH(), inferNet->getInputW());

    dsImages.clear();
    dsImages.emplace_back(curImage);
    cv::Mat trtInput = blobFromDsImages(dsImages, inferNet->getInputH(), inferNet->getInputW());

    inferNet->doInference(trtInput.data, 1);

    auto binfo = inferNet->decodeDetections(0, curImage.getImageHeight(), curImage.getImageWidth());
    auto remaining = nmsAllClasses(inferNet->getNMSThresh(), binfo, inferNet->getNumClasses());

    for (auto b : remaining)
    {
        printPredictions(b, inferNet->getClassName(b.label));
        curImage.addBBox(b, inferNet->getClassName(b.label));
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "detector");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("/images", 1, detectorCallback);

    int argc_ = 3;
    const char* argv_[] = {"execname", "--flagfile", "config.txt"};
    yoloConfigParserInit(argc_, const_cast<char **>(argv_));
    NetworkInfo yoloInfo = getYoloNetworkInfo();
    InferParams yoloInferParams = getYoloInferParams();
    uint64_t seed = getSeed();
    std::string networkType = getNetworkType();
    std::string precision = getPrecision();


    srand(unsigned(seed));

    inferNet = std::unique_ptr<Yolo>{new YoloV3(1, yoloInfo, yoloInferParams)};

    ros::spin();
    return 0;
}
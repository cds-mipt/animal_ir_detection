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

#include "Object.h"
#include "ObjectArray.h"
#include "RoiArray.h"

std::unique_ptr<Yolo> inferNet{nullptr};
std::vector<DsImage> dsImages;
ros::Publisher* objects_pub_ptr;
ros::Publisher* rois_pub_ptr;


void detectorCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    DsImage curImage(cv_ptr->image);

    dsImages.clear();
    dsImages.emplace_back(curImage);
    cv::Mat trtInput = blobFromDsImages(dsImages, inferNet->getInputH(), inferNet->getInputW());

    inferNet->doInference(trtInput.data, 1);

    auto binfo = inferNet->decodeDetections(0, curImage.getImageHeight(), curImage.getImageWidth());
    auto remaining = nmsAllClasses(inferNet->getNMSThresh(), binfo, inferNet->getNumClasses());

    detection::ObjectArray objects;
    objects.header.stamp = msg->header.stamp;
    for (auto b : remaining)
    {
        detection::Object object;
        object.label = b.label;
        object.bbox.x_offset = b.box.x1;
        object.bbox.y_offset = b.box.y1;
        object.bbox.width = b.box.x2 - b.box.x1;
        object.bbox.height = b.box.y2 - b.box.y1;
        objects.objects.emplace_back(object);
    }
    objects_pub_ptr->publish(objects);
}


void loadModel()
{
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
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "detector");
    ros::NodeHandle n;

    ros::Subscriber sub = n.subscribe("/images", 1, detectorCallback);
    ros::Publisher objects_pub = n.advertise<detection::ObjectArray>("/detection/yolo/objects", 1);
    ros::Publisher rois_pub = n.advertise<detection::RoiArray>("/detection/yolo/rois", 1);

    objects_pub_ptr = &objects_pub;
    rois_pub_ptr = &rois_pub;

    loadModel();

    ros::spin();
    return 0;
}
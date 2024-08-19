//
// Created by ZYN on 23-9-11.
//
#include <iostream>
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/LaserScan.h"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "Eigen/Eigen"
#include <chrono>


std::map<std::string, cv::Vec3b> label_bgr={
        {"vehicle",  cv::Vec3b(142, 0, 0)},
        {"road",     cv::Vec3b(128, 64, 128)},
        {"lane",     cv::Vec3b(50, 234, 157)},
        {"lamp",     cv::Vec3b(153, 153, 153)},
        {"sidewalk", cv::Vec3b(232, 35, 244)},
        {"signal",   cv::Vec3b(30, 170, 250)},
        {"median",   cv::Vec3b(81, 0, 81)},
        {"box1",   cv::Vec3b(50, 120, 170)},
        {"box2",   cv::Vec3b(160, 190, 110)},
        {"rider",   cv::Vec3b(60, 20, 220)}
};

const std::map<std::string, cv::Vec3b> label_bgr1 = {
        {"vehicle",  cv::Vec3b(142, 0, 0)},
        {"road",     cv::Vec3b(128, 64, 128)},
        {"lane",     cv::Vec3b(50, 234, 157)},
        {"lamp",     cv::Vec3b(153, 153, 153)},
        {"sidewalk", cv::Vec3b(232, 35, 244)}
};

class CADGT
{
public:
    CADGT(ros::NodeHandle &nh): _nh(nh)
    {
    	// for carla-0.9.10.1 , segmentation image topic: /carla/ego_vehicle/camera/semantic_segmentation/front/image_segmentation
    	// for carla-0.9.12 , segmentation image topic: /carla/ego_vehicle/semantic_segmentation_front/image
        _semanticsImageSub = nh.subscribe("/carla/ego_vehicle/semantic_segmentation_front/image", 1, &CADGT::semanticsImageCallback, this);
        _laserPub = nh.advertise<sensor_msgs::LaserScan>("/cad_carla_gt", 1);
    }

private:
    ros::NodeHandle &_nh;
    ros::Subscriber _semanticsImageSub;
    ros::Publisher _laserPub;
//    cv::Mat original_image;
//    cv::Mat ego_mask;
//    cv::Mat filtered_first;
//    cv::Mat filtered;
//    cv_bridge::CvImagePtr cvPtr;


    void semanticsImageCallback(const sensor_msgs::Image::ConstPtr &img)
    {
        auto time_start = std::chrono::steady_clock::now();
        cv_bridge::CvImagePtr cvPtr;
        try
        {
            cvPtr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
//        original_image = cvPtr->image;


        // 1. Calculating the ego mask
        cv::Mat ego_mask;
        ego_mask = calEgoMask_backup(cvPtr->image, label_bgr);
//        cv::imwrite("/home/vsisauto/filtered.png", ego_mask);
        // 2. Filtering
        cv::Mat filtered_first;
        filtered_first = processImg(cvPtr->image, label_bgr, ego_mask);
        cv::Mat filtered_second; // filter out the Blur
        cv::Mat filtered;
        cv::medianBlur(filtered_first, filtered, 3);

//        cv::imwrite("/home/vsisauto/filtered.png", filtered);

        // 3. Find the laser
        size_t laserNum = 384;
        double resolution = 30.0 / 600;
        Eigen::Vector2d origin(double(cvPtr->image.cols) / 2 - 0.5, double(cvPtr->image.rows) / 2 - 0.5 );
        std::vector<Eigen::Vector2d> ends(laserNum);
        calEnd(filtered, origin, ends, -M_PI, label_bgr1);

        auto laserPtr = cvtEndToLaser(origin, ends, resolution);

//        ROS_INFO("Acquire Laser Scan ...");
        // 4. Publish the laser
        publish_laser(laserPtr);
        auto time_end = std::chrono::steady_clock::now();

        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();


        ROS_INFO("FPS: %f, frames / s", 1000 / double(elapsed_time));
    }


    cv::Mat calEgoMask(const cv::Mat &src, std::map<std::string, cv::Vec3b> &label_bgr)
    {
        cv::Mat mask;
        cv::inRange(src, label_bgr["vehicle"], label_bgr["vehicle"], mask);

        cv::Mat labels;
        cv::connectedComponents(mask, labels);
        mask = labels == labels.at<int>(src.rows / 2, src.cols / 2);

        auto res = mask.clone();
        for (auto i = 3; i < mask.rows-3; ++i)
        {
            for (auto j = 3; j < mask.cols-3; ++j)
            {
                auto p = mask.at<uchar>(i, j);
                if (p == 255)
                {
                    for (int i_ = -3; i_ <= 3; ++i_)
                    {
                        for (int j_ = -3; j_ <= 3; ++j_)
                        {
                            res.at<uchar>(i + i_, j + j_) = 255;
                        }
                    }
                }
            }
        }
        return res;
    }


    cv::Mat calEgoMask_backup(const cv::Mat &src, std::map<std::string, cv::Vec3b> &label_bgr)
    {
        cv::Size size;
        size.height = src.rows;
        size.width = src.cols;
        cv::Mat mask = cv::Mat(size, CV_8U, cv::Scalar(0));
//        cv::inRange(src, label_bgr["vehicle"], label_bgr["vehicle"], mask);

//        cv::Mat labels;
//        cv::connectedComponents(mask, labels);
//        mask = labels == labels.at<int>(src.rows / 2, src.cols / 2);

        auto res = mask.clone();
        for (auto i = mask.rows / 2 - 22; i < mask.rows / 2 + 22; ++i) // original rows minus 15
        {
            for (auto j = mask.cols / 2 - 22; j < mask.cols / 2 + 22; ++j)
            {
                auto p = mask.at<uchar>(i, j);
                res.at<uchar>(i, j) = 255;
            }
        }
        return res;
    }


    cv::Mat processImg(const cv::Mat &img_, std::map<std::string, cv::Vec3b> &label_bgr, const cv::Mat &egoMask)
    {
        auto img = img_.clone();

        img.setTo(label_bgr["road"], egoMask);

        cv::Mat box1Mask, box2Mask, riderMask, vehicleMask;
        cv::inRange(img, label_bgr["box1"], label_bgr["box1"], box1Mask);
        cv::inRange(img, label_bgr["box2"], label_bgr["box2"], box2Mask);
        cv::inRange(img, label_bgr["rider"], label_bgr["rider"], riderMask);
        cv::inRange(img, label_bgr["vehicle"], label_bgr["vehicle"], vehicleMask);
        auto element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(box1Mask, box1Mask, cv::MORPH_CLOSE, element);
        cv::morphologyEx(box2Mask, box2Mask, cv::MORPH_CLOSE, element);

        cv::Mat laneMask;
        cv::inRange(img_, label_bgr["lane"], label_bgr["lane"], laneMask);
        img.setTo(label_bgr["road"], laneMask);

        cv::Mat signalMask;
        cv::inRange(img, label_bgr["signal"], label_bgr["signal"], signalMask);
        img.setTo(label_bgr["lamp"], signalMask);

        cv::Mat roadMask, medianMask, sidewalkMask;
        cv::inRange(img, label_bgr["road"], label_bgr["road"], roadMask);
        cv::inRange(img, label_bgr["median"], label_bgr["median"], medianMask);
        cv::inRange(img, label_bgr["sidewalk"], label_bgr["sidewalk"], sidewalkMask);

        cv::morphologyEx(roadMask, roadMask, cv::MORPH_CLOSE, element);
        cv::morphologyEx(medianMask, medianMask, cv::MORPH_CLOSE, element);
        cv::morphologyEx(sidewalkMask, sidewalkMask, cv::MORPH_CLOSE, element);

        img.setTo(label_bgr["road"], roadMask);
        img.setTo(label_bgr["median"], medianMask);
        img.setTo(label_bgr["sidewalk"], sidewalkMask);

        img.setTo(label_bgr["box1"], box1Mask);
        img.setTo(label_bgr["box2"], box2Mask);
        img.setTo(label_bgr["rider"], riderMask);
        img.setTo(label_bgr["vehicle"], vehicleMask);

        cv::Mat lampMask;
        cv::inRange(img, label_bgr["lamp"], label_bgr["lamp"], lampMask);
        img.setTo(label_bgr["road"], lampMask);
        return img;
    }


    Eigen::Vector2d voxelTraversal(
            const cv::Mat &img,
            const Eigen::Vector2d &start,
            const Eigen::Vector2d &ray,
            const std::map<std::string, cv::Vec3b> &label_bgr
    )
    {
        Eigen::Vector2i currentVoxel(int(ceil(start[0])), int(ceil(start[1])));

        auto &vx = currentVoxel[0], &vy = currentVoxel[1];
        const auto vxSize = img.cols, vySize = img.rows;

        auto stepX = (ray[0] >= 0) ? 1 : -1;
        auto stepY = (ray[1] >= 0) ? 1 : -1;

        auto tMaxX = (ray[0] != 0) ? double(stepX) / ray[0] : std::numeric_limits<float>::max();
        auto tMaxY = (ray[1] != 0) ? double(stepY) / ray[1] : std::numeric_limits<float>::max();

        auto tDeltaX = tMaxX;
        auto tDeltaY = tMaxY;

        Eigen::Vector2i diff(0, 0);
        bool negRay = false;
        if (ray[0] < 0)
        {
            --diff[0];
            negRay = true;
        }
        if (ray[1] < 0)
        {
            --diff[1];
            negRay = true;
        }

        if (negRay) currentVoxel += diff;

        int cntX = 0, cntY = 0, through;
        while (true)
        {
            if (tMaxX < tMaxY)
            {
                ++cntX;
                auto nextX_ = vx + stepX;
                if (nextX_ < 0 or
                    nextX_ >= vxSize or
                    img.at<cv::Vec3b>(img.rows - 1 - vy, nextX_) != label_bgr.at("road"))
                {
                    through = 1;  /// Hit vertical border
                    break;
                }
                vx = nextX_;
                tMaxX += tDeltaX;
            }
            else
            {
                ++cntY;
                auto nextY_ = vy + stepY;
                if (nextY_ < 0 or
                    nextY_ >= vySize or
                    img.at<cv::Vec3b>(img.rows - 1 - nextY_, vx) != label_bgr.at("road"))
                {
                    through = 0;  /// Hit horizontal border
                    break;
                }
                vy = nextY_;
                tMaxY += tDeltaY;
            }
        }

        double endX, endY;
        if (through == 0)
        {
            auto deltaY = cntY * stepY;
            endY = deltaY + start[1];
            endX = ray[0] / ray[1] * deltaY + start[0];
        }
        else
        {
            auto deltaX = cntX * stepX;
            endX = deltaX + start[0];
            endY = ray[1] / ray[0] * deltaX + start[1];
        }

        return {endX, endY};
    }


    void calEnd(
            const cv::Mat &img,
            const Eigen::Vector2d &start,
            std::vector<Eigen::Vector2d> &ends,
            double minAngle,
            const std::map<std::string, cv::Vec3b> &label_bgr
    )
    {
        double angleDiff = 2 * M_PI / double(ends.size());
        auto startAngle = minAngle + angleDiff / 2;
        for (auto i = 0; i < ends.size(); ++i)
        {
            auto theta = startAngle + i * angleDiff;
            Eigen::Vector2d ray(cos(theta), sin(theta));
            ends[i] = voxelTraversal(img, start, ray, label_bgr);
        }
    }


    std::shared_ptr<double[]> cvtEndToLaser(
            const Eigen::Vector2d &start,
            const std::vector<Eigen::Vector2d> &ends,
            double resolution
    )
    {
        std::shared_ptr<double[]> laser(new double[ends.size()]);
        for (auto i = 0; i < ends.size(); ++i)
        {
            laser[i] = (ends[i] - start).norm() * resolution;
            if(laser[i] > 15) laser[i] = 15;
        }

        return laser;
    }


    cv::Mat visualizeLaser(
            const cv::Mat &src,
            const Eigen::Vector2d &start,
            const std::vector<Eigen::Vector2d> &ends
    )
    {
        auto img = src.clone();
        for (auto &end: ends)
        {
            cv::line(
                    img,
                    cv::Point( int(start[0]), img.rows - 1 - int(start[1])),
                    cv::Point(int(end[0]), img.rows - 1 - int(end[1])),
                    cv::Scalar(128, 90, 128)
            );
            img.at<cv::Vec3b>(img.rows - 1 - int(end[1]), int(end[0])) = cv::Vec3b(0, 255, 255);

        }
        return img;
    }


    void publish_laser(std::shared_ptr<double[]> &laser)
    {
        sensor_msgs::LaserScan scan;
        scan.header.stamp = ros::Time::now();
        scan.header.frame_id = "ego_vehicle";
        scan.angle_min = -M_PI;
        scan.angle_max = M_PI;
        scan.angle_increment = 2*M_PI / 384;
        scan.range_min = 0.1;
        scan.range_max = 20;

        scan.ranges.resize(384);
        for(int i=0; i < 384; i++)
        {
            scan.ranges[i] = laser[i];
        }
        _laserPub.publish(scan);
        ROS_INFO("Publishing LaserScan ... ");

    }

};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "cad_gt_carla");
    ros::NodeHandle nh("~");
    CADGT cadgt(nh);
    ros::spin();
    return 0;
}

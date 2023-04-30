#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include <cv_bridge/cv_bridge.h>
#include <eigen_conversions/eigen_msg.h>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Path.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include <algorithm>
#include <ctime>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "feature.h"
#include "utils.h"
#include "visualOdometry.h"

#define PI 3.14159

class StereoVO
{
  public:
    StereoVO(cv::Mat projMatrl_, cv::Mat projMatrr_, ros::NodeHandle nh);

    cv::Mat rosImage2CvMat(sensor_msgs::ImageConstPtr img);

    void normalizeCvMat(cv::Mat &mat);

    // stereo pair callback
    void stereo_callback(const sensor_msgs::ImageConstPtr &image_left, const sensor_msgs::ImageConstPtr &image_right);

    // runs the pipeline
    void run();

    // mocap data callback
    void mocap_callback(const geometry_msgs::PoseStampedConstPtr &mocap_msg);

    // 是否显示跟踪图像
    bool display_track;

    // 利用动捕进行位姿校正的时候的距离和角度阈值
    double dist_threshold;
    double angle_threshold;

    bool mocap_adjust;
    bool use_lab_mocap;

  private:
    int frame_id = 0;

    // projection matrices for camera
    cv::Mat projMatrl, projMatrr;

    // images of current and next time step
    cv::Mat imageRight_t0, imageLeft_t0;
    cv::Mat imageRight_t1, imageLeft_t1;

    // initial pose variables
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);

    // std::cout << "frame_pose " << frame_pose << std::endl;
    cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);

    // set of features currently tracked
    FeatureSet currentVOFeatures;

    // for timing code
    clock_t t_a, t_b;
    clock_t t_1, t_2;

    // show vio path
    nav_msgs::Path vio_path;
    ros::Publisher vio_path_pub;
    // show mocap path
    nav_msgs::Path mocap_path;
    ros::Publisher mocap_path_pub;

    // publish vo pose
    ros::Publisher pose_pub;

    // mocap subscribtion
    ros::Subscriber mocap_sub;

    Eigen::Isometry3d T_w_b1;
    Eigen::Isometry3d T_b1_b2;
    Eigen::Isometry3d T_m_b2;
    Eigen::Isometry3d T_w_m;
    Eigen::Isometry3d T_m_b2_vio;

    std::mutex m_mocap_update;

    bool first_mocap_pose_msg;
    bool first_img_process;
};

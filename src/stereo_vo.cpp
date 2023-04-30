#include "stereo_visual_odometry/stereo_vo.h"
#include <stdexcept>

StereoVO::StereoVO(cv::Mat projMatrl_, cv::Mat projMatrr_, ros::NodeHandle nh)
{
    projMatrl = projMatrl_;
    projMatrr = projMatrr_;

    vio_path_pub = nh.advertise<nav_msgs::Path>("vio_path", 10);
    mocap_path_pub = nh.advertise<nav_msgs::Path>("mocap_path", 10);

    mocap_sub = nh.subscribe("/mavros/vision_pose/pose", 10, &StereoVO::mocap_callback, this);

    first_mocap_pose_msg = true;
}

cv::Mat StereoVO::rosImage2CvMat(sensor_msgs::ImageConstPtr img)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
        return cv::Mat();
    }
    return cv_ptr->image;
}

void StereoVO::mocap_callback(const geometry_msgs::PoseStampedConstPtr &mocap_msg)
{
    if (first_mocap_pose_msg)
    {
        Eigen::Quaterniond orientation;
        Eigen::Vector3d translation;
        tf::pointMsgToEigen(mocap_msg->pose.position, translation);
        tf::quaternionMsgToEigen(mocap_msg->pose.orientation, orientation);

        Eigen::Matrix3d R_b1_b2;
        R_b1_b2 << 0, -1, 0, 0, 0, -1, 1, 0, 0;
        T_b1_b2.linear() = R_b1_b2;
        T_b1_b2.translation() = Eigen::Vector3d::Zero();

        T_w_b1.linear() = Eigen::Matrix3d::Identity();
        T_w_b1.translation() = Eigen::Vector3d::Zero();

        T_m_b2.linear() = orientation.toRotationMatrix();
        T_m_b2.translation() = translation;

        T_w_m = T_w_b1 * T_b1_b2 * T_m_b2.inverse();

        first_mocap_pose_msg = false;
    }

    Eigen::Quaterniond orientation;
    Eigen::Vector3d translation;
    tf::pointMsgToEigen(mocap_msg->pose.position, translation);
    tf::quaternionMsgToEigen(mocap_msg->pose.orientation, orientation);
    T_m_b2.linear() = orientation.toRotationMatrix();
    T_m_b2.translation() = translation;

    tf::Transform T_m_b_gt_tf;
    tf::transformEigenToTF(T_m_b2, T_m_b_gt_tf);
    static tf::TransformBroadcaster gt_br;
    gt_br.sendTransform(tf::StampedTransform(T_m_b_gt_tf, mocap_msg->header.stamp, "map", "odom_mocap"));

    // Visualize mocap path
    geometry_msgs::PoseStamped mocap_pose_stamped;
    mocap_pose_stamped.header.stamp = ros::Time::now();
    mocap_pose_stamped.header.frame_id = "map";
    tf::poseEigenToMsg(T_m_b2, mocap_pose_stamped.pose);
    mocap_path.header.stamp = ros::Time::now();
    mocap_path.header.frame_id = "map";
    mocap_path.poses.push_back(mocap_pose_stamped);
    mocap_path_pub.publish(mocap_path);
}

void StereoVO::stereo_callback(const sensor_msgs::ImageConstPtr &image_left,
                               const sensor_msgs::ImageConstPtr &image_right)
{

    if (!frame_id)
    {
        imageLeft_t0 = rosImage2CvMat(image_left);
        imageRight_t0 = rosImage2CvMat(image_right);
        frame_id++;
        return;
    }

    imageLeft_t1 = rosImage2CvMat(image_left);
    imageRight_t1 = rosImage2CvMat(image_right);

    // run the pipeline
    ros::Time start_time = ros::Time::now();
    run();
    ROS_INFO("pipeline cost time: %f", (ros::Time::now() - start_time).toSec());
}

void StereoVO::run()
{
    std::cout << std::endl << "frame id " << frame_id << std::endl;

    t_a = clock();
    t_1 = clock();
    std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;
    matchingFeatures(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1, currentVOFeatures, pointsLeft_t0,
                     pointsRight_t0, pointsLeft_t1, pointsRight_t1);
    t_2 = clock();

    // set new images as old images
    imageLeft_t0 = imageLeft_t1;
    imageRight_t0 = imageRight_t1;

    // display visualize feature tracks
    if (display_track)
    {
        displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);
    }

    if (currentVOFeatures.size() < 5) // TODO should this be AND?
    {
        std::cout << "not enough features matched for pose estimation" << std::endl;
        frame_id++;
        return;
    }

    // ---------------------
    // Triangulate 3D Points
    // ---------------------
    cv::Mat points3D_t0, points4D_t0;
    cv::triangulatePoints(projMatrl, projMatrr, pointsLeft_t0, pointsRight_t0, points4D_t0);
    cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

    // ---------------------
    // Tracking transfomation
    // ---------------------
    // PnP: computes rotation and translation between pair of images
    trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t1, points3D_t0, rotation, translation, false);

    // ------------------------------------------------
    // Integrating
    // ------------------------------------------------
    cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);
    if (abs(rotation_euler[1]) < 0.1 && abs(rotation_euler[0]) < 0.1 && abs(rotation_euler[2]) < 0.1)
    {
        integrateOdometryStereo(frame_id, frame_pose, rotation, translation);
    }
    else
    {

        std::cout << "Too large rotation" << std::endl;
    }
    t_b = clock();
    float frame_time = 1000 * (double)(t_b - t_a) / CLOCKS_PER_SEC;
    float fps = 1000 / frame_time;
    // cout << "[Info] frame times (ms): " << frame_time << endl;
    // cout << "[Info] FPS: " << fps << endl;
    cv::Mat xyz = frame_pose.col(3).clone();
    cv::Mat R = frame_pose(cv::Rect(0, 0, 3, 3));

    // benchmark times
    if (false)
    {
        float time_matching_features = 1000 * (double)(t_2 - t_1) / CLOCKS_PER_SEC;
        std::cout << "time to match features " << time_matching_features << std::endl;
        std::cout << "time total " << float(t_b - t_a) / CLOCKS_PER_SEC * 1000 << std::endl;
    }

    // publish
    if (!first_mocap_pose_msg)
    {
        Eigen::Matrix3d rot;
        rot << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), R.at<double>(1, 0), R.at<double>(1, 1),
            R.at<double>(1, 2), R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
        Eigen::Vector3d pos;
        pos << xyz.at<double>(0), xyz.at<double>(1), xyz.at<double>(2);

        T_w_b1.linear() = rot;
        T_w_b1.translation() = pos;

        T_m_b2_vio = T_w_m.inverse() * T_w_b1 * T_b1_b2;
        Eigen::Matrix3d R_m_b2_vio = T_m_b2_vio.linear();
        Eigen::Vector3d t_m_b2_vio = T_m_b2_vio.translation();

        std::cout << "pos: " << pos.transpose() << std::endl;
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(t_m_b2_vio[0], t_m_b2_vio[1], t_m_b2_vio[2]));
        Eigen::Quaterniond q_m_b2_vio(R_m_b2_vio);
        tf::Quaternion q;
        q.setX(q_m_b2_vio.x());
        q.setY(q_m_b2_vio.y());
        q.setZ(q_m_b2_vio.z());
        q.setW(q_m_b2_vio.w());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "camera"));

        // Visualize vio path
        geometry_msgs::PoseStamped vio_pose_stamped;
        vio_pose_stamped.header.stamp = ros::Time::now();
        vio_pose_stamped.header.frame_id = "map";

        vio_pose_stamped.pose.position.x = transform.getOrigin().x();
        vio_pose_stamped.pose.position.y = transform.getOrigin().y();
        vio_pose_stamped.pose.position.z = transform.getOrigin().z();

        vio_pose_stamped.pose.orientation.x = q.x();
        vio_pose_stamped.pose.orientation.y = q.y();
        vio_pose_stamped.pose.orientation.z = q.z();
        vio_pose_stamped.pose.orientation.w = q.w();

        vio_path.header.stamp = ros::Time::now();
        vio_path.header.frame_id = "map";
        vio_path.poses.push_back(vio_pose_stamped);
        vio_path_pub.publish(vio_path);

        transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
        tf::Quaternion q2(0.0, 0.0, 0.0, 1.0);
        transform.setRotation(q2);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "odom"));
    }
    frame_id++;
}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "stereo_vo_node");

    ros::NodeHandle n;

    ros::Rate loop_rate(20);

    std::string filename; // TODO correct the name
    if (!(n.getParam("calib_yaml", filename)))
    {
        std::cerr << "no calib yaml" << std::endl;
        throw;
    }
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!(fs.isOpened()))
    {
        std::cerr << "cv failed to load yaml" << std::endl;
        throw;
    }
    float fx, fy, cx, cy, bf; // Projection matrix parameters
    fs["fx"] >> fx;
    fs["fy"] >> fy;
    fs["cx"] >> cx;
    fs["cy"] >> cy;
    fs["bf"] >> bf;

    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0, 0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0, 0., 1., 0.);

    // initialize VO object
    StereoVO stereo_vo(projMatrl, projMatrr, n);

    fs["display_track"] >> stereo_vo.display_track;
    fs["dist_threshold"] >> stereo_vo.dist_threshold;
    fs["angle_threshold"] >> stereo_vo.angle_threshold;

    // using message_filters to get stereo callback on one topic
    message_filters::Subscriber<sensor_msgs::Image> image1_sub(n, "/camera/infra1/image_rect_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> image2_sub(n, "/camera/infra2/image_rect_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image1_sub, image2_sub);
    sync.registerCallback(boost::bind(&StereoVO::stereo_callback, &stereo_vo, _1, _2));

    std::cout << "Stereo VO Node Initialized!" << std::endl;

    ros::spin();
    return 0;
}

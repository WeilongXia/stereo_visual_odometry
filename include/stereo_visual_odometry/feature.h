#ifndef FEATURE_H
#define FEATURE_H

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

// #define USE_CUDA

#if USE_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudaoptflow.hpp>
#endif

#include <algorithm>
#include <ctime>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

struct FeatureSet
{
    std::vector<cv::Point2f> points;
    std::vector<int> ages;
    int size()
    {
        return points.size();
    }
    void clear()
    {
        points.clear();
        ages.clear();
    }
};

void deleteUnmatchFeatures(std::vector<cv::Point2f> &points0, std::vector<cv::Point2f> &points1,
                           std::vector<uchar> &status);

void featureDetectionFast(cv::Mat image, std::vector<cv::Point2f> &points);

void featureDetectionGoodFeaturesToTrack(cv::Mat image, std::vector<cv::Point2f> &points);

void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2,
                     std::vector<uchar> &status);

void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f> &points0, std::vector<cv::Point2f> &points1,
                                 std::vector<cv::Point2f> &points2, std::vector<cv::Point2f> &points3,
                                 std::vector<cv::Point2f> &points0_return, std::vector<uchar> &status0,
                                 std::vector<uchar> &status1, std::vector<uchar> &status2, std::vector<uchar> &status3,
                                 std::vector<int> &ages);

void circularMatching(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1, cv::Mat img_r_1,
                      std::vector<cv::Point2f> &points_l_0, std::vector<cv::Point2f> &points_r_0,
                      std::vector<cv::Point2f> &points_l_1, std::vector<cv::Point2f> &points_r_1,
                      std::vector<cv::Point2f> &points_l_0_return, FeatureSet &current_features);

#if USE_CUDA
void circularMatching_gpu(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1, cv::Mat img_r_1,
                          std::vector<cv::Point2f> &points_l_0, std::vector<cv::Point2f> &points_r_0,
                          std::vector<cv::Point2f> &points_l_1, std::vector<cv::Point2f> &points_r_1,
                          std::vector<cv::Point2f> &points_l_0_return, FeatureSet &current_features);
#endif

void bucketingFeatures(cv::Mat &image, FeatureSet &current_features, int bucket_size, int features_per_bucket);

void appendNewFeatures(cv::Mat &image, FeatureSet &current_features);

void appendNewFeatures(std::vector<cv::Point2f> points_new, FeatureSet &current_features);

#endif

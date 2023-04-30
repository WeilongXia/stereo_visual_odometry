#ifndef UTILS_H
#define UTILS_H

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
#include <sstream>
#include <string>
#include <vector>

#include "feature.h"
#include "matrix.h"

// --------------------------------
// Visualization
// --------------------------------
void drawFeaturePoints(cv::Mat image, std::vector<cv::Point2f> &points);

void display(int frame_id, cv::Mat &trajectory, cv::Mat &pose, std::vector<Matrix> &pose_matrix_gt, float fps,
             bool showgt);

// --------------------------------
// Transformation
// --------------------------------
void integrateOdometryStereo(int frame_id, cv::Mat &frame_pose, const cv::Mat &rotation,
                             const cv::Mat &translation_stereo);

bool isRotationMatrix(cv::Mat &R);

cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R);

// --------------------------------
// I/O
// --------------------------------

void loadImageLeft(cv::Mat &image_color, cv::Mat &image_gary, int frame_id, std::string filepath);

void loadImageRight(cv::Mat &image_color, cv::Mat &image_gary, int frame_id, std::string filepath);

std::ostream &operator<<(std::ostream &os, const cv::Mat_<double> &mat)
{
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            os << mat(i, j);
            if (j < mat.cols - 1)
            {
                os << ", ";
            }
        }
        os << std::endl;
    }
    return os;
}

#endif

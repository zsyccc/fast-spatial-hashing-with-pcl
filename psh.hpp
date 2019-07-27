#pragma once

#ifndef _PSH_HPP_
#define _PSH_HPP_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/pcl_macros.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vsa.hpp>
#include <iostream>

template <typename T>
struct PointNormalTData {
    PCL_ADD_POINT4D;
    PCL_ADD_NORMAL4D;
    T costomData;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointNormalTData<bool>,
    (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
        float, normal_y, normal_y)(float, normal_z, normal_z))

namespace psh {

    template <typename T>
    class psh {
    public:
        using data_t = PointNormalTData<T>;

        psh(const typename pcl::PointCloud<data_t>::Ptr& in) {
            int k = 6;
            VSA vsa;
            vsa.setInputCloud<data_t>(in);
            vsa.setMetricOption(2);
            vsa.setEps(0.01);
            vsa.setK(k);
            auto res = vsa.compute();

            std::vector<typename pcl::PointCloud<data_t>::Ptr>
                transformed_clouds;
            std::generate_n(std::back_inserter(transformed_clouds), k, []() {
                typename pcl::PointCloud<data_t>::Ptr ret(
                    new pcl::PointCloud<data_t>);
                return ret;
            });
            pcl::visualization::PCLVisualizer viewer("x");
            for (int bucket = 0; bucket < res.size(); bucket++) {
                auto points = res[bucket];
                typename pcl::PointCloud<data_t>::Ptr source(
                    new typename pcl::PointCloud<data_t>);
                Eigen::MatrixXf X = Eigen::MatrixXf::Zero(3, points.size());
                Eigen::MatrixXf N(3, points.size());
                for (int i = 0; i < points.size(); i++) {
                    X(2, i) = 1;
                    const data_t& point = in->points[points[i]];
                    for (int j = 0; j < 3; j++) {
                        N(j, i) = point.normal[j];
                    }
                    source->points.push_back(point);
                }
                Eigen::MatrixXf R =
                    X * N.transpose() * (N * N.transpose()).inverse();
                Eigen::JacobiSVD<Eigen::MatrixXf> svd(
                    R, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix3f U = svd.matrixU().transpose();
                // Eigen::Matrix3f U = R;
                Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Zero();
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        if (std::fabs(U(i, j)) < 1e-5) U(i, j) = 0;
                        transform_matrix(i, j) = U(i, j);
                    }
                }
                std::cout << " -- " << std::endl;
                std::cout << X << "\n" << std::endl;
                std::cout << N << "\n" << std::endl;
                std::cout << R * N << "\n" << std::endl;
                std::cout << U * N << "\n" << std::endl;
                std::cout << R << "\n" << std::endl;
                std::cout << U << "\n" << std::endl;
                // std::cout << svd.singularValues() << "\n" << std::endl;
                transform_matrix(3, 3) = 1;
                // std::cout << transform_matrix << '\n' << std::endl;
                pcl::transformPointCloud(*source, *transformed_clouds[bucket],
                                         transform_matrix);
                pcl::visualization::PointCloudColorHandlerCustom<data_t>
                    handler(transformed_clouds[bucket], 240, 20, 20);
                viewer.addPointCloud(transformed_clouds[bucket], handler,
                                     std::string('c', bucket));
                pcl::visualization::PointCloudColorHandlerCustom<data_t>
                    handler2(source, 240, 240, 240);
                viewer.addPointCloud(source, handler2,
                                     std::string('x', bucket));
                // viewer.addCoordinateSystem(1.0, "cloud");
            }
            viewer.spin();
        }
    };
}  // namespace psh

#endif
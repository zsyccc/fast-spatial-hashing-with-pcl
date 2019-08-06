#pragma once

#ifndef _PSH_HPP_
#define _PSH_HPP_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/pcl_macros.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vsa.hpp>
#include <iostream>

#define PSH_DEBUG

template<typename T>
struct PointNormalTData {
    PCL_ADD_POINT4D;
    PCL_ADD_NORMAL4D;
    T costomData;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

template<typename T>
struct PointTData {
    PCL_ADD_POINT4D;
    T costomData;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
        PointNormalTData<bool>,
        (float, x, x)
                (float, y, y)
                (float, z, z)
                (float, normal_x, normal_x)
                (float, normal_y, normal_y)
                (float, normal_z, normal_z)
)

POINT_CLOUD_REGISTER_POINT_STRUCT(
        PointTData<bool>,
        (float, x, x)
                (float, y, y)
                (float, z, z)
)

namespace psh {

    template<typename T>
    class psh {
    public:
        using data_t = PointNormalTData<T>;

        psh(const typename pcl::PointCloud<data_t>::Ptr &in) {
            const int k = 6;
            VSA vsa;
            vsa.setInputCloud<data_t>(in);
            vsa.setMetricOption(2);
            vsa.setEps(1);
            vsa.setK(k);
            auto res = vsa.compute();
            auto proxies = vsa.getProxies();

            std::vector<typename pcl::PointCloud<data_t>::Ptr> transformed_clouds;
            std::generate_n(std::back_inserter(transformed_clouds), k, []() {
                typename pcl::PointCloud<data_t>::Ptr ret(new pcl::PointCloud<data_t>);
                return ret;
            });

#ifdef PSH_DEBUG
            // debug: print the size of each bucket
            for (const auto &it : res) {
                std::cout << it.size() << ' ';
            }
            std::cout << std::endl;
            // end debug
#endif
            for (int bucket = 0; bucket < k; bucket++) {
                // load source cloud
                typename pcl::PointCloud<data_t>::Ptr source(new typename pcl::PointCloud<data_t>);
                source->points.reserve(res[bucket].size());
                for (const auto &point_idx : res[bucket]) {
                    source->points.push_back(in->points[point_idx]);
                }
                // use proxy
                const auto &proxy = proxies[bucket];
                // rotation
                Eigen::Vector3f N(proxy.normal_x, proxy.normal_y, proxy.normal_z);
                Eigen::Vector3f X(0.0f, 0.0f, 1.0f);
                Eigen::Vector3f rotation_axis = N.cross(X);
                float rotation_angle = std::acos(N.dot(X) / N.norm());
                Eigen::AngleAxisf angle_axis(rotation_angle, rotation_axis);
                Eigen::Matrix3f rotation_matrix = angle_axis.matrix();
                Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Zero();
                transform_matrix(3, 3) = 1;
#ifdef PSH_DEBUG
                // debug: adjust the z axis
                transform_matrix(2, 3) = (float) bucket * 0.1f;
                // end debug
#endif
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        transform_matrix(i, j) = rotation_matrix(i, j);
                    }
                }
                pcl::transformPointCloud(*source, *transformed_clouds[bucket], transform_matrix);
            }

#ifdef PSH_DEBUG
            // debug: draw
            pcl::visualization::PCLVisualizer viewer("PSH_rotate");
            pcl::visualization::PCLVisualizer viewer2("VSA");

            std::default_random_engine generator;
            std::uniform_int_distribution random(0, 255);

            for (int bucket = 0; bucket < k; bucket++) {
                typename pcl::PointCloud<data_t>::Ptr source(
                        new typename pcl::PointCloud<data_t>);
                source->points.reserve(res[bucket].size());
                for (const auto &point_idx : res[bucket]) {
                    source->points.push_back(in->points[point_idx]);
                }

                int color_x = random(generator);
                int color_y = random(generator);
                int color_z = random(generator);

                pcl::visualization::PointCloudColorHandlerCustom<data_t>
                        handler(transformed_clouds[bucket], color_x, color_y,
                                color_z);
                viewer.addPointCloud(transformed_clouds[bucket], handler,
                                     std::string(bucket, 'c'));
                pcl::visualization::PointCloudColorHandlerCustom<data_t>
                        handler3(source, color_x, color_y, color_z);
                viewer2.addPointCloud(source, handler3,
                                      std::string(bucket, 'd'));
            }
            viewer.spin();
            viewer2.spin();
            // end debug
#endif


        }
    };
}  // namespace psh

#endif
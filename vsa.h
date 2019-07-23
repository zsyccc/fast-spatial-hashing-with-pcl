#pragma once

#ifndef _VSA_H_
#define _VSA_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>

struct PointNormalT {
    PCL_ADD_POINT4D;
    PCL_ADD_NORMAL4D;
    bool assigned;
    PointNormalT();
    PointNormalT(const PointNormalT& p);
    PointNormalT(const pcl::PointNormal& p, bool f);
    operator pcl::PointNormal();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointNormalT,
    (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
        float, normal_y, normal_y)(float, normal_z, normal_z));

template <typename PointT>
class VSA {
private:
    using Proxy = pcl::PointNormal;

    pcl::PointCloud<PointNormalT>::Ptr cloud;
    std::vector<Proxy> proxies;
    int metric_option;
    float eps;
    int k;

    bool metric2_proxy_normal(const std::vector<int>& region,
                              pcl::PointNormal& normal);

    float error_metric(const PointNormalT& point, const Proxy& proxy) const;

    struct PQElement;

public:
    VSA();
    void setInputCloud(const typename pcl::PointCloud<PointT>::Ptr in);
    void setMetricOption(int option);
    void setEps(float eps);
    void setK(int k);
    void proxy_fitting(const std::vector<int>& region, Proxy& p);
    std::vector<std::vector<int>> compute();
};

#endif
#pragma once
#ifndef _MESH_SAMPLING_HPP_
#define _MESH_SAMPLING_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

pcl::PointCloud<pcl::PointNormal>::Ptr
mesh_sampling_with_normal(const char *filename, int sample_points, float leaf_size);

pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_sampling(const char *filename, int sample_points, float leaf_size);

#endif
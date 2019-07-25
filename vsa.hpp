#pragma once

#ifndef _VSA_HPP_
#define _VSA_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
// #include <pcl/features/feature.h>
#include <pcl/kdtree/flann.h>
#include <random>
#include <queue>
#include <functional>
#include <Eigen/Dense>
#include <cassert>
#include <iostream>

struct PointNormalT {
    PCL_ADD_POINT4D;
    PCL_ADD_NORMAL4D;
    bool assigned;
    PointNormalT() : assigned(false) {}
    PointNormalT(const PointNormalT& p) {
        pcl::copyPoint(p, *this);
        assigned = p.assigned;
    }
    PointNormalT(const pcl::PointNormal& p, bool f) {
        pcl::copyPoint(p, *this);
        assigned = f;
    }
    operator pcl::PointNormal() {
        pcl::PointNormal ret;
        pcl::copyPoint(*this, ret);
        return ret;
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointNormalT,
    (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
        float, normal_y, normal_y)(float, normal_z, normal_z))

class VSA {
public:
    using Proxy = pcl::PointNormal;

    pcl::PointCloud<PointNormalT>::Ptr cloud;
    std::vector<Proxy> proxies;
    int metric_option;
    float eps;
    int k;

    bool metric2_proxy_normal(const std::vector<int>& region,
                              pcl::PointNormal& normal) {
        Eigen::MatrixXf input(region.size(), 3);
        for (int i = 0; i < region.size(); i++) {
            for (int j = 0; j < 3; j++) {
                input(i, j) = cloud->points[region[i]].normal[j];
            }
        }

        Eigen::MatrixXf meanVec = input.colwise().mean();
        Eigen::RowVectorXf meanVecRow(
            Eigen::RowVectorXf::Map(meanVec.data(), input.cols()));

        Eigen::MatrixXf zeroMeanMat = input;
        zeroMeanMat.rowwise() -= meanVecRow;
        Eigen::MatrixXf covMat;
        if (input.rows() == 1)
            covMat =
                (zeroMeanMat.adjoint() * zeroMeanMat) / double(input.rows());
        else
            covMat = (zeroMeanMat.adjoint() * zeroMeanMat) /
                     double(input.rows() - 1);

        Eigen::EigenSolver<Eigen::MatrixXf> solver(covMat);
        if (solver.info() == Eigen::Success) {
            auto eignevals_real = solver.eigenvalues().real();
            Eigen::MatrixXf::Index realMax;
            eignevals_real.rowwise().sum().maxCoeff(&realMax);
            auto v = solver.eigenvectors().col(realMax).real();
            normal.normal_x = v(0);
            normal.normal_y = v(1);
            normal.normal_z = v(2);
            return true;
        }
        return false;
    }

    float error_metric(const PointNormalT& point, const Proxy& proxy) const {
        if (metric_option == 1) {
            float t1 = point.x * proxy.normal_x + point.y * proxy.normal_y +
                       point.z * proxy.normal_z -
                       (proxy.x * proxy.normal_x + proxy.y * proxy.normal_y +
                        proxy.z * proxy.normal_z);
            return std::fabs(t1);
        } else if (metric_option == 2) {
            PointNormalT p;
            pcl::copyPoint(point, p);
            p.normal_x -= proxy.normal_x;
            p.normal_y -= proxy.normal_y;
            p.normal_z -= proxy.normal_z;
            return p.normal_x * p.normal_x + p.normal_y * p.normal_y +
                   p.normal_z * p.normal_z;
        }
    }

    struct PQElement {
        int index;
        int tag;
        float error;
        const VSA* vsa;
        PQElement(int id, int t, const VSA* _vsa)
            : index(id), tag(t), vsa(_vsa) {
            error =
                vsa->error_metric(vsa->cloud->points[index], vsa->proxies[tag]);
        }
        bool operator<(const PQElement& rhs) const { return error > rhs.error; }
    };

    VSA()
        : cloud(new pcl::PointCloud<PointNormalT>),
          metric_option(2),
          eps(0.1),
          k(100) {}

    template <typename PointT>
    void setInputCloud(const typename pcl::PointCloud<PointT>::Ptr in) {
        pcl::copyPointCloud(*in, *cloud);
    }

    void setMetricOption(int metric_option) {
        this->metric_option = metric_option;
    }

    void setEps(float eps) { this->eps = eps; }

    void setK(int k) { this->k = k; }

    void proxy_fitting(const std::vector<int>& region, Proxy& p) {
        // calculate barycenters
        p.x = p.y = p.z = p.normal_x = p.normal_y = p.normal_z = 0.0f;
        for (auto region_idx : region) {
            p.x += cloud->points[region_idx].x;
            p.y += cloud->points[region_idx].y;
            p.z += cloud->points[region_idx].z;
        }
        p.x /= region.size();
        p.y /= region.size();
        p.z /= region.size();

        // calculate normals
        if (metric_option == 1) {
            if (metric2_proxy_normal(region, p) == false) {
                printf("Error when calculating normal of a proxy!");
            }
        } else if (metric_option == 2) {
            for (auto region_idx : region) {
                p.normal_x += cloud->points[region_idx].normal_x;
                p.normal_y += cloud->points[region_idx].normal_y;
                p.normal_z += cloud->points[region_idx].normal_z;
            }
            p.normal_x /= region.size();
            p.normal_y /= region.size();
            p.normal_z /= region.size();
        }
    }

    void deleteProxy(const std::vector<std::vector<int>>& cur_partition,
                     int& delete_region_idx, int& delete_point_idx) {
        // deletion: merge 2 proxies
        float smallest_merge_error = INFINITY;
        int smallest_merge_error_index = -1;
        for (int i = 0; i < k - 1; i++) {
            std::vector<int> merging_region;
            merging_region.reserve(cur_partition[i].size() +
                                   cur_partition[i + 1].size());
            std::copy(cur_partition[i].begin(), cur_partition[i].end(),
                      back_inserter(merging_region));
            std::copy(cur_partition[i + 1].begin(), cur_partition[i + 1].end(),
                      back_inserter(merging_region));
            Proxy merging_proxy;
            proxy_fitting(merging_region, merging_proxy);
            float merging_error = 0.0f;
            float smallest = INFINITY;
            float smallest_idx = -1;
            for (auto it : merging_region) {
                float t = error_metric(cloud->points[it], merging_proxy);
                merging_error += t;
                if (t < smallest) {
                    smallest = t;
                    smallest_idx = it;
                }
            }
            if (merging_error < smallest_merge_error) {
                smallest_merge_error = merging_error;
                smallest_merge_error_index = i;
                delete_point_idx = smallest_idx;
            }
        }
        delete_region_idx = smallest_merge_error_index;
    }

    void insertProxy(const std::vector<std::vector<int>>& cur_partition,
                     int& insert_region_idx, int& insert_point_idx1,
                     int& insert_point_idx2) {
        // insertion: devide a region into a new region and a point
        float largest_error = -1.0f;
        int largest_error_index = -1;
        for (int i = 0; i < k; i++) {
            float region_error = 0.0f;
            float largest_point_error = 0.0f;
            int largest_point_error_index = -1;
            float smallest_point_error = INFINITY;
            int smallest_point_error_index = -1;
            for (int point_idx = 0; point_idx < cur_partition[i].size();
                 point_idx++) {
                float t = error_metric(cloud->points[point_idx], proxies[i]);
                region_error += t;
                if (t > largest_point_error) {
                    largest_point_error = t;
                    largest_point_error_index = point_idx;
                }
                if (t < smallest_point_error) {
                    smallest_point_error = t;
                    smallest_point_error_index = point_idx;
                }
            }
            if (region_error > largest_error) {
                largest_error = region_error;
                insert_point_idx1 = largest_point_error_index;
                insert_point_idx2 = smallest_point_error_index;
            }
        }
        insert_region_idx = largest_error_index;
    }

    std::vector<std::vector<int>> compute() {
        const int knn_search_k =
            10;  // it is likely that knn_search_k is not large enough to go
                 // through all the points

        proxies.resize(k);
        std::priority_queue<PQElement> point_priority_queue;
        std::vector<std::vector<int>> cur_partition(k);
        std::vector<int> barycenters;

        pcl::KdTreeFLANN<PointNormalT> kdtree;
        kdtree.setInputCloud(cloud);
        std::vector<int> pointIdxNKNSearch(knn_search_k);
        std::vector<float> pointNKNSquaredDistance(knn_search_k);

        // iterations
        int generation = 0;
        float total_error = 2 * eps;
        float total_error_new = 0.0f;

        // while (generation < 10) {
        while (fabs(total_error_new - total_error) > eps) {
            // find barycenters--i.e.,seed points
            if (generation == 0) {
                std::default_random_engine generator(std::time(0));
                static std::uniform_int_distribution<int> randint(
                    0, cloud->size() - 1);
                std::set<int> seed_index;
                int rand_int;

                for (int i = 0; i < k; i++) {
                    do {
                        rand_int = randint(generator);
                    } while (seed_index.count(rand_int));
                    seed_index.insert(rand_int);
                    barycenters.push_back(rand_int);
                    proxies[i] = (pcl::PointNormal)cloud->points[rand_int];
                }
                std::sort(barycenters.begin(), barycenters.end());
            } else {
                barycenters.clear();
                for (int i = 0; i < k; i++) {
                    float smallest_error = error_metric(
                        cloud->points[cur_partition[i][0]], proxies[i]);
                    int smallest_error_index = 0;
                    for (int j = 1; j < cur_partition[i].size(); j++) {
                        float cur_error = error_metric(
                            cloud->points[cur_partition[i][j]], proxies[i]);
                        if (cur_error < smallest_error) {
                            smallest_error = cur_error;
                            smallest_error_index = j;
                        }
                    }
                    barycenters.push_back(
                        cur_partition[i][smallest_error_index]);
                }
                // int delete_region_idx, delete_point_idx;
                // deleteProxy(cur_partition, delete_region_idx,
                // delete_point_idx); barycenters[delete_region_idx] =
                // delete_point_idx; int insert_region_idx, insert_point_idx1,
                // insert_point_idx2; insertProxy(cur_partition,
                // insert_region_idx, insert_point_idx1,
                //             insert_point_idx2);
                // barycenters[delete_region_idx + 1] = insert_point_idx1;
                // barycenters[insert_region_idx] = insert_point_idx2;
                // std::sort(barycenters.begin(), barycenters.end());  //
            }

            std::vector<std::vector<int>>(k).swap(cur_partition);
            for (int it = 0; it < cloud->size(); it++) {
                cloud->points[it].assigned = false;
            }

            // flooding
            // for each seed point(coming from random points or last iteration),
            // we insert its 3 adjancent points in the pq
            for (int i = 0; i < barycenters.size(); i++) {
                cloud->points[barycenters[i]].assigned = true;
                cur_partition[i].push_back(barycenters[i]);
                int debug_t;
                if ((debug_t = kdtree.nearestKSearch(
                         cloud->points[barycenters[i]], knn_search_k,
                         pointIdxNKNSearch, pointNKNSquaredDistance)) > 0) {
                    for (auto nearest_point : pointIdxNKNSearch) {
                        if (nearest_point != barycenters[i]) {
                            point_priority_queue.push(
                                PQElement(nearest_point, i, this));
                        }
                    }
                }
            }

            // do until pq is empty
            while (point_priority_queue.empty() == false) {
                PQElement testing_point = point_priority_queue.top();
                point_priority_queue.pop();
                if (cloud->points[testing_point.index].assigned == false) {
                    cloud->points[testing_point.index].assigned = true;
                    cur_partition[testing_point.tag].push_back(
                        testing_point.index);
                    int debug_t;
                    if ((debug_t = kdtree.nearestKSearch(
                             cloud->points[testing_point.index], knn_search_k,
                             pointIdxNKNSearch, pointNKNSquaredDistance)) > 0) {
                        for (auto nearest_point : pointIdxNKNSearch) {
                            if (cloud->points[nearest_point].assigned ==
                                false) {
                                point_priority_queue.push(PQElement(
                                    nearest_point, testing_point.tag, this));
                            }
                        }
                    }
                }
            }

            // proxy fitting
            proxies.clear();
            for (int i = 0; i < k; i++) {
                Proxy proxy;
                proxy_fitting(cur_partition[i], proxy);
                proxies.push_back(proxy);
            }

            // calculate total distortion error
            total_error = total_error_new;
            total_error_new = 0.0f;
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < cur_partition[i].size(); j++) {
                    total_error_new += error_metric(
                        cloud->points[cur_partition[i][j]], proxies[i]);
                }
            }

            printf("%d: %f\n", generation, total_error_new);  // debug
            generation++;
        }
        int cnt_assigned = 0;  // debug
        int cnt_not_assigned = 0;
        for (int i = 0; i < cloud->size(); i++) {
            if (cloud->points[i].assigned) {
                cnt_assigned++;
            } else {
                cnt_not_assigned++;
            }
        }
        printf("%d %d\n", cnt_assigned, cnt_not_assigned);
        return cur_partition;
    }
};

#endif
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
        // printf("region.size()=%lu \n", region.size());
        // std::assert(region.size());
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
            // static int cnt = 0;
            // while (cnt++ <= 10) {
            //     std::cout << solver.eigenvectors().col(realMax) << std::endl;
            // }
            normal.normal_x = v(0);
            normal.normal_y = v(1);
            normal.normal_z = v(2);
            return true;
        }
        return false;
    }

    float error_metric(int index, int tag) const {
        if (metric_option == 1) {
            pcl::PointNormal n;
            pcl::copyPoint(proxies[tag], n);
            const auto& point = cloud->points[index];
            float t1 = point.x * n.normal_x + point.y * n.normal_y +
                       point.z * n.normal_z -
                       (n.x * n.normal_x + n.y * n.normal_y + n.z * n.normal_z);
            float t2 = n.normal_x * n.normal_x + n.normal_y * n.normal_y +
                       n.normal_z * n.normal_z;
            return sqrt(t1 * t1 / t2);
        } else if (metric_option == 2) {
            pcl::Normal n;
            pcl::copyPoint(cloud->points[index], n);
            n.normal_x -= proxies[tag].normal_x;
            n.normal_y -= proxies[tag].normal_y;
            n.normal_z -= proxies[tag].normal_z;
            return n.normal_x * n.normal_x + n.normal_y * n.normal_y +
                   n.normal_z * n.normal_z;
        }
    }

    struct PQElement {
        int index;
        int tag;
        float error;
        const VSA* vsa;
        PQElement(int id, int t, const VSA* _vsa)
            : index(id), tag(t), vsa(_vsa) {
            error = vsa->error_metric(index, tag);
        }
        bool operator<(const PQElement& rhs) const { return error > rhs.error; }
    };

    VSA()
        : cloud(new pcl::PointCloud<PointNormalT>),
          metric_option(2),
          eps(0.1),
          k(100) {}

    void setInputCloud(const pcl::PointCloud<pcl::PointNormal>::Ptr in) {
        pcl::copyPointCloud(*in, *cloud);
    }

    void setMetricOption(int option) { this->metric_option = option; }

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
        } else {
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

    std::vector<std::vector<int>> compute() {
        const int knn_search_k = 10;

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

        // while (total_error > 5e7) {
        while (fabs(total_error_new - total_error) > eps) {
            total_error = total_error_new;
            total_error_new = 0.0f;

            std::vector<std::vector<int>>(k).swap(cur_partition);
            for (int it = 0; it < cloud->size(); it++) {
                cloud->points[it].assigned = false;
            }
            // initialize for the very first partitioning
            // pick k points at random and push them in 'barycenters'
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
                    // if (debug_t != 10) printf("%d ", debug_t);
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

            // merge 2 proxies and insert 1 proxy for each iteration
            // Proxy merging_proxy=
            // float smallest_merge_error = total_error_new;
            // int smallest_merge_error_index = -1;
            // for (int i = 0; i < k - 1; i++) {
            //     std::vector<int> merging_region;
            //     cur_partition[i];
            //     merging_region.reserve(cur_partition[i].size() +
            //                            cur_partition[i + 1].size());
            //     std::copy(cur_partition[i].begin(), cur_partition[i].end(),
            //               back_inserter(merging_region));
            //     std::copy(cur_partition[i + 1].begin(),
            //               cur_partition[i + 1].end(),
            //               back_inserter(merging_region));

            //     for (auto it :) {
            //     }

            // proxy fitting
            proxies.clear();
            for (int i = 0; i < k; i++) {
                Proxy proxy;
                proxy_fitting(cur_partition[i], proxy);
                proxies.push_back(proxy);
            }

            // find the triangle T_i of S that is most similar to its
            // respective fitting proxy
            printf("%d:", generation);  // debug
            barycenters.clear();
            for (int i = 0; i < k; i++) {
                float smallest_error = error_metric(cur_partition[i][0], i);
                int smallest_error_index = 0;
                total_error_new += smallest_error;
                for (int j = 1; j < cur_partition[i].size(); j++) {
                    float cur_error = error_metric(cur_partition[i][j], i);
                    total_error_new += cur_error;
                    if (cur_error < smallest_error) {
                        smallest_error = cur_error;
                        smallest_error_index = j;
                    }
                }
                barycenters.push_back(cur_partition[i][smallest_error_index]);
            }
            printf("%f\n", total_error_new);  // debug
            generation++;
        }
        printf("%f %f %f\n", total_error_new, total_error, eps);
        int cnt_assigned = 0;
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
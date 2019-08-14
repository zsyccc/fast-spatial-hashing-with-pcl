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

    float get_point_error(const PointNormalT& point, const Proxy& proxy) const {
        if (metric_option == 1) {
            float t1 = point.x * proxy.normal_x + point.y * proxy.normal_y +
                       point.z * proxy.normal_z -
                       (proxy.x * proxy.normal_x + proxy.y * proxy.normal_y +
                        proxy.z * proxy.normal_z);
            float t2 = proxy.normal_x * proxy.normal_x +
                       proxy.normal_y * proxy.normal_y +
                       proxy.normal_z * proxy.normal_z;
            return sqrt(t1 * t1 / t2);
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

    float get_region_error(const std::vector<int>& region, const Proxy& proxy) {
        float error = 0.0f;
        for (auto p : region) {
            error += get_point_error(cloud->points[p], proxy);
        }
        return error;
    }

    float get_total_error(const std::vector < std::vector<int>> &
                          cur_partition) {
        float error = 0.0f;
        for (int i = 0; i < k; i++) {
            error += get_region_error(cur_partition[i], proxies[i]);
        }
        return error;
    }

    struct PQElement {
        int index;
        int tag;
        float error;
        const VSA* vsa;
        PQElement(int id, int t, const VSA* _vsa)
            : index(id), tag(t), vsa(_vsa) {
            error = vsa->get_point_error(vsa->cloud->points[index],
                                         vsa->proxies[tag]);
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

    // pick k points at random and push them in 'barycenters'
    // and get k proxies defined as the xyz and normal of
    // barycenters
    void rand_init(std::vector<int>& barycenters) {
        std::default_random_engine generator(std::time(0));
        static std::uniform_int_distribution<int> randint(0, cloud->size() - 1);
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
    }

    void flooding(const std::vector<int>& barycenters,
                  std::vector<std::vector<int>>& cur_partition,
                  std::priority_queue<PQElement>& point_priority_queue,const int knn_search_k) {
        pcl::KdTreeFLANN<PointNormalT> kdtree;
        kdtree.setInputCloud(cloud);
        std::vector<int> pointIdxNKNSearch(knn_search_k);
        std::vector<float> pointNKNSquaredDistance(knn_search_k);

        // for each seed point(coming from random points or last
        // iteration), we insert its 3 adjancent points in the pq
        for (int i = 0; i < barycenters.size(); i++) {
            cloud->points[barycenters[i]].assigned = true;
            cur_partition[i].push_back(barycenters[i]);
            if (kdtree.nearestKSearch(cloud->points[barycenters[i]],
                                      knn_search_k, pointIdxNKNSearch,
                                      pointNKNSquaredDistance) > 0) {
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
                cur_partition[testing_point.tag].push_back(testing_point.index);
                if (kdtree.nearestKSearch(cloud->points[testing_point.index],
                                          knn_search_k, pointIdxNKNSearch,
                                          pointNKNSquaredDistance) > 0) {
                    for (auto nearest_point : pointIdxNKNSearch) {
                        if (cloud->points[nearest_point].assigned == false) {
                            point_priority_queue.push(PQElement(
                                nearest_point, testing_point.tag, this));
                        }
                    }
                }
            }
        }
    }

    void get_min_merge_error(const std::vector<std::vector<int>>& cur_partition,
                             std::vector<int>& order, float& err, int& idx) {
        for (int i = 0; i < k - 1; i++) {
            std::vector<int> merging_region;
            merging_region.reserve(cur_partition[order[i]].size() +
                                   cur_partition[order[i + 1]].size());
            std::copy(cur_partition[order[i]].begin(),
                      cur_partition[order[i]].end(),
                      back_inserter(merging_region));
            std::copy(cur_partition[order[i + 1]].begin(),
                      cur_partition[order[i + 1]].end(),
                      back_inserter(merging_region));
            Proxy merging_proxy;
            proxy_fitting(merging_region, merging_proxy);
            float merging_error =
                get_region_error(merging_region, merging_proxy);
            if (merging_error < err) {
                err = merging_error;
                idx = i;
            }
        }
    }

    void get_max_region_error(const std::vector<std::vector<int>>& cur_partition,
                              float& err, int& idx) {
        for (int i = 0; i < k; i++) {
            float error = get_region_error(cur_partition[i], proxies[i]);
            if (error > err) {
                err = error;
                idx = i;
            }
        }
    }

    void get_max_point_error(const std::vector<int>& region, const int proxy_id,float& err, int& idx) {
        for (int i = 0; i < region.size(); i++) {
            float error = get_point_error(
                cloud->points[region[i]],
                proxies[proxy_id]);
            if (error > err) {
                err = error;
                idx = i;
            }
        }
    }

    void teleportation(std::vector<int>& barycenters,
                       std::vector<std::vector<int>>& cur_partition) {
        // merge 2 proxies and insert 1 proxy for each iteration
        // barycenters[i] <=> proxies[order[i]] <=>
        // cur_partition[order[i]]

        std::vector<int> order(k);
        for (int i = 0; i < k; i++) order[i] = i;
        sort(order.begin(), order.end(), [&barycenters](int x, int y) {
            return barycenters[x] < barycenters[y];
        });
        std::sort(barycenters.begin(), barycenters.end());

        // find 2 regions with smallest mergeing error
        float smallest_merge_error = INFINITY;
        int smallest_merge_error_index = -1;
        get_min_merge_error(cur_partition, order, smallest_merge_error,
                            smallest_merge_error_index);

        // find the region with the largest region error
        float largest_region_error = 0.0f;
        int largest_region_error_index = -1;
        get_max_region_error(cur_partition, largest_region_error,
                             largest_region_error_index);

        // test whether do a region teleportation
        if (smallest_merge_error < 0.5 * largest_region_error) {
            // find the point with the largest point error in the
            // worst region
            float largest_point_error = 0.0f;
            int largest_point_error_index = -1;
            get_max_point_error(cur_partition[largest_region_error_index],largest_region_error_index,
                                largest_point_error, largest_point_error_index);

            // merge region[order[i]] and region[order[i+1]] into
            // region[order[i]]
            int idx = smallest_merge_error_index;
            std::copy(cur_partition[order[idx + 1]].begin(),
                      cur_partition[order[idx + 1]].end(),
                      back_inserter(cur_partition[order[idx]]));
            std::vector<int>().swap(cur_partition[order[idx + 1]]);
            cur_partition[order[idx + 1]].push_back(largest_point_error_index);
            cur_partition[largest_region_error_index].erase(
                cur_partition[largest_region_error_index].begin() +
                largest_point_error_index);
        }
    }

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

    void get_barycenters(std::vector<int>& barycenters,
                         const std::vector<std::vector<int>>& cur_partition) {
        // find the triangle T_i of S that is most similar to its
        // respective fitting proxy
        barycenters.clear();
        for (int i = 0; i < k; i++) {
            float smallest_error =
                get_point_error(cloud->points[cur_partition[i][0]], proxies[i]);
            int smallest_error_index = 0;
            for (int j = 1; j < cur_partition[i].size(); j++) {
                float cur_error = get_point_error(
                    cloud->points[cur_partition[i][j]], proxies[i]);
                if (cur_error < smallest_error) {
                    smallest_error = cur_error;
                    smallest_error_index = j;
                }
            }
            barycenters.push_back(cur_partition[i][smallest_error_index]);
        }
    }

    std::vector<std::vector<int>> compute() {
        const int knn_search_k = 20;

        proxies.resize(k);
        std::priority_queue<PQElement> point_priority_queue;
        std::vector<std::vector<int>> cur_partition(k);
        std::vector<int> barycenters;

        // iterations
        int generation = 0;
        float total_error = 2 * eps;
        float total_error_new = 0.0f;

        while (fabs(total_error_new - total_error) > eps) {
            total_error = total_error_new;
            total_error_new = 0.0f;

            std::vector<std::vector<int>>(k).swap(cur_partition);
            for (int it = 0; it < cloud->size(); it++) {
                cloud->points[it].assigned = false;
            }

            if (generation == 0)
                rand_init(
                    barycenters);  // initialize for the very first partitioning
            flooding(barycenters, cur_partition, point_priority_queue,knn_search_k);
            if (generation % 3 == 2) teleportation(barycenters, cur_partition);

            // proxy fitting
            proxies.clear();
            for (int i = 0; i < k; i++) {
                Proxy proxy;
                proxy_fitting(cur_partition[i], proxy);
                proxies.push_back(proxy);
            }

            get_barycenters(barycenters, cur_partition);

            total_error_new = get_total_error(cur_partition);
            printf("%d: %f\n", generation, total_error_new);  // debug
            generation++;
        }

        int cnt = 0;
        for (auto& region : cur_partition) {
            cnt += region.size();
        }
        printf("%lu %d\n", cloud->size(), cnt);
        return cur_partition;
    }
};

#endif
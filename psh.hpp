#pragma once

#ifndef _PSH_HPP_
#define _PSH_HPP_

#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>
#include <functional>
#include <cstring>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <vsa.hpp>
#include <point.hpp>

//#define PSH_DEBUG

#define VALUE(x) std::cout << #x "=" << x << std::endl
#define VALUE_MATRIX(x) std::cout << #x "=" << std::endl << x << std::endl

namespace psh {

    const uint layer_dimension = 2;

    template<class T>
    class map {
    private:
        using IndexInt = size_t;
        using PosInt = uint32_t;
        using LL = int64_t;
        using OffsetInt = int32_t;
        using HashInt = PosInt;

        struct layer_map;

        // data size
        IndexInt n;

        std::default_random_engine generator;

        // partition number from VSA
        IndexInt part_num;

        std::vector<layer_map> hash_buckets;

        // three primes for use in hashing
        IndexInt M0;

        IndexInt M1;

        IndexInt M2;

        // model leaf size
        float precision;

        map(IndexInt n, float precision);

    public:
        // number of data points
        struct data_t {
            pcl::PointXYZ location;
            T content;

            data_t() = default;

            data_t(const pcl::PointXYZ &point, T content) : location(point), content(content) {}
        };

        struct data_normal_t : public data_t {
            pcl::Normal normal;

            data_normal_t() = default;

            data_normal_t(const data_t &data, const pcl::Normal &normal) : data_t(data), normal(normal) {}

            data_normal_t(const pcl::PointXYZ &point, const pcl::Normal &normal, T content)
                    : data_t(point, content), normal(normal) {}
        };

        using data_function = std::function<data_t(IndexInt)>;
        using data_normal_function = std::function<data_normal_t(IndexInt)>;
        using content_function = std::function<T(IndexInt)>;

        map(const data_normal_function &data_normal, IndexInt n, float precision);

        map(const data_function &data, IndexInt n, float precision);

        static Eigen::Matrix4f get_rotation_matrix(const Eigen::Vector3f &before, const Eigen::Vector3f &after) {
            Eigen::Vector3f rotation_axis = before.cross(after);
            float rotation_angle = std::acos(before.dot(after) / before.norm());
            Eigen::AngleAxisf angle_axis(rotation_angle, rotation_axis);
            Eigen::Matrix3f rotation_matrix = angle_axis.matrix();
            Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Zero();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    transform_matrix(i, j) = rotation_matrix(i, j);
                }
            }
            transform_matrix(3, 3) = 1;
            return transform_matrix;
        }

        static void show_cloud(const pcl::PointCloud<pcl::PointNormal>::Ptr &cloud) {
            pcl::visualization::PCLVisualizer viewer("cloud");

            viewer.addPointCloud<pcl::PointNormal>(cloud);
            viewer.addPointCloudNormals<pcl::PointNormal>(cloud, 10, 0.1f, "normal");

            viewer.spin();
        }

        void init_map(const pcl::PointCloud<pcl::PointNormal>::Ptr &cloud, const content_function &content) {
            // show_cloud(cloud);
            const int k = 6;
            part_num = k;
            VSA vsa;
            vsa.setInputCloud<pcl::PointNormal>(cloud);
            vsa.setMetricOption(2);
            vsa.setEps(1);
            vsa.setK(k);
            auto res = vsa.compute();
            auto proxies = vsa.getProxies();

//#ifdef PSH_DEBUG
            // debug: print the size of each bucket
            for (const auto &it : res) {
                std::cout << it.size() << ' ';
            }
            std::cout << std::endl;
            // end debug
//#endif

            std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> transformed_clouds;
            std::generate_n(std::back_inserter(transformed_clouds), k, [] {
                pcl::PointCloud<pcl::PointNormal>::Ptr ret(new pcl::PointCloud<pcl::PointNormal>);
                return ret;
            });

            for (int bucket = 0; bucket < k; bucket++) {
                // load source cloud
                pcl::PointCloud<pcl::PointNormal>::Ptr source(new pcl::PointCloud<pcl::PointNormal>);
                source->points.reserve(res[bucket].size());
                for (auto point_index: res[bucket]) {
                    source->points.push_back(cloud->points[point_index]);
                }

                const auto &proxy = proxies[bucket];
                Eigen::Vector3f N(proxy.normal_x, proxy.normal_y, proxy.normal_z);
                Eigen::Vector3f X(0.0f, 0.0f, 1.0f);
                Eigen::Matrix4f transform_matrix = get_rotation_matrix(N, X);
                transform_matrix(2, 3) = 0.1f * bucket;
                pcl::transformPointCloud(*source, *transformed_clouds[bucket], transform_matrix);
            }
            Eigen::Vector3f v = Eigen::Vector3f::Zero();

#ifdef PSH_DEBUG
            // debug: draw
            pcl::visualization::PCLVisualizer viewer("PSH_rotate");
            pcl::visualization::PCLVisualizer viewer2("VSA");

            std::default_random_engine colorGenerator(0);
            std::uniform_int_distribution random(0, 255);

            for (int bucket = 0; bucket < k; bucket++) {
                pcl::PointCloud<pcl::PointNormal>::Ptr source(new pcl::PointCloud<pcl::PointNormal>);
                source->points.reserve(res[bucket].size());
                for (auto point_index: res[bucket]) {
                    source->points.push_back(cloud->points[point_index]);
                }

                int color_x = random(colorGenerator);
                int color_y = random(colorGenerator);
                int color_z = random(colorGenerator);

                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal>
                        handler(transformed_clouds[bucket], color_x, color_y, color_z);
                viewer.addPointCloud(transformed_clouds[bucket], handler, std::string(bucket, 'c'));
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> handler2(source, color_x, color_y,
                                                                                            color_z);
                viewer2.addPointCloud(source, handler2, std::string(bucket, 'd'));
            }
            viewer.spin();
            viewer2.spin();
            // end debug
#endif
            create_hash_buckets(transformed_clouds, proxies, content);
        }

        void init_layer(
                layer_map &layer,
                const pcl::PointCloud<pcl::PointNormal>::Ptr &cloud,
                const VSA::Proxy &proxy,
                std::vector<typename layer_map::data_t> &out
        ) const {
            const uint d = layer_dimension;
//            IndexInt _n = cloud->points.size();
//            float min_bound[d];
//            float max_bound[d];
//            for (uint i = 0; i < d; i++) {
//                min_bound[i] = INFINITY;
//                max_bound[i] = -INFINITY;
//            }
//            for (const auto &point:*cloud) {
//                for (uint i = 0; i < d; i++) {
//                    min_bound[i] = std::min(point.data[i], min_bound[i]);
//                    max_bound[i] = std::max(point.data[i], max_bound[i]);
//                }
//            }
//            float width = 0;
//            for (uint i = 0; i < d; i++) {
//                width = std::max(width, max_bound[i] - min_bound[i]);
//            }
//            PosInt _u_bar = std::ceil(width / precision);
//            layer.init(_n, _u_bar, min_bound, precision, proxy);
            IndexInt _n = cloud->points.size();

            PosInt zoom = std::round(1.0f / precision);
            VALUE(zoom);
            std::vector<typename layer_map::data_t>(_n).swap(out);
            std::vector<point<d, LL>> temp(_n);
            for (IndexInt i = 0; i < _n; i++) {
                for (uint j = 0; j < d; j++) {
                    temp[i][j] = std::round(cloud->points[i].data[j] * zoom);
                }
            }
            LL minbound[d];
            LL maxbound[d];
            for (uint i = 0; i < d; i++) {
                minbound[i] = INT64_MAX;
                maxbound[i] = INT64_MIN;
            }
            for (IndexInt i = 0; i < _n; i++) {
                for (uint j = 0; j < d; j++) {
                    minbound[j] = std::min(minbound[j], temp[i][j]);
                    maxbound[j] = std::max(maxbound[j], temp[i][j]);
                }
            }
            for (IndexInt i = 0; i < _n; i++) {
                auto &point = out[i].location;
                for (uint j = 0; j < d; j++) {
                    point[j] = temp[i][j] - minbound[j];
                }
            }
            LL width = 0;
            for (uint i = 0; i < d; i++) {
                width = std::max(width, maxbound[i] - minbound[i]);
            }
            PosInt _u_bar = width;
            OffsetInt _offset[d];
            for (uint i = 0; i < d; i++) {
                _offset[i] = minbound[i];
            }
            std::ofstream fout("out.txt");
            for (IndexInt i = 0; i < _n; i++) {
                auto &point = out[i].location;
                fout << "cloud: " << cloud->points[i].x << ' ' << cloud->points[i].y << std::endl;
                fout << "transformed: " << point.data.x << ' ' << point.data.y << std::endl;
            }
            fout.close();
            layer.init(_n, _u_bar, _offset, precision, proxy);
        }

        void convert_points(const layer_map &layer, const pcl::PointCloud<pcl::PointNormal>::Ptr &cloud,
                            std::vector<point<layer_dimension, PosInt>> &out) {
            const uint d = layer_dimension;
            auto traslation_matrix = layer.get_translation_matrix();
            auto scale_matrix = layer.get_scale_matrix();
            auto transformation_matrix = scale_matrix * traslation_matrix;
            VALUE_MATRIX(traslation_matrix);
            VALUE_MATRIX(scale_matrix);
            VALUE_MATRIX(transformation_matrix);

            assert(layer.n == cloud->points.size());
            std::vector<point<d, PosInt>>(layer.n).swap(out);

            pcl::PointCloud<pcl::PointNormal>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointNormal>);
            pcl::transformPointCloud(*cloud, *transformed_cloud, transformation_matrix);
            for (IndexInt i = 0; i < layer.n; i++) {
                const auto &point = transformed_cloud->points[i];
                for (uint j = 0; j < d; j++) {
                    out[i][j] = (PosInt) std::floor(point.data[j]);
                }
            }
            std::cout << "ok" << std::endl;
            std::ofstream fout("out.txt");
            for (IndexInt i = 0; i < layer.n; i++) {
                fout << cloud->points[i].x << ' ' << cloud->points[i].y << std::endl;
                fout << out[i].data.x << ' ' << out[i].data.y << std::endl;
            }
            fout.close();
        }

        void create_hash_buckets(
                const std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> &clouds,
                const std::vector<VSA::Proxy> &proxies,
                const content_function &content) {
            assert(part_num == clouds.size());

            hash_buckets.resize(part_num);
            for (IndexInt i = 0; i < part_num; i++) {
                bool create_succeed = false;
                layer_map &layer = hash_buckets[i];
                const pcl::PointCloud<pcl::PointNormal>::Ptr &layer_cloud = clouds[i];
                std::vector<typename layer_map::data_t> scaled_cloud;
                init_layer(layer, layer_cloud, proxies[i], scaled_cloud);


//              TODO : create_hash
//                int fail_times = -1;
//                do {
//                    fail_times++;
//                } while (!create_succeed);
            }
        }

        static pcl::PointCloud<pcl::Normal>::Ptr get_normals(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
            ne.setInputCloud(cloud);
            ne.setSearchMethod(tree);
            ne.setKSearch(50);
            pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
            ne.compute(*cloud_normals);
            return cloud_normals;
        }

        IndexInt prime() {
            static const std::vector<IndexInt> primes{
                    53, 97, 193, 389, 769, 1543,
                    3079, 6151, 12289, 24593, 49157, 98317,
                    196613, 393241, 786433, 1572869, 3145739, 6291469};
            static std::uniform_int_distribution<IndexInt> prime_dist(
                    0, primes.size() - 1);

            return primes[prime_dist(generator)];
        }

    private:
        struct entry {
            // stored data
            T contents;
            // parameter for the hash function
            HashInt k;
            // result of the hash function
            HashInt hk;
        };

        struct layer_map {
            static const uint d{layer_dimension};

            // point numbers
            IndexInt n;
            // width of the hash table
            PosInt m_bar;
            // size of the hash table
            IndexInt m;
            // width of the offset table
            PosInt r_bar;
            // size of the offset table
            IndexInt r;
            // u_bar is the limit of the domain in each dimension
            PosInt u_bar;
            // u is the number of elements in the domain
            IndexInt u;
            // offset table
            std::vector<pcl::PointXYZ> phi;
            // hash table
            std::vector<entry> H;

            float precision;

            OffsetInt offset[d];

            VSA::Proxy proxy;

            struct data_t {
                point<d, PosInt> location;
                T content;
            };

            void set_r_bar(PosInt _r_bar) {
                this->r_bar = _r_bar;
                this->r = std::pow(r_bar, d);
            }

            void init(IndexInt _n, PosInt _u_bar, OffsetInt _offset[d], float _precision, const VSA::Proxy &_proxy) {
                this->n = _n;
                this->m_bar = std::ceil(std::pow(n, 1.0f / d));
                this->m = std::pow(m_bar, d);
                set_r_bar(std::ceil(std::pow(n / d, 1.0f / d)) - 1);
                this->u_bar = _u_bar;
                this->u = std::pow(u_bar, d);
                std::memcpy(this->offset, _offset, sizeof(this->offset));
                this->precision = _precision;
                this->proxy = _proxy;
            }

            [[nodiscard]] Eigen::Matrix4f get_rotation_matrix() const {
                Eigen::Vector3f N(proxy.normal_x, proxy.normal_y, proxy.normal_z);
                Eigen::Vector3f X(0.0f, 0.0f, 1.0f);
                return map::get_rotation_matrix(N, X);
            }

            [[nodiscard]] Eigen::Matrix4f get_translation_matrix() const {
                Eigen::Matrix4f ret = Eigen::Matrix4f::Identity();
                ret(0, 3) = -offset[0];
                ret(1, 3) = -offset[1];
                return ret;
            }

            [[nodiscard]] Eigen::Matrix4f get_scale_matrix() const {
                Eigen::Matrix4f ret = Eigen::Matrix4f::Identity();
                float zoom = 1.0f / precision;
                for (uint i = 0; i < d; i++) {
                    ret(i, i) = zoom;
                }
                return ret;
            }

        };

    };

    template<class T>
    map<T>::map(map::IndexInt n, float precision) : n(n), precision(precision), generator(time(nullptr)) {}

    template<class T>
    map<T>::map(const map::data_normal_function &data_normal, map::IndexInt n, float precision) : map(n, precision) {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normal(new pcl::PointCloud<pcl::PointNormal>);
        cloud_with_normal->points.reserve(n);
        for (IndexInt i = 0; i < n; i++) {
            pcl::PointNormal pointNormal;
            pcl::copyPoint(data_normal(i).location, pointNormal);
            pcl::copyPoint(data_normal(i).normal, pointNormal);
            cloud_with_normal->points.push_back(pointNormal);
        }
        init_map(cloud_with_normal, [&](IndexInt i) -> T {
            return data_normal(i).content;
        });
    }

    template<class T>
    map<T>::map(const map::data_function &data, map::IndexInt n, float precision) : map(n, precision) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr inCloud(new pcl::PointCloud<pcl::PointXYZ>);
        inCloud->points.reserve(n);
        for (IndexInt i = 0; i < n; i++) {
            inCloud->points.push_back(data(i).location);
        }
        auto normals = get_normals(inCloud);

        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normal(new pcl::PointCloud<pcl::PointNormal>);
        pcl::concatenateFields(*inCloud, *normals, *cloud_with_normal);
        init_map(cloud_with_normal, [&](IndexInt i) -> T {
            return data(i).content;
        });
    }
}

#endif
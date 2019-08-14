#pragma once

#ifndef _PSH_HPP_
#define _PSH_HPP_

#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>
#include <functional>
#include <set>
#include <unordered_set>
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
#include "util.hpp"

//#define PSH_DEBUG

#define VALUE(x) std::cout << #x "=" << x << std::endl
#define VALUE_MATRIX(x) std::cout << #x "=" << std::endl << x << std::endl

namespace psh {

    const uint layer_dimension = 3;

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

        // partition number from VSA
        IndexInt part_num;

        std::vector<layer_map> hash_buckets;

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
                std::vector<decltype(layer_map::data_t::location)> &out
        ) const {
            const uint d = layer_dimension;
            IndexInt _n = cloud->points.size();
            float min_bound[d];
            float max_bound[d];
            for (uint i = 0; i < d; i++) {
                min_bound[i] = INFINITY;
                max_bound[i] = -INFINITY;
            }
            for (const auto &point:*cloud) {
                for (uint i = 0; i < d; i++) {
                    min_bound[i] = std::min(point.data[i], min_bound[i]);
                    max_bound[i] = std::max(point.data[i], max_bound[i]);
                }
            }
            float width = 0;
            for (uint i = 0; i < d; i++) {
                width = std::max(width, max_bound[i] - min_bound[i]);
            }
            PosInt _u_bar = std::ceil(width / precision);
            layer.init(_n, _u_bar, min_bound, precision, proxy);

            auto ok_func = [](const std::vector<decltype(layer_map::data_t::location)> &data) -> bool {
                std::unordered_set<decltype(layer_map::data_t::location)> set;
                bool ok = true;
                for (const auto &point:data) {
                    if (set.count(point)) {
                        ok = false;
                        break;
                    }
                    set.insert(point);
                }
                return ok;
            };
            auto search_zoom_factor_func = [&](float L, float R, float step, uint mask) -> float {
                float zoom_factor[d];
                while (R - L >= step) {
                    float M = L + (R - L) / 2.0f;
                    for (uint i = 0; i < d; i++) {
                        if (mask & ((uint) 1 << i)) {
                            zoom_factor[i] = M;
                        }
                    }
                    layer.setZoom(zoom_factor, mask);
                    convert_points(layer, cloud, out);
                    if (ok_func(out)) {
                        R = M - step;
                    } else {
                        L = M + step;
                    }
                }
                for (uint i = 0; i < d; i++) {
                    if (mask & ((uint) 1 << i)) {
                        zoom_factor[i] = R;
                    }
                }
                layer.setZoom(zoom_factor, mask);
                assert(R < 5.0f);
                return R;
            };

            float step = 0.01f;
            float res = search_zoom_factor_func(0.5f, 5.0f, step, 7);
            for (uint i = 0; i < 3; i++) {
                float start = 0.5f;
                if (i == 2) start = 0.0f;
                search_zoom_factor_func(start, res, step, (uint) 1 << i);
            }
            std::cout << layer.zoom_factor[0] << ' ' << layer.zoom_factor[1] << ' ' << layer.zoom_factor[2]
                      << std::endl;
            convert_points(layer, cloud, out);
            static int cnt = 0;
            cnt++;
            char filename[10];
            sprintf(filename, "out%d.txt", cnt);
            std::ofstream fout(filename);
            for (IndexInt i = 0; i < _n; i++) {
                fout << out[i] << std::endl;
            }
            fout.close();
        }

        static void convert_points(const layer_map &layer, const pcl::PointCloud<pcl::PointNormal>::Ptr &cloud,
                                   std::vector<decltype(layer_map::data_t::location)> &out) {
            const uint d = layer_dimension;
            auto translation_matrix = layer.get_translation_matrix();
            auto scale_matrix = layer.get_scale_matrix();
            auto transformation_matrix = scale_matrix * translation_matrix;

            assert(layer.n == cloud->points.size());
            std::vector<decltype(layer_map::data_t::location)>(layer.n).swap(out);

            pcl::PointCloud<pcl::PointNormal>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointNormal>);
            pcl::transformPointCloud(*cloud, *transformed_cloud, transformation_matrix);
            for (IndexInt i = 0; i < layer.n; i++) {
                const auto &point = transformed_cloud->points[i];
                for (uint j = 0; j < d; j++) {
                    out[i][j] = (PosInt) std::floor(point.data[j]);
                }
            }
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
                std::vector<decltype(layer_map::data_t::location)> scaled_cloud;
                init_layer(layer, layer_cloud, proxies[i], scaled_cloud);
                layer.build([&](IndexInt i) {
                    return typename layer_map::data_t(scaled_cloud[i], content(i));
                });
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

    private:
        class layer_map {
        public:
            static const uint d{layer_dimension};

            class entry;

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
            std::vector<point<d, PosInt>> phi;
            // hash table
            std::vector<entry> H;

            // three primes for use in hashing
            IndexInt M0;
            IndexInt M1;
            IndexInt M2;

            std::default_random_engine generator;

            float precision;

            float offset_factor[d];

            float zoom_factor[d];

            VSA::Proxy proxy;

            struct data_t {
                point<d, PosInt> location;
                T contents;

                data_t() = default;

                data_t(const point<d, PosInt> &location, const T &content) : location(location), contents(content) {}
            };

            using data_function = std::function<data_t(IndexInt)>;

            layer_map();

            void set_r_bar(PosInt _r_bar) {
                this->r_bar = _r_bar;
                this->r = std::pow(r_bar, d);
            }

            void init(IndexInt _n, PosInt _u_bar, const float _offset[d], float _precision, const VSA::Proxy &_proxy) {
                this->n = _n;
                this->m_bar = std::ceil(std::pow(n, 1.0f / d));
                this->m = std::pow(m_bar, d);
//                set_r_bar(std::ceil(std::pow(n / d, 1.0f / d)) - 1);
                set_r_bar(0);
                this->u_bar = _u_bar;
                this->u = std::pow(u_bar, d);
                std::memcpy(this->offset_factor, _offset, sizeof(this->offset_factor));
                this->precision = _precision;
                this->proxy = _proxy;
            }

            void setZoom(float x, float y, float z) {
                zoom_factor[0] = x;
                zoom_factor[1] = y;
                zoom_factor[2] = z;
            }

            void setZoom(const float _zoom_factor[d], uint mask) {
                for (uint i = 0; i < d; i++) {
                    if (mask & ((uint) 1 << i)) {
                        zoom_factor[i] = _zoom_factor[i];
                    }
                }
            }

            [[nodiscard]] Eigen::Matrix4f get_rotation_matrix() const {
                Eigen::Vector3f N(proxy.normal_x, proxy.normal_y, proxy.normal_z);
                Eigen::Vector3f X(0.0f, 0.0f, 1.0f);
                return map::get_rotation_matrix(N, X);
            }

            [[nodiscard]] Eigen::Matrix4f get_translation_matrix() const {
                Eigen::Matrix4f ret = Eigen::Matrix4f::Identity();
                for (uint i = 0; i < d; i++) {
                    ret(i, 3) = -offset_factor[i];
                }
                return ret;
            }

            [[nodiscard]] Eigen::Matrix4f get_scale_matrix() const {
                Eigen::Matrix4f ret = Eigen::Matrix4f::Identity();
                float zoom = 1.0f / precision;
                for (uint i = 0; i < d; i++) {
                    ret(i, i) = zoom * zoom_factor[i];
                }
                return ret;
            }

            void build(const data_function &data) {
                // generate primes, M0 must be different from M1
                M0 = prime();
                while ((M1 = prime()) == M0);
                M2 = prime();

                VALUE(m);
                VALUE(uint(m_bar));

                VALUE(M0);
                VALUE(M1);
                VALUE(M2);

                std::uniform_int_distribution<IndexInt> m_dist(0, m - 1);
                bool create_succeeded = false;
                int fail = -1;
                do {
                    // if we fail, we try again with a larger offset table
                    fail++;
                    set_r_bar(r_bar + d);

                    VALUE(r);
                    VALUE(uint(r_bar));

                    create_succeeded = create(data, m_dist);
                } while (!create_succeeded);
                std::cout << "=======Failed for " << fail << " times=========" << std::endl;
            }


//        private:
            // internal data structures

            // a bucket is simply a vector<data_t> which is then sorted
            // by the original phi_index (descending order)
            struct bucket : public std::vector<data_t> {
                IndexInt phi_index;

                bucket(IndexInt phi_index) : phi_index(phi_index) {}

                friend bool operator<(const bucket &lhs, const bucket &rhs) {
                    return lhs.size() > rhs.size();
                }
            };

            // data type for each entry in the hash table
            struct entry {
                // stored data
                T contents;
                // parameter for the hash function
                HashInt k;
                // result of the hash function
                HashInt hk;

                entry() : contents(T()), k(1), hk(1) {};

                entry(const data_t &data, IndexInt M2) : contents(data.contents), k(1) {
                    rehash(data, M2);
                }

                static constexpr HashInt h(const point<d, PosInt> &p, IndexInt M2, HashInt k) {
                    // p dot {k, k * k, k * k * k...} * M2
                    // creds to David
                    return p * point<d, PosInt>::increasing_pow(k) * M2;
                }

                void rehash(const point<d, PosInt> &location, IndexInt M2, HashInt new_k = 1) {
                    k = new_k;
                    hk = h(location, M2, k);
                }

                void rehash(const data_t &data, IndexInt M2, HashInt new_k = 1) {
                    rehash(data.location, M2, new_k);
                }

                constexpr bool equals(const point<d, PosInt> &p, IndexInt M2) const {
                    return hk == h(p, M2, k);
                }
            };

            // a larger version of entry to also keep the original location
            // later discarded (using slicing!) for memory efficiency
            struct entry_large : public entry {
                point<d, PosInt> location;

                entry_large() : entry(), location(point<d, PosInt>()) {};

                entry_large(const data_t &data, IndexInt M2)
                        : entry(data, M2), location(data.location) {};

                void rehash(IndexInt M2) {
                    entry::rehash(location, M2, entry::k + 1);
                };
            };

            // internal functions

            // provides the index in the hash table for a given position in the
            // domain, optionally with a temporary offset table
            point<d, PosInt> h(const point<d, PosInt> &p, const decltype(phi) &phi_hat) const {
                auto h0 = p * M0;
                auto h1 = p * M1;
                auto i = point_to_index(h1, r_bar, r);
                auto offset = phi_hat[i];
                return h0 + offset;
            }

            point<d, PosInt> h(const point<d, PosInt> &p) const {
                return h(p, phi);
            }

            // returns a random prime from a small predefined list
            IndexInt prime() {
                static const std::vector<IndexInt> primes{
                        53, 97, 193, 389, 769, 1543,
                        3079, 6151, 12289, 24593, 49157, 98317,
                        196613, 393241, 786433, 1572869, 3145739, 6291469};
                static std::uniform_int_distribution<IndexInt> prime_dist(
                        0, primes.size() - 1);

                return primes[prime_dist(generator)];
            }

            // tried to create the hash table given a certain offset table size
            bool create(const data_function &data, std::uniform_int_distribution<IndexInt> &m_dist) {
                // _hats are temporary variables, later moved into the real vectors
                decltype(phi) phi_hat(r);
                std::vector<entry_large> H_hat(m);
                // lookup for whether a certain slot in the hash table contains an
                // entry
                std::vector<bool> H_b_hat(m, false);
                std::cout << "creating " << r << " buckets" << std::endl;

                if (bad_m_r()) return false;

                // find out what order we should do the hashing to optimize success
                // rate
                auto buckets = create_buckets(data);
                std::cout << "jiggling offsets" << std::endl;

                for (IndexInt i = 0; i < buckets.size(); i++) {
                    // if a bucket is empty, then the rest will also be empty
                    if (buckets[i].size() == 0) break;
                    if (i % (buckets.size() / 10) == 0)
                        std::cout << (100 * i) / buckets.size() << "% done"
                                  << std::endl;

                    // try to jiggle the offsets until an injective mapping is found
                    if (!jiggle_offsets(H_hat, H_b_hat, phi_hat, buckets[i], m_dist)) {
                        return false;
                    }
                }

                std::cout << "done!" << std::endl;
                phi = std::move(phi_hat);
                if (!hash_positions(data, H_hat)) return false;
                H.reserve(H_hat.size());
                std::copy(H_hat.begin(), H_hat.end(), std::back_inserter(H));

                return true;
            }

            // certain values for m_bar and r_bar are bad, empirically found to be
            // if: m_bar is coprime with r_bar <==> gcd(m_bar, r_bar) != 1 <==>
            // m_bar % r_bar ∈ {1, r_bar - 1} creds to Euclid
            bool bad_m_r() const {
                auto m_mod_r = m_bar % r_bar;
                return m_mod_r == 1 || m_mod_r == r_bar - 1;
            }

            // creates buckets, each buckets corresponds to one entry in the offset
            // table they are then sorted by their index in the offset table so we
            // can assign the largest buckets first
            std::vector<bucket> create_buckets(const data_function &data) const {
                std::vector<bucket> buckets;
                buckets.reserve(r);
                {
                    IndexInt i = 0;
                    std::generate_n(std::back_inserter(buckets), r, [&] { return bucket(i++); });
                }

                for (IndexInt i = 0; i < n; i++) {
                    auto element = data(i);
                    auto h1 = element.location * M1;
                    buckets[point_to_index(h1, r_bar, r)].push_back(element);
                }

                std::cout << "buckets created" << std::endl;

                sort(buckets.begin(), buckets.end());
                std::cout << "buckets sorted" << std::endl;

                return buckets;
            }

            // jiggle offsets to avoid collisions
            bool jiggle_offsets(std::vector<entry_large> &H_hat,
                                std::vector<bool> &H_b_hat, decltype(phi) &phi_hat,
                                const bucket &b,
                                std::uniform_int_distribution<IndexInt> &m_dist) {
                // start at a random point
                auto start_offset = m_dist(generator);

                bool found = false;
                point<d, PosInt> found_offset;

                for (IndexInt i = 0; i < r && !found; i++) {
                    // wrap around m to stay inside the table
                    auto phi_offset = index_to_point<d>(
                            (start_offset + i) % m, m_bar, m);

                    bool collision = false;
                    for (auto &element : b) {
                        auto h0 = element.location * M0;
                        auto h1 = element.location * M1;
                        auto index = point_to_index(h1, r_bar, r);
                        // use existing offsets for others, but if
                        // the current index is the one we're
                        // jiggling, we use the temporary offset
                        auto offset = index == b.phi_index ? phi_offset : phi_hat[index];
                        auto hash = h0 + offset;

                        // if the index is already used, this offset
                        // is invalid
                        collision = H_b_hat[point_to_index(hash, m_bar, m)];
                        if (collision) break;
                    }

                    // if there were no collisions, we succeeded
                    if (!collision) {
                        if (!found) {
                            found = true;
                            found_offset = phi_offset;
                        }
                    }
                }

                if (found) {
                    // if we found a valid offset, insert it
                    phi_hat[b.phi_index] = found_offset;
                    insert(b, H_hat, H_b_hat, phi_hat);
                    return true;
                }
                return false;
            }

            // permanently inserts a bucket into a temporary hash table
            void insert(const bucket &b, std::vector<entry_large> &H_hat,
                        std::vector<bool> &H_b_hat, const decltype(phi) &phi_hat) {
                for (auto &element : b) {
                    auto hashed = h(element.location, phi_hat);
                    auto i = point_to_index(hashed, m_bar, m);
                    H_hat[i] = entry_large(element, M2);
                    // mark off the slot as used
                    H_b_hat[i] = true;
                }
            }

            bool hash_positions(const data_function &data, std::vector<entry_large> &H_hat) {
                // u_bar - 1 to get the highest indices in each direction
                // width is assumed to be equal in all directions
                // and then add 1 to get the size, not the index
                IndexInt domain_i_max =
                        point_to_index(point<d, PosInt>::repeating(u_bar) - PosInt(1), u_bar, IndexInt(-1)) + 1;


                // in the first sweep we go through all points in the domain without
                // a data entry
                std::vector<bool> indices(m, false);
                {
                    std::vector<bool> data_b(domain_i_max);
                    for (IndexInt i = 0; i < n; i++) {
                        data_b[point_to_index(data(i).location, u_bar,
                                              domain_i_max)] = true;
                    }
                    for (IndexInt i = 0; i < domain_i_max; i++) {
                        if (data_b[i]) {
                            continue;
                        }

                        auto p = index_to_point<d, PosInt>(i, u_bar, domain_i_max);
                        auto l = point_to_index(h(p), m_bar, m);

                        // if their position hash collides with the existing
                        // element..
                        if (H_hat[l].hk == entry::h(p, M2, 1)) {
                            // ..remember the index
                            indices[l] = true;
                        }
                    }
                    std::cout << "data size: " << n << std::endl;
                    std::cout << "indices size: " << indices.size() << std::endl;
                }

                // in the second sweep we go through the stored indices, and
                // remember all points in the domain that map to that same index,
                // regardless of whether that point has data or not
                std::unordered_map<IndexInt, std::vector<IndexInt>> collisions;
                for (IndexInt i = 0; i < domain_i_max; i++) {
                    // for each point p in original image

                    auto p = index_to_point<d, PosInt>(i, u_bar, domain_i_max);
                    auto l = point_to_index(h(p), m_bar, m);

                    // collect everyone that maps to the same thing
                    if (indices[l]) {
                        collisions[l].push_back(i);
                    }
                }

                // in the third sweep we try to change the positional hash parameter
                // until it works
                bool success = true;
                for (const decltype(collisions)::value_type &kvp:collisions) {
                    if (!fix_k(H_hat[kvp.first], kvp.second)) {
                        success = false;
                        break;
                    }
                }
                return success;
            }

            // try all values for the positional hash parameter until it works
            bool fix_k(entry_large &H_entry, const std::vector<IndexInt> &collisions) {
                H_entry.rehash(M2);
                // if k == 0, we've rolled around and already tried all the values
                if (H_entry.k == 0) return false;

                bool success = true;
                for (IndexInt i : collisions) {
                    // i is the index in the domain
                    // fail if one of these have the same positional hash as the
                    // entry in the hash table
                    auto p = index_to_point<d, PosInt>(i, u_bar, IndexInt(-1));
                    auto hk = entry::h(p, M2, H_entry.k);
                    if (H_entry.location != p && H_entry.hk == hk) {
                        success = false;
                        break;
                    }
                }
                // if we didn't find a valid k, recursively move on to the next k
                if (!success) return fix_k(H_entry, collisions);
                return true;
            }
        };

    };

    template<class T>
    map<T>::layer_map::layer_map() : generator(time(nullptr)) {
        for (uint i = 0; i < d; i++) {
            zoom_factor[i] = 1.0f;
        }
    }

    template<class T>
    map<T>::map(map::IndexInt n, float precision) : n(n), precision(precision) {}

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
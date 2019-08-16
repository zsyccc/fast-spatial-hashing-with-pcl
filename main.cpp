#define PCL_NO_PRECOMPILE

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d.h>
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <mesh_sampling.h>
#include <psh.hpp>

using namespace std;
using namespace pcl;
const int default_number_samples = 100000;
const float default_leaf_size = 0.01f;

int main(int argc, char **argv) {
    using pixel = size_t;
    using map = psh::map<pixel>;

    if (argc < 2) {
        console::print_error("no enough param\n");
        return (-1);
    }

    int sample_points = default_number_samples;
    console::parse_argument(argc, argv, "-n_samples", sample_points);
    float leaf_size = default_leaf_size;
    console::parse_argument(argc, argv, "-leaf_size", leaf_size);

    map::VSAOption vsaOption(6, 0.01f, 2);
    console::parse_argument(argc, argv, "-k", vsaOption.part_num);
    console::parse_argument(argc, argv, "-eps", vsaOption.eps);
    console::parse_argument(argc, argv, "-metric", vsaOption.metricOption);


    std::vector<int> obj_file_indices = console::parse_file_extension_argument(argc, argv, ".obj");

    if (obj_file_indices.size() != 1) {
        console::print_error("Need a single input OBJ file to continue.\n");
        return (-1);
    }
    const char *filename = argv[obj_file_indices[0]];
    PointCloud<PointXYZ>::Ptr voxel_cloud = mesh_sampling(filename, sample_points, leaf_size);

//    PointCloud<Normal>::Ptr normals = get_normals(voxel_cloud);
//    PointCloud<PointNormal>::Ptr pNormal(new PointCloud<PointNormal>);
//    concatenateFields(*voxel_cloud, *normals, *pNormal);

    float zoom = 1.0f / default_leaf_size;
//    for (auto &point:*voxel_cloud) {
//        for (int i = 0; i < 3; i++) {
//            point.data[i] = (float) std::floor(point.data[i] * zoom) / zoom;
//        }
//    }

    map::data_function get_data_func = [&](size_t i) -> map::data_t {
        pcl::PointXYZ point;
        pcl::copyPoint(voxel_cloud->points[i], point);
        for (int di = 0; di < 3; di++) {
            point.data[di] = (float) std::floor(point.data[di] * zoom) / zoom;
        }
        return map::data_t(point, i);
    };

    map s(get_data_func, voxel_cloud->points.size(), leaf_size, vsaOption);


    std::cout << "exhaustive test" << std::endl;
    int cnt = 0;
    for (size_t i = 0; i < voxel_cloud->points.size(); i++) {
        try {
            size_t data = s.get(get_data_func(i).location);
            if (data != i) {
                std::cout << "error at" << ' ' << get_data_func(i).location << std::endl;
            }
        } catch (const std::out_of_range &e) {
            std::cout << e.what() << ' ' << get_data_func(i).location << std::endl;
            cnt++;
        }
    }
    std::cout << "fail = " << cnt << std::endl;
    std::cout << "total = " << voxel_cloud->points.size() << std::endl;
    std::cout << "done" << std::endl;

    return 0;
}

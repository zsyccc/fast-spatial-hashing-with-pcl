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
    if (argc < 2) {
        console::print_error("no enough param\n");
        return (-1);
    }

    std::vector<int> obj_file_indices =
            console::parse_file_extension_argument(argc, argv, ".obj");

    if (obj_file_indices.size() != 1) {
        console::print_error("Need a single input OBJ file to continue.\n");
        return (-1);
    }
    const char *filename = argv[obj_file_indices[0]];
    PointCloud<PointXYZ>::Ptr voxel_cloud =
            mesh_sampling(filename, default_number_samples, default_leaf_size);

    using pixel = bool;
    using map = psh::map<pixel>;

//    PointCloud<Normal>::Ptr normals = get_normals(voxel_cloud);
//    PointCloud<PointNormal>::Ptr pNormal(new PointCloud<PointNormal>);
//    concatenateFields(*voxel_cloud, *normals, *pNormal);

    float zoom = 1.0f / default_leaf_size;
//    for (auto &point:*voxel_cloud) {
//        for (int i = 0; i < 3; i++) {
//            point.data[i] = (float) std::floor(point.data[i] * zoom) / zoom;
//        }
//    }
//    std::ofstream fout("out.txt", ios::out);
    map::data_function get_data_func = [&](size_t i) -> map::data_t {
        pcl::PointXYZ point;
        pcl::copyPoint(voxel_cloud->points[i], point);
        for (int di = 0; di < 3; di++) {
            point.data[di] = (float) std::floor(point.data[di] * zoom) / zoom;
        }
//        fout << point << std::endl;
        return map::data_t(point, true);
    };
//    for (size_t i = 0; i < voxel_cloud->points.size(); i++) get_data_func(i);
//    fout.close();

    map s(get_data_func, voxel_cloud->points.size(), default_leaf_size);

    return 0;
}

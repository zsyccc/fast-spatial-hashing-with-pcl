#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <mesh_sampling.h>

using namespace std;
using namespace pcl;
const int default_number_samples = 100000;
const float default_leaf_size = 0.01f;

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("no enough param\n");
        return (-1);
    }

    std::vector<int> obj_file_indices =
        console::parse_file_extension_argument(argc, argv, ".obj");

    if (obj_file_indices.size() != 1) {
        console::print_error("Need a single input OBJ file to continue.\n");
        return (-1);
    }
    const char* filename = argv[obj_file_indices[0]];
    auto voxel_cloud = mesh_sampling_with_normal(
        filename, default_number_samples, default_leaf_size);

    visualization::PCLVisualizer viewer("VOXELIZED SAMPLES CLOUD");
    visualization::PointCloudColorHandlerCustom<PointNormal> handler(
        voxel_cloud, 255, 255, 255);
    // viewer.addPointCloud<pcl::PointNormal>(voxel_cloud);
    viewer.addPointCloud(voxel_cloud, handler, "r");
    // viewer.addPointCloudNormals<pcl::PointNormal>(voxel_cloud, 1, 0.2f,
    //                                               "cloud_normals");
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0);
    // viewer.addCoordinateSystem(1.0, "cloud", 0);
    viewer.spin();
    return 0;
}

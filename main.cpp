#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <mesh_sampling.h>

using namespace std;
using namespace pcl;
const int default_number_samples = 100000;
const float default_leaf_size = 0.01f;

pcl::PointCloud<pcl::Normal>::Ptr get_normals(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(50);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
        new pcl::PointCloud<pcl::Normal>);
    ne.compute(*cloud_normals);
    return cloud_normals;
}

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
    PointCloud<PointXYZ>::Ptr voxel_cloud =
        mesh_sampling(filename, default_number_samples, default_leaf_size);

    PointCloud<Normal>::Ptr normals = get_normals(voxel_cloud);
    PointCloud<PointNormal>::Ptr pNormal(new PointCloud<PointNormal>);
    concatenateFields(*voxel_cloud, *normals, *pNormal);
    // pNormal = mesh_sampling_with_normal(filename, default_number_samples,
    //                                     default_leaf_size);
    visualization::PCLVisualizer viewer("VOXELIZED SAMPLES CLOUD");
    visualization::PointCloudColorHandlerCustom<PointNormal> handler(
        pNormal, 20, 200, 20);
    // viewer.addPointCloud<pcl::PointNormal>(voxel_cloud);
    viewer.addPointCloud(pNormal, handler, "r");
    viewer.addPointCloudNormals<pcl::PointNormal>(pNormal, 10, 0.2f,
                                                  "cloud_normals");
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0);
    // viewer.addCoordinateSystem(1.0, "cloud", 0);
    viewer.spin();
    printf("%d %d\n", (int)voxel_cloud->points.size(),
           (int)normals->points.size());
    return 0;
}

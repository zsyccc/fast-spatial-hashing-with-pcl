#define PCL_NO_PRECOMPILE
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <mesh_sampling.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>

#include <pcl/io/obj_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/vtk_lib_io.h>

using namespace std;
using namespace pcl;
const int default_number_samples = 100000;
const float default_leaf_size = 0.01f;

struct PointNormalT {
    PCL_ADD_POINT4D;
    PCL_ADD_NORMAL4D;
    uint32_t index;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointNormalT,
    (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
        float, normal_y, normal_y)(float, normal_z, normal_z)(uint32_t, index,
                                                              index))

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
        console::print_error("no enough param\n");
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

    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(
        new pcl::search::KdTree<pcl::PointNormal>);  //定义搜索树对象
    tree2->setInputCloud(pNormal);  //利用点云构建搜索树
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;  //定义三角化对象
    pcl::PolygonMesh triangles;  //存储最终三角化的网格模型
    gp3.setSearchRadius(
        0.025);  //设置连接点之间的最大距离（即为三角形最大边长）为0.025
    //设置各参数特征值，详见本小节前部分对参数设置的描述
    gp3.setMu(
        2.5);  //设置被样本点搜索其邻近点的最远距离为2.5，为了适应点云密度的变化
    gp3.setMaximumNearestNeighbors(100);  //设置样本点可搜索的邻域个数为100
    gp3.setMaximumSurfaceAngle(
        M_PI / 4);  //设置某点法线方向偏离样本点法线方向的最大角度为45度
    gp3.setMinimumAngle(M_PI / 18);  //设置三角化后得到三角形内角最小角度为10度
    gp3.setMaximumAngle(2 * M_PI /
                        3);  //设置三角化后得到三角形内角最大角度为120度
    gp3.setNormalConsistency(false);  //设置该参数保证法线朝向一致
    gp3.setInputCloud(pNormal);  //设置输入点云为有向点云cloud_with_normals
    gp3.setSearchMethod(tree2);  //设置搜索方式为tree2
    gp3.reconstruct(triangles);  //重建提取三角化

    //附加顶点信息
    std::vector<int> parts = gp3.getPartIDs();
    std::vector<int> states = gp3.getPointStates();

    // PolygonMesh mesh;
    // io::loadPolygonFile(filename, mesh);

    visualization::PCLVisualizer viewer("VOXELIZED SAMPLES CLOUD");
    viewer.addPolygonMesh(triangles);
    // visualization::PointCloudColorHandlerCustom<PointNormal> handler(
    //     pNormal, 20, 200, 20);
    // viewer.addPointCloud<pcl::PointNormal>(voxel_cloud);
    // viewer.addPointCloud(pNormal, handler, "r");
    // viewer.addPointCloudNormals<pcl::PointNormal>(pNormal, 10, 0.2f,
    //   "cloud_normals");
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0);
    // viewer.addCoordinateSystem(1.0, "cloud", 0);
    viewer.spin();
    // printf("%d %d\n", (int)voxel_cloud->points.size(),
    //        (int)normals->points.size());
    return 0;
}

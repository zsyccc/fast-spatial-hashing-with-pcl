#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <mesh_sampling.h>
#include <random>

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

double distance(const PointNormal& a, const PointNormal& b) {
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
           (a.z - b.z) * (a.z - b.z);
}

double find_min_dist(const vector<double>& v) {
    double m = v[0];
    int ret = 0;
    for (int i = 1; i < v.size(); i++) {
        if (v[i] < m) {
            m = v[i];
            ret = i;
        }
    }
    return ret;
}

void get_barycenters(const vector<PointCloud<PointNormal>::Ptr>& v,
                     vector<PointNormal>& barycenters) {
    barycenters.clear();
    for (auto pc : v) {
        PointNormal center;
        int cnt = pc->size();
        for (auto p : *pc) {
            center.x += p.x;
            center.y += p.y;
            center.z += p.z;
        }
        center.x /= cnt;
        center.y /= cnt;
        center.z /= cnt;
        barycenters.push_back(center);
    }
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
    // begin cluster

    PointCloud<PointNormal>& points = *pNormal;
    const int k = 50;
    double eps = 1e-3;
    std::default_random_engine generator;
    static std::uniform_int_distribution<int> randint(0, points.size() - 1);
    set<int> seeds;
    vector<PointNormal> barycenters;
    vector<PointCloud<PointNormal>::Ptr> ret;
    int rand_int;
    generate_n(back_inserter(ret), k, []() {
        PointCloud<PointNormal>::Ptr ret(new PointCloud<PointNormal>);
        return ret;
    });
    for (int i = 0; i < k; i++) {
        do {
            rand_int = randint(generator);
        } while (seeds.count(rand_int));
        seeds.insert(rand_int);
        barycenters.push_back(points[rand_int]);
    }
    int gen = 0;
    double e = 0;
    double last_e = -100;
    while (fabs(last_e - e) > eps) {  // iterations
        last_e = e;
        e = 0;
        if (gen) {
            get_barycenters(ret, barycenters);
            for (auto it : ret) {
                (*it).clear();
            }
        }
        for (auto pa : points) {
            vector<double> dist;
            for (auto pb : barycenters) {
                dist.push_back(distance(pa, pb));
            }
            int min_dist = find_min_dist(dist);
            ret[min_dist]->push_back(pa);
        }
        printf("%d:", gen);
        for (int i = 0; i < k; i++) {
            printf("%lu ", ret[i]->size());
            for (int j = 0; j < ret[i]->size(); j++) {
                e += distance((*(ret[i]))[j], barycenters[i]);
            }
        }
        printf("\n");
        gen++;
    }
    // printf("%lu\n", ret.size());
    // end cluster
    // pNormal = mesh_sampling_with_normal(filename, default_number_samples,
    //                                     default_leaf_size);
    std::uniform_int_distribution<int> color(0, 255);
    visualization::PCLVisualizer viewer("VOXELIZED SAMPLES CLOUD");
    int cnt = 0;
    for (auto pointcloud_ptr : ret) {
        // printf("%lu ", pointcloud_ptr->size());
        visualization::PointCloudColorHandlerCustom<PointNormal> handler(
            pointcloud_ptr, color(generator), color(generator),
            color(generator));
        viewer.addPointCloud(pointcloud_ptr, handler, string(cnt++, 'x'));
    }
    // viewer.addPointCloud<pcl::PointNormal>(voxel_cloud);
    // viewer.addPointCloudNormals<pcl::PointNormal>(pNormal, 10, 0.2f,
    //                                               "cloud_normals");
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0);
    // viewer.addCoordinateSystem(1.0, "cloud", 0);
    viewer.spin();
    printf("%d %d\n", (int)voxel_cloud->points.size(),
           (int)normals->points.size());
    return 0;
}

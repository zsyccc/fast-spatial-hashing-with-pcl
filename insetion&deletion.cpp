// deletion and insertion for each iteration
// deletion: merge 2 proxies
float smallest_merge_error = INFINITY;
int smallest_merge_error_index = -1;
for (int i = 0; i < k - 1; i++) {
    std::vector<int> merging_region;
    merging_region.reserve(cur_partition[i].size() +
                           cur_partition[i + 1].size());
    std::copy(cur_partition[i].begin(), cur_partition[i].end(),
              back_inserter(merging_region));
    std::copy(cur_partition[i + 1].begin(), cur_partition[i + 1].end(),
              back_inserter(merging_region));
    Proxy merging_proxy;
    proxy_fitting(merging_region, merging_proxy);
    float merging_error = 0.0f;
    for (auto it : merging_region) {
        merging_error += error_metric(cloud->points[it], merging_proxy);
    }
    if (merging_error < smallest_merge_error) {
        smallest_merge_error = merging_error;
        smallest_merge_error_index = i;
    }
}
std::copy(cur_partition[smallest_merge_error_index + 1].begin(),
          cur_partition[smallest_merge_error_index + 1].end(),
          back_inserter(cur_partition[smallest_merge_error_index]));
// insertion: devide a region into a new region and a point
float largest_error = -1.0f;
int largest_error_index = -1;
float largest_point_error = 0.0f;
int largest_point_error_index = -1;
for (int i = 0; i < k; i++) {
    float region_error = 0.0f;
    float cur_largest_point_error = 0.0f;
    int cur_largest_point_error_index = -1;
    for (int point_idx = 0; point_idx < cur_partition[i].size(); point_idx++) {
        float t = error_metric(cloud->points[point_idx], proxies[i]);
        region_error += t;
        if (t > cur_largest_point_error) {
            cur_largest_point_error = t;
            cur_largest_point_error_index = point_idx;
        }
    }
    if (region_error > largest_error) {
        largest_error = region_error;
        largest_error_index = i;
        largest_point_error = cur_largest_point_error;
        largest_point_error_index = cur_largest_point_error_index;
    }
    // total_error_new += region_error;
}
cur_partition[largest_error_index][largest_point_error_index] =
    cur_partition[largest_error_index].back();
cur_partition[largest_error_index].pop_back();
if (largest_error_index < smallest_merge_error_index) {
    for (int i = smallest_merge_error; i > largest_point_error_index; i--) {
        cur_partition[i].swap(cur_partition[i + 1]);
    }
    cur_partition[largest_error_index + 1].push_back(largest_point_error_index);
} else if (largest_error_index > smallest_merge_error_index) {
    for (int i = smallest_merge_error_index + 1; i <= largest_error_index;
         i++) {
        cur_partition[i].swap(cur_partition[i - 1]);
    }
    cur_partition[largest_error_index].push_back(largest_point_error_index);
} else {
    cur_partition[smallest_merge_error + 1].push_back(
        largest_point_error_index);
}
// deletion and insertion for each iteration
// deletion: merge 2 proxies
float smallest_merge_error = INFINITY;
int smallest_merge_error_index = -1;
for (int i = 0; i < k - 1; i++) {
    std::vector<int> merging_region;
    merging_region.reserve(cur_partition[i].size() +
                           cur_partition[i + 1].size());
    std::copy(cur_partition[i].begin(), cur_partition[i].end(),
              back_inserter(merging_region));
    std::copy(cur_partition[i + 1].begin(), cur_partition[i + 1].end(),
              back_inserter(merging_region));
    Proxy merging_proxy;
    proxy_fitting(merging_region, merging_proxy);
    float merging_error = 0.0f;
    for (auto it : merging_region) {
        merging_error += error_metric(cloud->points[it], merging_proxy);
    }
    if (merging_error < smallest_merge_error) {
        smallest_merge_error = merging_error;
        smallest_merge_error_index = i;
    }
}
std::copy(cur_partition[smallest_merge_error_index + 1].begin(),
          cur_partition[smallest_merge_error_index + 1].end(),
          back_inserter(cur_partition[smallest_merge_error_index]));
// insertion: devide a region into a new region and a point
float largest_error = -1.0f;
int largest_error_index = -1;
float largest_point_error = 0.0f;
int largest_point_error_index = -1;
for (int i = 0; i < k; i++) {
    float region_error = 0.0f;
    float cur_largest_point_error = 0.0f;
    int cur_largest_point_error_index = -1;
    for (int point_idx = 0; point_idx < cur_partition[i].size(); point_idx++) {
        float t = error_metric(cloud->points[point_idx], proxies[i]);
        region_error += t;
        if (t > cur_largest_point_error) {
            cur_largest_point_error = t;
            cur_largest_point_error_index = point_idx;
        }
    }
    if (region_error > largest_error) {
        largest_error = region_error;
        largest_error_index = i;
        largest_point_error = cur_largest_point_error;
        largest_point_error_index = cur_largest_point_error_index;
    }
    // total_error_new += region_error;
}
cur_partition[largest_error_index][largest_point_error_index] =
    cur_partition[largest_error_index].back();
cur_partition[largest_error_index].pop_back();
if (largest_error_index < smallest_merge_error_index) {
    for (int i = smallest_merge_error; i > largest_point_error_index; i--) {
        cur_partition[i].swap(cur_partition[i + 1]);
    }
    cur_partition[largest_error_index + 1].push_back(largest_point_error_index);
} else if (largest_error_index > smallest_merge_error_index) {
    for (int i = smallest_merge_error_index + 1; i <= largest_error_index;
         i++) {
        cur_partition[i].swap(cur_partition[i - 1]);
    }
    cur_partition[largest_error_index].push_back(largest_point_error_index);
} else {
    cur_partition[smallest_merge_error + 1].push_back(
        largest_point_error_index);
}

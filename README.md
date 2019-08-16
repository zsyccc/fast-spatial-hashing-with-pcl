# fast spatial hashing

## 编译说明

```bash
mkdir build
cd build
cmake ..
make
```

## to-do
+ 对barycenters排序,引起抖动
+ insert和delete并没有产生有利的结果
+ metric1在k很大时(cube.obj设置k=100),大致呈收敛趋势,其他模型中收敛趋势较为明显
+ metric1的效果出现条带状

+ 需要手动调参:体素化的一个(?)参数,k,eps,metric
+ 代码结构
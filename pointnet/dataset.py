from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import multiprocessing

# 解决Windows多进程启动问题
if os.name == 'nt':
    multiprocessing.set_start_method('spawn', force=True)

def get_segmentation_classes(root):
    """生成分割类别统计文件（仅当需要分割任务时使用）"""
    pass  # 这里留空，因为当前代码主要处理分类任务

def gen_modelnet_id(root):
    """生成ModelNet数据集的类别ID映射"""
    train_file = os.path.join(root, 'train.txt')
    if not os.path.exists(train_file):
        print(f"错误：ModelNet训练文件不存在！路径：{train_file}")
        sys.exit(1)
    
    classes = set()
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_name = line.split('/')[0]
            classes.add(cls_name)
    classes = sorted(list(classes))
    
    misc_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc')
    os.makedirs(misc_dir, exist_ok=True)
    save_path = os.path.join(misc_dir, 'modelnet_id.txt')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for idx, cls_name in enumerate(classes):
            f.write(f"{cls_name}\t{idx}\n")
    
    print(f"ModelNet类别映射已保存至：{save_path}")

class ShapeNetDataset(data.Dataset):
    """ShapeNet数据集加载类（修复数据类型不匹配问题）"""
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=True,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        super(ShapeNetDataset, self).__init__()
        self.npoints = npoints
        self.root = root
        self.classification = classification
        self.data_augmentation = data_augmentation
        self.split = split
        
        # 加载类别映射
        self.catfile = os.path.join(os.path.dirname(__file__), "synsetoffset2category.txt")
        print(f"读取类别映射文件：{self.catfile}")
        if not os.path.exists(self.catfile):
            print("错误：类别映射文件不存在！")
            sys.exit(1)
        
        self.cat = {}
        with open(self.catfile, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_name, cls_id = line.split()
                self.cat[cls_name] = cls_id
        
        # 筛选指定类别
        if class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
            if not self.cat:
                print(f"错误：指定类别 {class_choice} 不存在")
                sys.exit(1)
        
        self.id2cat = {v: k for k, v in self.cat.items()}
        self.classes = list(self.cat.keys())  # 类别列表属性
        
        # 加载训练/测试划分文件
        splitfile = os.path.join(
            os.path.dirname(__file__), 
            "train_test_split", 
            f"shuffled_{split}_file_list.json"
        )
        print(f"读取{split}集划分文件：{splitfile}")
        if not os.path.exists(splitfile):
            print(f"错误：{split}集划分文件不存在！")
            sys.exit(1)
        
        try:
            with open(splitfile, 'r', encoding='utf-8') as f:
                self.filelist = json.load(f)
            print(f"加载{split}集划分：共{len(self.filelist)}条记录")
        except Exception as e:
            print(f"读取划分文件失败：{str(e)}")
            sys.exit(1)
        
        # 过滤缺失样本
        self.datapath = self._filter_missing_samples()
        self.seg_classes = 0  # 无分割标签
        
        # 打印数据集信息
        print(f"\n{split}集信息：")
        print(f"   - 有效样本数：{len(self.datapath)}")
        print(f"   - 类别数：{len(self.cat)}")
        print(f"   - 类别列表：{self.classes}")
        print(f"   - 任务类型：分类任务（无分割标签）")

    def _filter_missing_samples(self):
        """过滤缺失的样本（只检查点云文件）"""
        valid_samples = []
        missing_count = 0
        prefix_removed_count = 0
        
        print(f"\n检查{self.split}集样本有效性...")
        for idx, record in enumerate(self.filelist):
            try:
                # 处理shape_data前缀
                original_record = record
                if record.startswith('shape_data/'):
                    record = record[len('shape_data/'):]
                    prefix_removed_count += 1
                
                cls_id, sample_id = record.split('/')
            except ValueError:
                print(f"跳过格式错误记录（{idx+1}）：{original_record}")
                continue
            
            if cls_id not in self.id2cat:
                continue
            
            # 拼接点云文件路径
            cls_name = self.id2cat[cls_id]
            pts_path = os.path.join(self.root, cls_id, f"{sample_id}.txt")
            
            if os.path.exists(pts_path):
                valid_samples.append((cls_name, pts_path, None))
            else:
                missing_count += 1
                print(f"缺失点云文件：{pts_path}")
        
        if prefix_removed_count > 0:
            print(f"已自动去除 {prefix_removed_count} 条记录中的 'shape_data/' 前缀")
        
        if not valid_samples:
            print(f"错误：{self.split}集无有效样本！")
            sys.exit(1)
        
        print(f"样本检查完成：有效{len(valid_samples)}/原始{len(self.filelist)}，缺失{missing_count}")
        return valid_samples

    def __getitem__(self, index):
        """获取单个样本（确保数据类型为float32）"""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                cls_name, pts_path, _ = self.datapath[index]
                cls_index = self.classes.index(cls_name)
                
                # 读取点云数据并强制转换为float32
                point_set = np.loadtxt(pts_path, dtype=np.float32)
                
                # 从7维数据中提取前3维坐标
                if point_set.ndim == 2 and point_set.shape[1] >= 3:
                    point_set = point_set[:, :3]  # 提取前3维
                else:
                    raise ValueError(f"点云数据格式异常，无法提取3维坐标：{pts_path}")
                
                # 检查数据有效性
                if len(point_set) < 1:
                    raise ValueError(f"点云文件为空：{pts_path}")
                if point_set.shape[1] != 3:
                    raise ValueError(f"提取后仍不是3维数据：{pts_path}")
                
                # 点云重采样
                if len(point_set) >= self.npoints:
                    choice = np.random.choice(len(point_set), self.npoints, replace=False)
                else:
                    choice = np.random.choice(len(point_set), self.npoints, replace=True)
                point_set = point_set[choice, :]
                
                # 归一化
                point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), axis=0)
                dist_max = np.max(np.sqrt(np.sum(point_set **2, axis=1)))
                if dist_max > 1e-6:
                    point_set = point_set / dist_max
                
                # 数据增强
                if self.data_augmentation:
                    theta = np.random.uniform(0, 2 * np.pi)
                    rotation_matrix = np.array([
                        [np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]
                    ])
                    point_set = point_set.dot(rotation_matrix)
                    point_set += np.random.normal(0, 0.02, size=point_set.shape)
                
                # 转换为Tensor并确保为float32类型
                point_set = torch.from_numpy(point_set).float()  # 显式转换为float32
                cls_label = torch.from_numpy(np.array([cls_index]).astype(np.int64))

                return point_set, cls_label
                    
            except Exception as e:
                print(f" 处理样本出错：{str(e)}，尝试下一个样本...")
                index = (index + 1) % len(self.datapath)
                if attempt == max_attempts - 1:
                    print(f"连续{max_attempts}个样本处理失败，无法继续")
                    sys.exit(1)

    def __len__(self):
        return len(self.datapath)

class ModelNetDataset(data.Dataset):
    """ModelNet数据集加载类"""
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        
        split_file = os.path.join(root, f'{split}.txt')
        if not os.path.exists(split_file):
            print(f"错误：ModelNet {split} 文件不存在！路径：{split_file}")
            sys.exit(1)
            
        with open(split_file, 'r', encoding='utf-8') as f:
            self.fns = [line.strip() for line in f if line.strip()]

        # 读取类别映射
        self.cat = {}
        modelnet_id_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt')
        if not os.path.exists(modelnet_id_path):
            print("未找到ModelNet类别文件，自动生成...")
            gen_modelnet_id(root)
            
        with open(modelnet_id_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_name, cls_id = line.split()
                self.cat[cls_name] = int(cls_id)

        self.classes = list(self.cat.keys())
        print(f"ModelNet {split} 集加载完成：{len(self.fns)}个样本，{len(self.cat)}个类别")

    def __getitem__(self, index):
        fn = self.fns[index]
        cls_name = fn.split('/')[0]
        cls = self.cat[cls_name]
        
        # 读取PLY文件
        ply_path = os.path.join(self.root, fn)
        if not os.path.exists(ply_path):
            print(f"错误：PLY文件不存在！路径：{ply_path}")
            sys.exit(1)
            
        try:
            with open(ply_path, 'rb') as f:
                plydata = PlyData.read(f)
        except Exception as e:
            print(f"读取PLY文件失败：{str(e)}")
            sys.exit(1)
            
        # 提取点云数据
        pts = np.vstack([
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        ]).T.astype(np.float32)  # 确保为float32
        
        # 点云重采样
        if len(pts) < self.npoints:
            choice = np.random.choice(len(pts), self.npoints, replace=True)
        else:
            choice = np.random.choice(len(pts), self.npoints, replace=False)
        point_set = pts[choice, :]

        # 归一化
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist_max = np.max(np.sqrt(np.sum(point_set** 2, axis=1)))
        if dist_max > 1e-6:
            point_set = point_set / dist_max

        # 数据增强
        if self.data_augmentation:
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
            point_set = point_set.dot(rotation_matrix)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        # 转换为Tensor并确保为float32
        point_set = torch.from_numpy(point_set).float()
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        return len(self.fns)

if __name__ == '__main__':
    # 测试数据集加载
    if len(sys.argv) < 3:
        print("用法：python dataset.py [shapenet/modelnet] [数据集路径]")
        print("示例：python dataset.py shapenet C:\\path\\to\\shapenet_dataset")
        sys.exit(1)
    
    dataset_type = sys.argv[1]
    datapath = sys.argv[2]

    if dataset_type == 'shapenet':
        print("测试ShapeNet数据集加载...")
        d = ShapeNetDataset(root=datapath, classification=True, split='train')
        print(f"训练集样本数：{len(d)}")
        ps, cls = d[0]
        print(f"点云形状：{ps.shape}, 数据类型：{ps.dtype}, 类别标签：{cls}")
    elif dataset_type == 'modelnet':
        print("测试ModelNet数据集加载...")
        d = ModelNetDataset(root=datapath, split='train')
        print(f"训练集样本数：{len(d)}")
        ps, cls = d[0]
        print(f"点云形状：{ps.shape}, 数据类型：{ps.dtype}, 类别标签：{cls}")
    else:
        print("错误：不支持的数据集类型！")
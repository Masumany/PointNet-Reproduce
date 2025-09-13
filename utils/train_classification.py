from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing  # 新增：导入多进程模块

# 解决Windows下多进程问题
if os.name == 'nt':
    multiprocessing.set_start_method('spawn', force=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=0)  # 默认为0避免Windows问题
    parser.add_argument(
        '--nepoch', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    opt = parser.parse_args()
    print(opt)

    # 设置控制台颜色输出函数
    blue = lambda x: '\033[94m' + x + '\033[0m'

    # 固定随机种子
    opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # 加载数据集
    if opt.dataset_type == 'shapenet':
        dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            npoints=opt.num_points)

        test_dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    elif opt.dataset_type == 'modelnet40':
        dataset = ModelNetDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='trainval')

        test_dataset = ModelNetDataset(
            root=opt.dataset,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    else:
        exit('错误：不支持的数据集类型！')

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers),
        pin_memory=True)  # 新增：启用内存锁定加速GPU传输

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers),
        pin_memory=True)

    print(f"训练集样本数: {len(dataset)}, 测试集样本数: {len(test_dataset)}")
    num_classes = len(dataset.classes)
    print('类别数:', num_classes)

    # 创建输出文件夹
    try:
        os.makedirs(opt.outf, exist_ok=True)  # 修改：exist_ok=True避免文件夹已存在时出错
    except OSError as e:
        print(f"创建文件夹失败: {e}")
        return

    # 初始化模型
    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

    # 加载预训练模型
    if opt.model != '':
        try:
            classifier.load_state_dict(torch.load(opt.model))
            print(f"成功加载预训练模型: {opt.model}")
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            return

    # 移至GPU
    if torch.cuda.is_available():
        classifier.cuda()
        print("使用GPU进行训练")
    else:
        print("警告：未检测到GPU，将使用CPU训练")

    # 优化器和学习率调度器
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    num_batch = len(dataset) // opt.batchSize  # 修改：使用整数除法避免浮点数

    # 训练循环
    for epoch in range(opt.nepoch):
        # 训练阶段
        classifier.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        # 进度条显示
        train_pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{opt.nepoch}")
        
        for i, data in train_pbar:
            points, target = data
            target = target[:, 0]  # 调整标签形状
            points = points.transpose(2, 1)  # [B, N, 3] -> [B, 3, N]

            # 移至GPU
            if torch.cuda.is_available():
                points, target = points.cuda(), target.cuda()

            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            pred, trans, trans_feat = classifier(points)
            
            # 计算损失
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001  # 特征变换正则化
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            batch_accuracy = correct.item() / float(points.size()[0])
            
            # 累计指标
            total_train_loss += loss.item()
            total_train_correct += correct.item()
            total_train_samples += points.size()[0]
            
            # 更新进度条
            train_pbar.set_postfix({
                '训练损失': f'{loss.item():.4f}',
                '训练准确率': f'{batch_accuracy:.4f}'
            })

        # 每个epoch结束后更新学习率（修复警告的关键）
        scheduler.step()
        
        # 打印训练集epoch总结
        avg_train_loss = total_train_loss / len(dataloader)
        avg_train_acc = total_train_correct / total_train_samples
        print(f"\nEpoch {epoch} 训练总结: 平均损失 = {avg_train_loss:.4f}, 平均准确率 = {avg_train_acc:.4f}")

        # 测试阶段
        classifier.eval()
        total_test_loss = 0.0
        total_test_correct = 0
        total_test_samples = 0

        with torch.no_grad():  # 禁用梯度计算加速测试
            test_pbar = tqdm(enumerate(testdataloader), total=len(testdataloader), desc="测试中")
            for i, data in test_pbar:
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)

                if torch.cuda.is_available():
                    points, target = points.cuda(), target.cuda()

                pred, _, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                batch_accuracy = correct.item() / float(points.size()[0])
                
                total_test_loss += loss.item()
                total_test_correct += correct.item()
                total_test_samples += points.size()[0]
                
                test_pbar.set_postfix({
                    '测试损失': f'{loss.item():.4f}',
                    '测试准确率': f'{batch_accuracy:.4f}'
                })

        # 打印测试集epoch总结
        avg_test_loss = total_test_loss / len(testdataloader)
        avg_test_acc = total_test_correct / total_test_samples
        print(f"Epoch {epoch} 测试总结: 平均损失 = {avg_test_loss:.4f}, 平均准确率 = {avg_test_acc:.4f}")

        # 保存模型
        torch.save(classifier.state_dict(), f'{opt.outf}/cls_model_{epoch}.pth')
        print(f"模型已保存至: {opt.outf}/cls_model_{epoch}.pth")

    # 最终测试
    print("\n开始最终测试...")
    total_correct = 0
    total_testset = 0
    classifier.eval()
    with torch.no_grad():
        for data in tqdm(testdataloader, desc="最终测试"):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            
            if torch.cuda.is_available():
                points, target = points.cuda(), target.cuda()
                
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]

    final_accuracy = total_correct / float(total_testset)
    print(f"最终测试准确率: {final_accuracy:.4f}")
    # 保存最终准确率到文件
    with open(f"{opt.outf}/final_accuracy.txt", "w") as f:
        f.write(f"Final Accuracy: {final_accuracy:.4f}")

# 关键：使用if __name__ == '__main__'包裹主函数调用，解决Windows多进程问题
if __name__ == '__main__':
    # 解决KMP冲突问题
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
    
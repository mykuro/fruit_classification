import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹再重新创建
        rmtree(file_path)
    os.makedirs(file_path)


# 将给定数据集划分成训练集和测试集
def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.1

    # getcwd()：获得当前运行脚本的路径
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "fruit_data")
    origin_fruit_path = os.path.join(data_root, "fruit_data")
    assert os.path.exists(
        origin_fruit_path), "file: '{}' does not exist.".format(
            origin_fruit_path)
    # isdir()：判断某一路径是否为目录
    # listdir()：返回指定文件夹包含的文件或文件夹名字列表
    fruit_class = [
        cla for cla in os.listdir(origin_fruit_path)
        if os.path.isdir(os.path.join(origin_fruit_path, cla))
    ]

    # 创建训练集train文件夹，并由类名在其目录下创建子目录
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in fruit_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 创建验证集val文件夹，并由类名在其目录下创建子目录
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in fruit_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    # 遍历所有类别的图像并按比例分成训练集和验证集
    for cla in fruit_class:
        cla_path = os.path.join(origin_fruit_path, cla)
        # images列表存储了该目录下所有图像的名称
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        # 从images列表中随机抽取k个图像名称
        # random.sample：用于截取列表的指定长度的随机数，返回列表
        # eval_index保存验证集val的图像名称
        eval_index = random.sample(images, k=int(num * split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配到验证集的文件复制到对应的目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配到训练集的文件复制到对应的目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num),
                  end="")
        print()

    print("processing done!")


if __name__ == '__main__':
    main()

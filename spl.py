# 将图片和标注数据按比例切分为 训练集和测试集
import shutil
import random
import os

# 原始路径
image_original_path = "data/images/"
label_original_path = "data/labels/"

cur_path = os.getcwd()
# 训练集路径
train_image_path = os.path.join(cur_path, "datasets/images/train/")
train_label_path = os.path.join(cur_path, "datasets/labels/train/")

# 验证集路径
val_image_path = os.path.join(cur_path, "datasets/images/val/")
val_label_path = os.path.join(cur_path, "datasets/labels/val/")

# 测试集路径
test_image_path = os.path.join(cur_path, "datasets/images/test/")
test_label_path = os.path.join(cur_path, "datasets/labels/test/")

# 训练集目录
list_train = os.path.join(cur_path, "datasets/train.txt")
list_val = os.path.join(cur_path, "datasets/val.txt")
list_test = os.path.join(cur_path, "datasets/test.txt")

# 划分比例
train_percent = 0.8
val_percent = 0.1
test_percent = 0.1

# 支持的图片格式（可根据实际情况添加，如jpg/jpeg/bmp等）
SUPPORTED_IMG_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')


def del_file(path):
    """删除目录下所有文件"""
    if not os.path.exists(path):
        return
    for i in os.listdir(path):
        file_data = os.path.join(path, i)
        if os.path.isfile(file_data):
            os.remove(file_data)


def mkdir():
    """创建目录（若存在则清空）"""
    dirs = [train_image_path, train_label_path, val_image_path, val_label_path,
            test_image_path, test_label_path]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            del_file(dir_path)


def clearfile():
    """删除原有txt列表文件"""
    files = [list_train, list_val, list_test]
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def get_valid_samples():
    """
    获取有效样本：以图片为基准，匹配对应的标注文件
    返回：有效样本名称列表（不含后缀）
    """
    valid_samples = []
    # 遍历图片目录，只处理支持的图片格式
    for img_file in os.listdir(image_original_path):
        if img_file.lower().endswith(SUPPORTED_IMG_FORMATS):
            # 提取样本名称（不含后缀）
            sample_name = os.path.splitext(img_file)[0]
            # 检查对应的标注文件是否存在
            label_file = os.path.join(label_original_path, f"{sample_name}.txt")
            if os.path.exists(label_file):
                valid_samples.append(sample_name)
            else:
                print(f"警告：样本 {sample_name} 缺少标注文件，跳过")
    return valid_samples


def main():
    mkdir()
    clearfile()

    # 获取有效样本（图片+标注都存在）
    valid_samples = get_valid_samples()
    if not valid_samples:
        print("错误：未找到有效样本（图片和标注需同时存在）")
        return
    total_num = len(valid_samples)
    print(f"有效样本总数：{total_num}")

    # 随机打乱样本（保证划分随机性）
    random.shuffle(valid_samples)

    # 计算各集数量
    train_num = int(total_num * train_percent)
    val_num = int(total_num * val_percent)
    test_num = total_num - train_num - val_num

    # 划分样本
    train_samples = valid_samples[:train_num]
    val_samples = valid_samples[train_num:train_num + val_num]
    test_samples = valid_samples[train_num + val_num:]

    print(f"训练集数目：{len(train_samples)}, 验证集数目：{len(val_samples)}, 测试集数目：{len(test_samples)}")

    # 打开txt文件准备写入
    with open(list_train, 'w') as f_train, open(list_val, 'w') as f_val, open(list_test, 'w') as f_test:
        # 处理训练集
        for sample in train_samples:
            # 拼接源文件路径
            src_img = None
            # 查找图片文件（处理不同格式）
            for fmt in SUPPORTED_IMG_FORMATS:
                img_path = os.path.join(image_original_path, f"{sample}{fmt}")
                if os.path.exists(img_path):
                    src_img = img_path
                    break
            src_label = os.path.join(label_original_path, f"{sample}.txt")

            # 拼接目标文件路径（保留原图片格式）
            img_ext = os.path.splitext(src_img)[1]
            dst_train_img = os.path.join(train_image_path, f"{sample}{img_ext}")
            dst_train_label = os.path.join(train_label_path, f"{sample}.txt")

            # 复制文件
            shutil.copyfile(src_img, dst_train_img)
            shutil.copyfile(src_label, dst_train_label)

            # 写入txt列表
            f_train.write(dst_train_img + '\n')

        # 处理验证集
        for sample in val_samples:
            src_img = None
            for fmt in SUPPORTED_IMG_FORMATS:
                img_path = os.path.join(image_original_path, f"{sample}{fmt}")
                if os.path.exists(img_path):
                    src_img = img_path
                    break
            src_label = os.path.join(label_original_path, f"{sample}.txt")

            img_ext = os.path.splitext(src_img)[1]
            dst_val_img = os.path.join(val_image_path, f"{sample}{img_ext}")
            dst_val_label = os.path.join(val_label_path, f"{sample}.txt")

            shutil.copyfile(src_img, dst_val_img)
            shutil.copyfile(src_label, dst_val_label)
            f_val.write(dst_val_img + '\n')

        # 处理测试集
        for sample in test_samples:
            src_img = None
            for fmt in SUPPORTED_IMG_FORMATS:
                img_path = os.path.join(image_original_path, f"{sample}{fmt}")
                if os.path.exists(img_path):
                    src_img = img_path
                    break
            src_label = os.path.join(label_original_path, f"{sample}.txt")

            img_ext = os.path.splitext(src_img)[1]
            dst_test_img = os.path.join(test_image_path, f"{sample}{img_ext}")
            dst_test_label = os.path.join(test_label_path, f"{sample}.txt")

            shutil.copyfile(src_img, dst_test_img)
            shutil.copyfile(src_label, dst_test_label)
            f_test.write(dst_test_img + '\n')

    print("数据集划分完成！")


if __name__ == "__main__":
    # 固定随机种子（可选，便于复现结果）
    random.seed(42)
    main()

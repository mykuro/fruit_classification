import os
import json
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from torchvision import transforms
from model import GoogLeNet


def predict(img_path, root):
    # 如果有NVIDA显卡，转到GPU训练，否则用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 将多个transforms的操作整合在一起
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # root = tk.Toplevel()
    root.title("水果卡路里识别")
    # root.geometry('400x400')
    # 确定图片存在，否则反馈错误
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(
        img_path)
    img = Image.open(img_path)
    photo = ImageTk.PhotoImage(img.resize((400, 400)))
    tk.Label(root, image=photo).pack()

    # [C, H, W]，转换图像格式
    img = data_transform(img)
    # [N, C, H, W]，增加一个维度N
    img = torch.unsqueeze(img, dim=0)

    # 获取结果类型
    json_path = './class_indices.json'
    # 确定路径存在，否则反馈错误
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(
        json_path)
    # 读取内容
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 获取对应水果营养素信息
    nutrition_path = './fruit_nutrition.json'
    # 确定路径存在，否则反馈错误
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(
        nutrition_path)
    # 读取内容
    with open(nutrition_path, "r") as f:
        class_nutrition = json.load(f)

    # 模型实例化，将模型转到device，结果类型有5种
    # 实例化模型时不需要辅助分类器
    model = GoogLeNet(num_classes=5, aux_logits=False).to(device)
    # 载入模型权重
    weights_path = "./googleNet.pth"
    # 确定模型存在，否则反馈错误
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(
        weights_path)
    # 在加载训练好的模型参数时，由于其中是包含有辅助分类器的，需要设置strict=False舍弃不需要的参数
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(
        weights_path, map_location=device),
                                                          strict=False)

    # 进入验证阶段
    model.eval()
    with torch.no_grad():
        # 预测类别
        # squeeze()：维度压缩，返回一个tensor(张量)，其中input中大小为1的所有维都已删除
        output = torch.squeeze(model(img.to(device))).cpu()
        # softmax：归一化指数参数，将预测结果输入进行非负性和归一化处理，最后将某一维度值处理为0-1之间的分类概率
        predict = torch.softmax(output, dim=0)
        # argmax(input)：返回制定维度最大值的序号
        # .numpy()：把tensor转换成numpy的格式
        predict_cla = torch.argmax(predict).numpy()

    # 输出的预测值和真实值
    print_res = "class: {}  prob: {:.3}".format(class_indict[str(predict_cla)],
                                                predict[predict_cla].numpy())
    text = """热量(kcal/100g):{},
    碳水化合物(g/100g):{},
    脂肪(g/100g):{},
    蛋白质(g/100g):{},
    纤维素(g/100g):{}""".format(
        class_nutrition[class_indict[str(predict_cla)]][0],
        class_nutrition[class_indict[str(predict_cla)]][1],
        class_nutrition[class_indict[str(predict_cla)]][2],
        class_nutrition[class_indict[str(predict_cla)]][3],
        class_nutrition[class_indict[str(predict_cla)]][4])

    tk.Label(root, text=print_res).pack()
    tk.Label(root, text=text).pack()

    root.mainloop()


# 用自己的数据集进行分类测试
def main():
    # 加载图片
    root = tk.Tk()
    # root.withdraw()
    img_path = filedialog.askopenfilename()

    predict(img_path, root)


if __name__ == '__main__':
    main()

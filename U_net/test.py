import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import MyDataset
from unet import UNet


def calculate_metrics(pred, target):
    """计算二分类的 Pixel Accuracy 和 IoU"""
    # 输入的 pred, target 要是 numpy.ndarray
    correct = (pred == target).sum()
    # 修改这里：numpy 没有 .numel()，使用 .size 或直接用长宽乘积
    total = target.size
    pixel_acc = correct / total

    # numpy 的位运算符号一样，但逻辑上建议确保是布尔或整数
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)

    return pixel_acc, iou


def test():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 1. 初始化 TensorBoard 和路径
    writer = SummaryWriter('./logs')
    test_data_path = './crack_segmentation_dataset/test'
    weight_path = './model_weights/unet_epoch_2.pth'  # 请确保权重文件存在

    # 2. 加载数据与模型
    dataset = MyDataset(test_data_path)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    net = UNet(num_classes=2).to(device)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.eval()

    total_PA = 0
    total_iou = 0
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image, mask = image.to(device), mask.to(device)

            # 处理真实标签为 0/1
            gt_mask = torch.where(mask > 0, 1, 0).long().squeeze(0)  # [256, 256]

            # 模型预测
            out = net(image)
            pred_mask = torch.argmax(out, dim=1).squeeze(0)  # [256, 256]

            # 计算指标
            acc, iou = calculate_metrics(pred_mask.cpu().numpy().astype(int),
                                         gt_mask.cpu().numpy().astype(int))
            total_PA += acc
            total_iou += iou

            # --- 可视化拼接逻辑 ---
            # 1. 原始彩色图 (反标准化)
            img_show = image.squeeze(0).cpu()  # [3, 256, 256]

            # 2. 真实掩码图转为 RGB 格式方便显示
            gt_show = torch.stack([gt_mask.cpu().float() * 255] * 3, dim=0) / 255.0

            # 3. 预测掩码图转为 RGB 格式
            pred_show = torch.stack([pred_mask.cpu().float() * 255] * 3, dim=0) / 255.0

            # 横向拼接: [彩色图, 真实图, 预测图]
            combined_img = torch.cat([img_show, gt_show, pred_show], dim=2)  # 在宽度(W)维度拼接

            # 将指标写入 TensorBoard
            # 这里的 tag 包含了每张图片的指标信息，方便在 TensorBoard 中直接看到
            tag = f"Result_{i}_Acc_{acc:.3f}_IoU_{iou:.3f}"
            writer.add_image(tag, combined_img)

            # 记录全局平均指标
            writer.add_scalar('Metrics/Pixel_Accuracy', acc, i)
            writer.add_scalar('Metrics/IoU', iou, i)

            if i % 10 == 0:
                print(f"Processed {i} images, Current IoU: {iou:.4f}")

    writer.close()
    print("测试完成！请运行: tensorboard --logdir=./logs")
    avg_PA = total_PA / len(test_loader)
    avg_iou = total_iou / len(test_loader)
    print(f"该权重下，avg_PA={avg_PA:.3f}，avg_iou={avg_iou:.3f}")


if __name__ == '__main__':
    test()

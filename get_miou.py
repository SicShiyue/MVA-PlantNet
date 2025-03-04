import os

from PIL import Image
from tqdm import tqdm

from gdeeplab import PlantNet 
from utils.utils_metrics import compute_mIoU, show_results

if __name__ == "__main__":

    miou_mode       = 0

    num_classes     = 4

    #name_classes = ["background","leaves","peppers","peduncles","stems","shoots and leaf stems","wires","cuts"]
    name_classes    = ["bg","branch","leaf","pepper"]

    data_path  = ''

    image_ids       = open(os.path.join(data_path, "txt/val.txt"), 'r').read().splitlines()
    gt_dir          = os.path.join(data_path, "lab/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        deeplab = DeeplabV3()
        #插值1
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(data_path, "data"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision,FWIOUs = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes,FWIOUs)
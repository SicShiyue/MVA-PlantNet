with open(r"E:\plant seg\expin1_200\VOCPaperFT\VOC2007\ImageSets\Segmentation\val.txt","w") as f:
    for i in range(30,50):
        i = "{}".format(i+1)
        str =  "empirical_label_class_all_grayscale_"+i
        f.write(str)
        f.write("\n")
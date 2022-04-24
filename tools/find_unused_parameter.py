import torch
sd1 = torch.load("/home/wengkangming/map_file/mmocr/result/ctw1500/psenet_r50_bifpnf_resasapp_coordconv_600e_ctw1500+totaltext/epoch_10.pth")["state_dict"]
sd4 = torch.load("/home/wengkangming/map_file/mmocr/result/ctw1500/psenet_r50_bifpnf_resasapp_coordconv_600e_ctw1500+totaltext/epoch_20.pth")["state_dict"]
for k in sd1:
    v1 = sd1[k]
    v4 = sd4[k]
    if (v1 == v4).all():
        print(k)
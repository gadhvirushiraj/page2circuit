import torch
import cv2
from torchvision import transforms

def class_component(img):
    # thin the lines in image
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    model = torch.load('model.pkl')
    model.eval()
    # transform the image
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64,64)),
        transforms.Lambda(lambda x: x[0,:,:].T),
    ])
    # convert the img
    img = trans(img)
    # classes in the model
    label_list = ['ac_src_r0','ac_src_r1','battery_r0','battery_r1','battery_r2',
                  'battery_r3','cap_r0','cap_r1','curr_src_r0','curr_src_r1',
                  'curr_src_r2','curr_src_r3','dc_volt_src_1_r0','dc_volt_src_1_r1','dc_volt_src_1_r2',
                  'dc_volt_src_1_r3','dc_volt_src_2_r0','dc_volt_src_2_r1','dc_volt_src_2_r2','dc_volt_src_2_r3',
                  'dep_curr_src_r0','dep_curr_src_r1','dep_curr_src_r2','dep_curr_src_r3','dep_volt_r0',
                  'dep_volt_r1','dep_volt_r2','dep_volt_r3','diode_r0','diode_r1',
                  'diode_r2','diode_r3','gnd_1','inductor_r0','inductor_r1','resistor_r0','resistor_r1']
    # predict the class
    pred = model(img.unsqueeze(0))
    pred = torch.argmax(pred, dim=1)
    
    return label_list[pred]

def driver_classify(comp_imgs, original_img, rects):
    classified = []

    for img in comp_imgs:
        classified.append(class_component(img))

    for i,rect in enumerate(rects):
        x = rect[0]
        y = rect[3]
        w = (rect[1] - rect[0])
        h = (rect[2] - rect[3])

        original_img = cv2.rectangle(original_img, (x, y), (x + w, y + h), (36,255,12), 5)
        cv2.putText(original_img, classified[i], (x, y + h -10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 3)

    return classified, original_img
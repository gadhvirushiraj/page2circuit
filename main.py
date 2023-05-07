import preprocess
import classify
import numpy as np
from PIL import Image

def main():
    pass

if __name__ == '__main__':
    main()

def preprocess_img(input_img):
    input_img = np.array(Image.open(input_img))
    result_img, transitional_imgs, components, rects = preprocess.driver_preprocess(input_img)
    return result_img, transitional_imgs, components, rects

def classify_components(list_comp, img, rects):
    img = np.array(Image.open(img))
    classified, annoted_img = classify.driver_classify(list_comp, img, rects)
    return classified, annoted_img

def make_schematics():
    pass
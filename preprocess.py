import cv2
import numpy as np
from sklearn.cluster import DBSCAN

transition_list = []

def initial_filtering(img):
    # converting to grey scale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # applying adaptive threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 21)
    transition_list.append(img)
    # opening i.e. eroding the img and then dialating it for removing noise & dialating again
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k3, iterations=2)
    k3_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    transition_list.append(img)
    img = cv2.dilate(img, k3_2, iterations=2)
    

    return img

def cluster_anchor_points(img, original_img):
    anchor_points = cv2.goodFeaturesToTrack(image=img,maxCorners=100,qualityLevel=0.40,minDistance=20, blockSize=15)
    anchor_points = np.float32(anchor_points.reshape((-1, 2)))
    clustering = DBSCAN(eps=80, min_samples=4).fit(anchor_points)
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)

    # add intial clustering to transition list
    cluster_plot = original_img.copy()
    for l in unique_labels:
        if l == -1: continue
        px = [int(i[0]) for i in anchor_points[labels == l]]
        py = [int(i[1]) for i in anchor_points[labels == l]]
        color = np.random.choice(range(256), size=3)
        color = (int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
        for i in range(len(px)):
            cv2.circle(cluster_plot, (px[i], py[i]), 5, color=tuple(color), thickness=10)

    transition_list.append(cluster_plot)

    # calculating mean cords of each cluster
    cluster_centers = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = anchor_points[labels == label]
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)

    # perform fast algorithm to detect corners
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=False, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    kp = fast.detect(img, None)

    # add fast algorithm to transition list
    transition_list.append(cv2.drawKeypoints(original_img.copy(), kp, None, color=(255,0,0)))

    # combining clustering outputs
    fast_features = [np.array([int(kp[i].pt[0]), int(kp[i].pt[1])]) for i in range(len(kp))]
    final_keypoints = anchor_points.copy()
    final_labels = labels.copy()

    for i in fast_features:
        for j in range(len(cluster_centers)):
            if np.linalg.norm(i-cluster_centers[j]) < 60:
                final_keypoints = np.vstack((final_keypoints, [i]))
                final_labels = np.append(final_labels, unique_labels[j+1])
                break
    
    # add combined clustering to transition list
    final_cluster = original_img.copy()
    for l in unique_labels:
        if l == -1: continue
        px = [int(i[0]) for i in final_keypoints[final_labels == l]]
        py = [int(i[1]) for i in final_keypoints[final_labels == l]]
        color = np.random.choice(range(256), size=3)
        color = (int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
        for i in range(len(px)):
            cv2.circle(final_cluster, (px[i], py[i]), 5, color=tuple(color), thickness=10)

    transition_list.append(final_cluster)
    
    return final_keypoints, final_labels, unique_labels, counts


def remove_components(img, anchor_points, labels, unique_labels, counts):
    components_ext = []
    rects = []
    rects_contour = []

    ext_img = img.copy()
    for l,_ in enumerate(counts):
        if l == 0: continue
        px = [int(i[0]) for i in anchor_points[labels == unique_labels[l]]]
        py = [int(i[1]) for i in anchor_points[labels == unique_labels[l]]]
        if abs(max(py)-min(py))*abs(max(px)-min(px)) > 700:
            components_ext.append(ext_img[min(py)-15:max(py)+15, min(px)-15:max(px)+15])
            img[min(py)-15:max(py)+15, min(px)-15:max(px)+15] = 0      
            rects_contour.append((np.array([[min(px)-15, min(py)-15],[max(px)+15, min(py)-15],[max(px)+15, max(py)+15],[min(px)-15, max(py)+15]])))
            rects.append([min(px)-22,max(px)+22,min(py)-22,max(py)+22])
    
    transition_list.append(img)

    return img, rects, rects_contour, components_ext

def wire_mapping(img, rects, rects_contour, original_img):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_cnts = [c for c in cnts[0] if cv2.contourArea(c) > 600]

    wiring_dict = {}
    last_key = 0
    for c in filtered_cnts:
        for rect in rects:
            for p in c:
                # rect[0] is xmin, rect[1] is xmax, rect[2] is ymin, rect[3] is ymax
                # p[0][0] is x, p[0][1] is y
                if p[0][0] >= rect[0] and p[0][0] <= rect[1] and p[0][1] >= rect[2] and p[0][1] <= rect[3]:
                    if last_key not in wiring_dict.keys():
                        wiring_dict[last_key] = [rect]
                    else:
                        wiring_dict[last_key].append(rect)
                    break
        last_key += 1

    key = 0
    dict_len = len(wiring_dict)
    del_wire = []
    while key < dict_len:
        if len(wiring_dict[key]) == 1:
            del_wire.append(key)
            del wiring_dict[key]
        key += 1
     
    for i in del_wire:
        del filtered_cnts[i]
    
    cv2.drawContours(original_img, tuple(rects_contour), -1, (255,0,0), 3)
    cv2.drawContours(original_img, filtered_cnts, -1, (0,255,0), 10)

    return original_img


def driver_preprocess(img):
    transition_list.clear()
    original_img = img.copy()
    img = initial_filtering(img)
    anchor_points, labels, unique_labels, counts = cluster_anchor_points(img.copy(), original_img)
    img, rects, rects_contour, components = remove_components(img, anchor_points, labels, unique_labels, counts)
    img = wire_mapping(img, rects, rects_contour, original_img)
    return img, transition_list, components, rects
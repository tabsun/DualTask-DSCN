import numpy as np
import os, cv2

def convert2image(image):
    image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
    image = (image * 255.).astype(np.uint8)
    return image

def visualize2(image_A, image_B, mask_A, mask_B, change_mask, names):
    image_A = image_A.numpy()
    image_B = image_B.numpy()
    mask_A = mask_A.numpy()
    mask_B = mask_B.numpy()
    change_mask = change_mask.numpy()
    for i in range(image_A.shape[0]):
        temp = change_mask[i]
        name = names[i].split('/')[-1]
        cv2.imwrite('visualize/A/'+name, convert2image(image_A[i]))
        cv2.imwrite('visualize/B/'+name, convert2image(image_B[i]))
        cv2.imwrite('visualize/label_A/'+name, convert2image(mask_A[i]))
        cv2.imwrite('visualize/label_B/'+name, convert2image(mask_B[i]))
        cv2.imwrite('visualize/change/'+name, convert2image(change_mask[i]))
        if(np.max(temp) == 0):
            print(i, " no change ", names[i])
            image = cv2.imread(os.path.join('/data1/tabsun/SAR/TianZhi/train/', names[i]))
            print("Actually :", np.max(image))
        else:
            print(i, np.max(temp))

def visualize1(image_A, image_B, output):
    b, c, h, w = image_A.shape
    assert(b == 1)
    std_size = 512
    image_A = convert2image(image_A.reshape(c, h, w))
    image_B = convert2image(image_B.reshape(c, h, w))
    image_A = (image_A * 255.).astype(np.uint8)
    image_B = (image_B * 255.).astype(np.uint8)
    image_A = cv2.resize(image_A, (std_size, std_size))
    image_B = cv2.resize(image_B, (std_size, std_size))

    output = output.reshape(2, h, w)
    print("Output:", np.min(output), np.max(output))
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask[:, :, 0][output[0,:,:] > 0] = 255
    mask[:, :, 2][output[1,:,:] > 0] = 255
    mask = cv2.resize(mask, (std_size, std_size))

    image = np.zeros((std_size, 3*std_size, 3), dtype=np.uint8)
    image[:, :std_size, :] = image_A
    image[:, std_size:std_size*2, :] = image_B
    image[:, 2*std_size:, :] = mask
    return image

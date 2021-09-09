import numpy as np
import os, cv2

def convert2image(image):
    image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
    image = (image * 255.).astype(np.uint8)
    return image

def convert2rgb(image):
    h, w, c = image.shape
    assert(c == 1)
    temp = np.zeros((h, w, 3), dtype=np.uint8)
    temp[:,:,0] = image.reshape(h,w)
    temp[:,:,1] = image.reshape(h,w)
    temp[:,:,2] = image.reshape(h,w)
    return temp

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

def visualize1(image_A_batch, image_B_batch, gt_batch, pd_batch):
    image_A_batch = image_A_batch.cpu().numpy()
    image_B_batch = image_B_batch.cpu().numpy()
    gt_batch = gt_batch.cpu().numpy()
    pd_batch = pd_batch.cpu().numpy()

    std_size = 512
    whole_images = []
    for i in range(image_A_batch.shape[0]):
        image_A = image_A_batch[i]
        image_B = image_B_batch[i]
        gt = gt_batch[i]
        pd = pd_batch[i]
        image_A = cv2.resize(convert2image(image_A), (std_size, std_size))
        image_B = cv2.resize(convert2image(image_B), (std_size, std_size))
        gt = cv2.resize(convert2rgb(convert2image(gt)), (std_size, std_size))
        pd = cv2.resize(convert2rgb(convert2image(pd)), (std_size, std_size))

        whole = np.zeros((std_size*2, std_size*2, 3), dtype=np.uint8)
        whole[:std_size, :std_size, :] = image_A
        whole[:std_size, std_size:, :] = image_B
        whole[std_size:, :std_size, :] = gt
        whole[std_size:, std_size:, :] = pd

        whole_images.append(whole)

    return whole_images

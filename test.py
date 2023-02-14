import os
import yaml
import matplotlib.pyplot as plt
from dataset.utils.photometric_augmentation import *
from dataset.coco import *

if __name__=='__main__':
    with open('./config/superpoint_train.yaml','r') as fin:
        config = yaml.safe_load(fin)

    coco = COCODataset(config['data'],True)
    cdataloader = DataLoader(coco,collate_fn=coco.batch_collator,batch_size=1,shuffle=True)

    for i,d in enumerate(cdataloader):
        if i>=10:
            break
        img = (d['raw']['img']*255).cpu().numpy().squeeze().astype(np.int).astype(np.uint8)
        img_warp = (d['warp']['img']*255).cpu().numpy().squeeze().astype(np.int).astype(np.uint8)
        img = cv2.merge([img, img, img])
        img_warp = cv2.merge([img_warp, img_warp, img_warp])
        ##
        kpts = np.where(d['raw']['kpts_map'].squeeze().cpu().numpy())
        kpts = np.vstack(kpts).T
        kpts = np.round(kpts).astype(np.int)
        for kp in kpts:
            cv2.circle(img, (kp[1], kp[0]), radius=3, color=(0,255,0))
        kpts = np.where(d['warp']['kpts_map'].squeeze().cpu().numpy())
        kpts = np.vstack(kpts).T
        kpts = np.round(kpts).astype(np.int)
        for kp in kpts:
            cv2.circle(img_warp, (kp[1], kp[0]), radius=3, color=(0,255,0))

        mask = d['raw']['mask'].cpu().numpy().squeeze().astype(np.int).astype(np.uint8)*255
        warp_mask = d['warp']['mask'].cpu().numpy().squeeze().astype(np.int).astype(np.uint8)*255

        img = cv2.resize(img, (640,480))
        img_warp = cv2.resize(img_warp,(640,480))

        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.subplot(2,2,2)
        plt.imshow(mask)
        plt.subplot(2,2,3)
        plt.imshow(img_warp)
        plt.subplot(2,2,4)
        plt.imshow(warp_mask)
        plt.show()

    print('Done')
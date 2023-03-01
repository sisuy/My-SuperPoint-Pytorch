#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn.functional as F
from utils.keypoint_op import warp_points
from utils.tensor_op import pixel_shuffle_inv



def loss_func(config, data, prob, desc=None, prob_warp=None, desc_warp=None, device='cpu'):

    det_loss = detector_loss(data['raw']['kpts_map'],
                             prob['logits'],
                             data['raw']['mask'],
                             config['grid_size'],
                             device=device)

    if desc is None or prob_warp is None or desc_warp is None:
        return det_loss

    det_loss_warp = detector_loss(data['warp']['kpts_map'],
                                  prob_warp['logits'],
                                  data['warp']['mask'],
                                  config['grid_size'],
                                  device=device)

    weighted_des_loss = descriptor_loss(config,
                               desc['desc_raw'],
                               desc_warp['desc_raw'],
                               data['homography'],
                               data['warp']['mask'],
                               device)

    loss = det_loss + det_loss_warp + weighted_des_loss

    a, b, c = det_loss.item(), det_loss_warp.item(), weighted_des_loss.item()
    print('debug: {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(a, b,c,a+b+c))
    return loss

def detector_loss(keypoint_map, logits, valid_mask=None, grid_size=8, device='cpu'):
    '''
    Parameters:
        keypoint_map: [B,H,W]
        logits: [B,C,H/gird_size,W/grid_size]
        valid_mask: [B,H,W]
        grid_size: int(default=8)
    '''
    labels = keypoint_map.unsqueeze(dim=1) # [1, 1, 240, 320]

    # pixel shuffle inverse 1 channels -> 64 channels
    pixelShuffle_inv = torch.nn.PixelUnshuffle(grid_size)
    labels = pixelShuffle_inv(labels) # [1, 64, 30, 40]
    B,C,H,W = labels.shape

    # add dusbin(torch.ones -> [1,1,30,40])
    labels = torch.cat([2*labels,torch.ones([B,1,H,W],device=device)],dim=1) # [1, 65, 30, 40]
    labels = torch.argmax(labels + torch.zeros(labels.shape,device=device).uniform_(0,0.1),dim=1)#B*)H/grid_size)*(W/grid_size)

    # generate valid_mask: [B,H,W] -> [B,H/8,/W/8]
    if valid_mask is None:
        valid_mask = torch.ones_like(keypoint_map)
    else:
        valid_mask = valid_mask

    valid_mask = valid_mask.unsqueeze(dim=1) # [B,1,H,W] 
    valid_mask =  pixelShuffle_inv(valid_mask) # [B,64,H/8,W/8]
    valid_mask = torch.prod(valid_mask,dim=1).unsqueeze(dim=1) # [B,1,H/8,W/8]

    # apply softmax on predicted logits value
    loss = torch.nn.functional.log_softmax(logits,dim=1)
    mask = valid_mask.type(torch.float32)

    # generate the loss covered by valid mask
    mask = mask.squeeze(dim=1) # [1,30,40]
    mask = mask/(torch.mean(mask)+1e-5)
    loss = mask*loss

    # use cross-entropy to get the loss
    lossFunction = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = lossFunction(logits,labels) # [1,30,40]
    return loss

def descriptor_loss(config, descriptors, warped_descriptors, homographies, valid_mask=None, device='cpu'):
    """
    :param descriptors: [B,C,H/8,W/8]
    :param warped_descriptors: [B,C.H/8,W/8]
    :param homographies: [B,3,3]
    :param config:
    :param valid_mask:[B,H,W]
    :param device:
    :return:
    """
    grid_size = config['grid_size']
    positive_margin = config['loss']['positive_margin']
    negative_margin = config['loss']['negative_margin']
    lambda_d = config['loss']['lambda_d']
    lambda_loss = config['loss']['lambda_loss']

    (batch_size, _, Hc, Wc) = descriptors.shape
    coord_cells = torch.stack(torch.meshgrid([torch.arange(Hc,device=device),
                                              torch.arange(Wc,device=device)]),dim=-1)#->[Hc,Wc,2]
    coord_cells = coord_cells * grid_size + grid_size // 2  # (Hc, Wc, 2)
    # coord_cells is now a grid containing the coordinates of the Hc x Wc
    # center pixels of the 8x8 cells of the image

    # Compute the position of the warped center pixels
    warped_coord_cells = warp_points(coord_cells.reshape(-1, 2), homographies, device=device)
    # warped_coord_cells is now a list of the warped coordinates of all the center
    # pixels of the 8x8 cells of the image, shape (B, Hc x Wc, 2)

    # Compute the pairwise distances and filter the ones less than a threshold
    # The distance is just the pairwise norm of the difference of the two grids
    # Using shape broadcasting, cell_distances has shape (B, Hc, Wc, Hc, Wc)
    coord_cells = torch.reshape(coord_cells, [1,1,1,Hc,Wc,2]).type(torch.float32)
    warped_coord_cells = torch.reshape(warped_coord_cells, [batch_size, Hc, Wc, 1, 1, 2])
    cell_distances = torch.norm(coord_cells - warped_coord_cells, dim=-1, p=2)
    s = (cell_distances<=(grid_size-0.5)).float()#
    # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by the
    # homography is at a distance from (h', w') less than config['grid_size']
    # and 0 otherwise

    # Normalize the descriptors and
    # compute the pairwise dot product between descriptors: d^t * d'
    descriptors = torch.reshape(descriptors, [batch_size, -1, Hc, Wc, 1, 1])
    descriptors = F.normalize(descriptors, p=2, dim=1)
    warped_descriptors = torch.reshape(warped_descriptors, [batch_size, -1, 1, 1, Hc, Wc])
    warped_descriptors = F.normalize(warped_descriptors, p=2, dim=1)
    ##
    dot_product_desc = torch.sum(descriptors * warped_descriptors, dim=1)

    dot_product_desc = F.relu(dot_product_desc)
    # print("dot_product_desc: {}".format(dot_product_desc))

    ##l2_normalization
    dot_product_desc = torch.reshape(F.normalize(torch.reshape(dot_product_desc, [batch_size, Hc, Wc, Hc * Wc]),
                                                 p=2,
                                                 dim=3), [batch_size, Hc, Wc, Hc, Wc])
    dot_product_desc = torch.reshape(F.normalize(torch.reshape(dot_product_desc, [batch_size, Hc * Wc, Hc, Wc]),
                                                 p=2,
                                                 dim=1), [batch_size, Hc, Wc, Hc, Wc])

    # dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
    # descriptor at position (h, w) in the original descriptors map and the
    # descriptor at position (h', w') in the warped image

    positive_dist = torch.maximum(torch.tensor(0.,device=device), positive_margin - dot_product_desc)
    negative_dist = torch.maximum(torch.tensor(0.,device=device), dot_product_desc - negative_margin)

    loss = lambda_d * s * positive_dist + (1 - s) * negative_dist

    # Mask the pixels if bordering artifacts appear
    valid_mask = torch.ones([batch_size, Hc*grid_size, Wc*grid_size],
                             dtype=torch.float32, device=device) if valid_mask is None else valid_mask
    valid_mask = valid_mask.unsqueeze(dim=1).type(torch.float32)  # [B, H, W]->[B,1,H,W]
    valid_mask = pixel_shuffle_inv(valid_mask, grid_size)# ->[B,64,Hc,Wc]
    valid_mask = torch.prod(valid_mask, dim=1)
    valid_mask = torch.reshape(valid_mask, [batch_size, 1, 1, Hc, Wc])

    normalization = torch.sum(valid_mask)*(Hc*Wc)

    positive_sum = torch.sum(valid_mask*lambda_d*s*positive_dist) / normalization
    negative_sum = torch.sum(valid_mask*(1-s)*negative_dist) / normalization

    print('positive_dist:{:.7f}, negative_dist:{:.7f}'.format(positive_sum, negative_sum))

    loss = lambda_loss*torch.sum(valid_mask * loss)/normalization

    print("descriptor_loss: {}".format(loss))
    return loss


def precision_recall(pred, keypoint_map, valid_mask):
    pred = valid_mask * pred
    labels = keypoint_map

    precision = torch.sum(pred*labels)/torch.sum(pred)
    recall = torch.sum(pred*labels)/torch.sum(labels)

    return {'precision': precision, 'recall': recall}


if __name__=='__main__':
    '''
    :param keypoint_map: [B,H,W]
    :param logits: [B,65,Hc,Wc]
    :param valid_mask:[B, H, W]
    :param grid_size: 8 default
    '''

    import tensorflow as tf
    def detector_loss_tf(keypoint_map, logits, valid_mask=None):
        # Convert the boolean labels to indices including the "no interest point" dustbin
        labels = tf.to_float(keypoint_map[..., tf.newaxis])  # for GPU
        labels = tf.space_to_depth(labels, 8)
        shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
        labels = tf.concat([2 * labels, tf.ones(shape)], 3)
        # Add a small random matrix to randomly break ties in argmax
        labels = tf.argmax(labels + tf.random_uniform(tf.shape(labels), 0, 0.1),
                           axis=3)

        # Mask the pixels if bordering artifacts appear
        valid_mask = tf.ones_like(keypoint_map) if valid_mask is None else valid_mask
        valid_mask = tf.to_float(valid_mask[..., tf.newaxis])  # for GPU
        valid_mask = tf.space_to_depth(valid_mask, 8)
        valid_mask = tf.reduce_prod(valid_mask, axis=3)  # AND along the channel dim

        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits, weights=valid_mask)
        return loss



    import cv2
    keypoint_map = np.random.randint(0,2,(2,24,32))
    logits = np.random.uniform(-0.1,0.1, (2,65,3,4)).astype(np.float32)
    h = np.array([[1,0.5,0],[0.5,1,0],[0,0,1]]).astype(np.float32)
    mask = cv2.warpPerspective(np.ones([24,32]), h, (32,24))
    mask = np.stack([mask,mask])
    mask = mask.astype(np.int).astype(np.float32)
    ##
    loss = detector_loss_tf(keypoint_map, np.transpose(logits, (0,2,3,1)), valid_mask=mask)

    with tf.Session() as sess:
        l = sess.run(loss)
        print(l)

    keypoint_map = torch.tensor(keypoint_map)
    logits = torch.tensor(logits)
    mask = torch.tensor(mask)
    loss = detector_loss(keypoint_map, logits, valid_mask=mask, grid_size=8, device='cpu')
    print(loss)

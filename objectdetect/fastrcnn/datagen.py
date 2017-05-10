import theano

import numpy as np
import numpy.random as npr

from skimage.io import imread
from skimage.transform import resize
from skimage import color

from bnr_ml.utils.helpers import format_image

import pickle
import cv2

import pdb

def generate_examples_for_annotations(annotations, n_box=20, n_neg=200, n_pos=200, verbose=True):
    examples = []
    for i in range(annotations.__len__()):
        imsize = annotations[i]['size']
        annotation = annotations[i]['annotations']
        boxes = format_boxes(annotation)
        proposals = generate_proposal_boxes(boxes, imsize, n_box=n_box)
        neg_idx, pos_idx, obj_idx = find_valid_boxes(boxes, proposals)
        if neg_idx.size < n_neg:
            neg_idx = np.concatenate((neg_idx, npr.choice(neg_idx, size=n_neg - neg_idx.size, replace=True)))
        if pos_idx.size < n_pos:
            idx = np.arange(pos_idx.size)
            idx = np.concatenate((idx, npr.choice(idx, size=n_pos - idx.size, replace=True)))
            pos_idx, obj_idx = pos_idx[idx], obj_idx[idx]

        neg_idx = npr.choice(neg_idx, size=n_neg, replace=False)
        idx = np.arange(pos_idx.size); idx = npr.choice(idx, size=n_pos, replace=False)
        pos_idx, obj_idx = pos_idx[idx], obj_idx[idx]

        example = {}
        example['negative'] = proposals[neg_idx]
        pos_example = {}
        pos_example['proposals'] = proposals[neg_idx]
        pos_example['object'] = obj_idx
        example['positive'] = pos_example
        examples.append(example)
        if verbose:
            print('Proposals generated for {} of {} images.'.format(i+1, annotations.__len__()))
    return examples

def format_boxes(annotation):
    boxes = np.zeros((annotation.__len__(),4))
    for i in range(boxes.shape[0]):
        boxes[i] = [annotation[i]['x'], annotation[i]['y'], annotation[i]['w'], annotation[i]['h']]
    return boxes

def generate_proposal_boxes(boxes, image_size, n_box=20, min_size=.05 * .05):
    '''
    Generate proposal regions using boxes; boxes should be 
    an Nx4 matrix, were boxes[i] = [x,y,w,h]
    
    N - number of proposals per box
    '''
    proposals = np.zeros((0,4))
    
    for i in range(boxes.shape[0]):
        box = boxes[i]
        xi_b, yi_b, w_b, h_b = box[0], box[1], box[2], box[3]
        xf_b, yf_b = xi_b + w_b, yi_b + h_b
        
        xi, xf = np.linspace(0, xi_b + (3.*w_b)/4, n_box), np.linspace(xi_b, image_size[1], n_box)
        yi, yf = np.linspace(0, yi_b + (3.*h_b)/4, n_box), np.linspace(yi_b, image_size[0], n_box)
        
        xi, yi, xf, yf = np.meshgrid(xi, yi, xf, yf)
        xi, yi, xf, yf = xi.flatten(), yi.flatten(), xf.flatten(), yf.flatten()
        
        valid = np.bitwise_and(
            np.bitwise_and(xf > xi, yf > yi),
            (xf-xi) * (yf-yi) > min_size
        )
        
        proposal = np.concatenate(
            (
                xi[valid].reshape((-1,1)),
                yi[valid].reshape((-1,1)),
                (xf-xi)[valid].reshape((-1,1)),
                (yf-yi)[valid].reshape((-1,1))
            ),
            axis=1
        )
        
        proposals = np.concatenate((proposals, proposal), axis=0)

    return proposals

def find_valid_boxes(boxes, proposals):
    '''
    from the proposals, find the valid negative/positive examples
    '''
    boxes = boxes.reshape((1,) + boxes.shape)
    proposals = proposals.reshape(proposals.shape + (1,))
    
    # calculate iou
    xi = np.maximum(boxes[:,:,0], proposals[:,0])
    yi = np.maximum(boxes[:,:,1], proposals[:,1])
    xf = np.minimum(boxes[:,:,[0,2]].sum(axis=2), proposals[:,[0,2]].sum(axis=1))
    yf = np.minimum(boxes[:,:,[1,3]].sum(axis=2), proposals[:,[1,3]].sum(axis=1))
    
    w, h = np.maximum(xf - xi, 0.), np.maximum(yf - yi, 0.)
    
    isec = w * h
    union = boxes[:,:,2:].prod(axis=2) + proposals[:,2:].prod(axis=1) - isec
    
    iou = isec / union
    
    overlap = isec / boxes[:,:,2:].prod(axis=2)
    
    neg_idx = np.bitwise_and(
        np.bitwise_and(
            iou > 0.1, 
            iou < 0.5
        ),
        overlap < 1.
    )
    
    pos_idx = iou > 0.5
    
    # get any box which doesn't overlap with any box
    neg_idx = np.bitwise_and(np.sum(neg_idx, axis=1) >= 1, np.sum(iou < .5, axis=1) == iou.shape[1])
    # neg_idx = np.bitwise_and(np.sum(neg_idx, axis=1) > 0, np.sum(overlap < .4, axis=1) == iou.shape[1])
    #neg_idx = np.sum(neg_idx, axis=1) >= 1
    
    # filter out boxes that have an iou > .5 with more than one object
    pos_idx = np.sum(pos_idx, axis=1) == 1
    
    indices = np.arange(proposals.shape[0])
    
    # get object index for matched object
    obj_idx = iou[pos_idx,:].argmax(axis=1)
    
    return indices[neg_idx], indices[pos_idx], obj_idx

def colour_space_augmentation(im):
    im = color.rgb2hsv(im)
    im[:,:,2] *= (0.2 * npr.rand() + 0.9)
    idx = im[:,:,2] > 1.0
    im[:,:,2][idx] = 1.0
    im = color.hsv2rgb(im)
    return im

def generate_example(
        im,
        input_shape,
        num_classes,
        label_to_num,
        annotation,
        proposals,
        indices,
        n_neg,
        n_pos
    ):
    neg_idx, pos_idx, obj_idx = indices

    if neg_idx.size == 0 or pos_idx.size == 0:
        print('Warning, no valid prosals were given.')
        return None
    
    neg_examples = proposals[neg_idx,:]

    # X = np.zeros((n_neg + n_pos, 3) + input_shape, dtype=theano.config.floatX)
    y = np.zeros((n_neg + n_pos, 4 + num_classes + 1), dtype=theano.config.floatX)
    boxes = np.zeros((1,n_neg + n_pos,4), dtype=theano.config.floatX)
    
    try:
        pos_choice = npr.choice(np.arange(pos_idx.size), size=n_pos, replace=False)
    except:
        print('Warning, positive examples sampled with replacement from proposal boxes.')
        pos_choice = npr.choice(np.arange(pos_idx.size), size=n_pos, replace=True)
    try:
        neg_choice = npr.choice(np.arange(neg_idx.size), size=n_neg, replace=False)
    except:
        print('Warning, negative examples sampled with replacement from proposal boxes.')
        neg_choice = npr.choice(np.arange(neg_idx.size), size=n_neg, replace=True)
    
    neg_idx, pos_idx, obj_idx = neg_idx[neg_choice], pos_idx[pos_choice], obj_idx[pos_choice]
    
    # generate negative examples
    cls = np.zeros(num_classes + 1)
    cls[-1] = 1.
    coord = np.asarray([0.,0.,1.,1.])
    for i in range(n_neg):
        xi, yi = max(0,proposals[neg_idx[i],0]) / im.shape[1], max(0,proposals[neg_idx[i],1]) / im.shape[0]
        xf = min(im.shape[1], proposals[neg_idx[i],[0,2]].sum()) / im.shape[1]
        yf = min(im.shape[0], proposals[neg_idx[i],[1,3]].sum()) / im.shape[0]
        boxes[0,i,:] = [xi,yi,xf,yf]

        # subim = colour_space_augmentation(resize(im[yi:yf,xi:xf], input_shape))


        # if npr.rand() < 0.5:
        #   subim = subim[::-1,:]
        # if npr.rand() < 0.5:
        #   subim = subim[:,::-1]

        y[i,:4] = coord
        y[i,-(num_classes + 1):] = cls
        # X[i] = subim.swapaxes(2,1).swapaxes(1,0)
    
    cls[-1] = 0.
    # generate positive examples
    for i in range(n_pos):
        xi, yi = max(0,proposals[pos_idx[i],0]) / im.shape[1], max(0,proposals[pos_idx[i],1]) / im.shape[0]
        xf = min(im.shape[1], proposals[pos_idx[i],[0,2]].sum()) / im.shape[1]
        yf = min(im.shape[0], proposals[pos_idx[i],[1,3]].sum()) / im.shape[0]
        boxes[0,i+n_neg] = [xi,yi,xf,yf]

        # subim = colour_space_augmentation(resize(im[yi:yf,xi:xf], input_shape))

        x_box = (annotation[obj_idx[i]]['x'] + annotation[obj_idx[i]]['w']/2 - proposals[pos_idx[i],0]) / proposals[pos_idx[i],2]
        y_box = (annotation[obj_idx[i]]['y'] + annotation[obj_idx[i]]['h']/2 - proposals[pos_idx[i],1]) / proposals[pos_idx[i],3]
        log_w, log_h = np.log(float(annotation[obj_idx[i]]['w']) / proposals[pos_idx[i], 2]), np.log(float(annotation[obj_idx[i]]['h']) / proposals[pos_idx[i], 3])

        # # flip vertically randomly
        # if npr.rand() < 0.5:
        #   subim = subim[::-1,:]
        #   y_box = 1. - y_box
        # # flip horizontally
        # if npr.rand() < 0.5:
        #   subim = subim[:,::-1]
        #   x_box = 1. - x_box

        coord[:4] = [x_box, y_box, log_w, log_h]
        cls[label_to_num[annotation[obj_idx[i]]['label']]] = 1.
        # X[i+n_neg] = subim.swapaxes(2,1).swapaxes(1,0)
        y[i+n_neg,:4], y[i+n_neg,-(num_classes+1):] = coord, cls
 
    idx = np.arange(y.shape[0]); npr.shuffle(idx)   
    return boxes[:,idx], y[idx]

def generate_data(
        annotations,
        input_shape=None,
        num_classes=None,
        label_to_num=None,
        n_neg=9,
        n_pos=3,
        batch_size=2,
        n_box=20
    ):
    if not isinstance(annotations, np.ndarray):
        annotations = np.asarray(annotations)
    npr.shuffle(annotations)
    n_total = n_neg + n_pos
    cnt = 0

    X = np.zeros((batch_size, 3) + input_shape, dtype=theano.config.floatX)
    boxes = np.zeros((batch_size, n_total, 4), dtype=theano.config.floatX)
    y = np.zeros((n_total * batch_size, 4 + num_classes + 1), dtype=theano.config.floatX)

    for i in range(annotations.size):
        proposal_boxes = format_boxes(annotations[i]['annotations'])
        proposals = generate_proposal_boxes(proposal_boxes, annotations[i]['size'], n_box=n_box)
        indices = find_valid_boxes(proposal_boxes, proposals)
        im = format_image(imread(annotations[i]['image']), dtype=theano.config.floatX)
        data = generate_example(im, input_shape, num_classes, label_to_num, annotations[i]['annotations'], proposals, indices, n_neg, n_pos)
        im = resize(im, input_shape)
        if data is not None:
            X[cnt] = im.swapaxes(2,1).swapaxes(1,0)
            boxes[cnt,:,:], y[cnt*n_total:(cnt+1)*n_total] = data[0], data[1]
            # X[cnt*n_total:(cnt+1)*n_total], y[cnt*n_total:(cnt+1)*n_total] = data[0], data[1]
            cnt += 1

        if cnt == batch_size or (i-1) == annotations.size:
            X, boxes, y = X[:cnt], boxes[:cnt], y[:cnt*n_total]
            if X.shape[0] > 0:
                
                yield X, boxes, y
            X = np.zeros((batch_size, 3) + input_shape, dtype=theano.config.floatX)
            boxes = np.zeros((batch_size, n_total, 4), dtype=theano.config.floatX)
            y = np.zeros((n_total * batch_size, 4 + num_classes + 1), dtype=theano.config.floatX)
            cnt = 0

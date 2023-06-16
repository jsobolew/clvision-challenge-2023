from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.models.utils import avalanche_forward

import torch

import numpy as np
import random
import cv2

class MyPluginNoise(SupervisedPlugin):
    """
    Implemented your plugin (if any) here.
    """

    def __init__(
        self,
    ):
        """
        :param
        """
        super().__init__()

    def before_backward(self, strategy, **kwargs):
        """
        Example callback: before backward.
        """
        
        device = strategy.device

        # get noise images bath
        X, y = self.generate_noise_img_batch(strategy.train_mb_size, strategy.model.linear.out_features)
        X, y = X.to(device), y.to(device)


        # calculate loss
        out = avalanche_forward(strategy.model, X, y)
        noise_loss = strategy._criterion(out, y)

        # add to loss
        strategy.loss += noise_loss

    def generate_noise_img_batch(self, batch_size, num_classes):
        img_batch = np.zeros((batch_size, 3, 32, 32))
        labels = np.random.randint(num_classes, size=batch_size)

        for i in range(batch_size):
            img_batch[i,:,:,:] = self.dead_leaves(32, 1, shape_mode='mixed', max_iters=5000)/255

        return torch.from_numpy(img_batch).float(), torch.from_numpy(labels).long()


    def dead_leaves(self, res, sigma, shape_mode='mixed', max_iters=5000):
        img = np.zeros((res, res, 3), dtype=np.uint8)
        rmin = 0.03
        rmax = 1

        # compute distribution of radiis (exponential distribution with lambda = sigma):
        k = 200
        r_list = np.linspace(rmin, rmax, k)
        r_dist = 1./(r_list ** sigma)
        if sigma > 0:
            # normalize so that the tail is 0 (p(r >= rmax)) = 0
            r_dist = r_dist - 1/rmax**sigma
        r_dist = np.cumsum(r_dist)
        # normalize so that cumsum is 1.
        r_dist = r_dist/r_dist.max()

        for i in range(max_iters):
            available_shapes = ['circle', 'square', 'oriented_square','rectangle', 'triangle', 'quadrilater']
            assert shape_mode in available_shapes or shape_mode == 'mixed'
            if shape_mode == 'mixed':
                shape = random.choice(available_shapes)
            else:
                shape = shape_mode

            color = tuple([int(k) for k in np.random.uniform(0, 1, 3) * 255])

            r_p = np.random.uniform(0,1)
            r_i = np.argmin(np.abs(r_dist - r_p))
            radius = max(int(r_list[r_i] * res), 1)

            center_x, center_y = np.array(np.random.uniform(0,res, size=2),dtype='uint8')
            if shape == 'circle':
                img = cv2.circle(img, (center_x, center_y),radius=radius, color=color, thickness=-1)
            else:
                if shape == 'square' or shape == 'oriented_square':
                    side = radius * np.sqrt(2)
                    corners = np.array(((- side / 2, - side / 2),
                                        (+ side / 2, - side / 2),
                                        (+ side / 2, + side / 2),
                                        (- side / 2, + side / 2)), dtype='int32')
                    if shape == 'oriented_square':
                        theta = np.random.uniform(0, 2 * np.pi)
                        c, s = np.cos(theta), np.sin(theta)
                        R = np.array(((c, -s), (s, c)))
                        corners = (R @ corners.transpose()).transpose()
                elif shape == 'rectangle':
                    # sample one points in the firrst quadrant, and get the two other symmetric
                    a = np.random.uniform(0, 0.5*np.pi, 1)
                    corners = np.array(((+ radius * np.cos(a), + radius * np.sin(a)),
                                        (+ radius * np.cos(a), - radius * np.sin(a)),
                                        (- radius * np.cos(a), - radius * np.sin(a)),
                                        (- radius * np.cos(a), + radius * np.sin(a))), dtype='int32')[:,:,0]

                else:
                    # we sample three or 4 points on a circle of the given radius
                    angles = sorted(np.random.uniform(0, 2*np.pi, 3 if shape == 'triangle' else 4))
                    corners = []
                    for a in angles:
                        corners.append((radius * np.cos(a), radius * np.sin(a)))

                corners = np.array((center_x, center_y)) + np.array(corners)
                img = cv2.fillPoly(img, np.array(corners, dtype='int32')[None], color=color)
            if (img.sum(-1) == 0).sum() == 0:
                break

        img = img.transpose((2,0,1))
        return np.clip(img, 0, 255)
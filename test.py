"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import time

import torch
import math

def test(opt):
    dataloader = data.create_dataloader(opt)

    model = Pix2PixModel(opt)
    model.eval()

    visualizer = Visualizer(opt)

    # create a webpage that summarizes the all results
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))
    time_total = 0
    # test
    for i, data_i in enumerate(dataloader):
        torch.cuda.reset_max_memory_allocated()
        if i * opt.batchSize >= opt.how_many:
            break
        print(i)
        start = time.time()
        generated = model(data_i, mode='inference')
        torch.cuda.synchronize(device='cuda')
        end = time.time()
        f_time = end-start
        if i != 0:
            time_total += f_time
        print("time_%d:%f"%(i,f_time))
        print(torch.cuda.max_memory_allocated(device=None))

        img_path = data_i['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('gt', data_i['image'][b]),
                                   ('synthesized_image', generated[b])])
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])

    webpage.save()
    print("average time per image = %f" % (time_total/(i)))

if __name__ == '__main__':
    opt = TestOptions().parse()
    test(opt)

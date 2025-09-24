import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
import skimage.measure
import cv2
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from lib.datasets.kitti.kitti_eval_python.eval import get_distance_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
from copy import deepcopy
from .pd import PhotometricDistort, PhotometricDistortStereo

class CropTop(object):
    def __init__(self, crop_top_index=None, output_height=None):
        if crop_top_index is None and output_height is None:
            print("Either crop_top_index or output_height should not be None, set crop_top_index=0 by default")
            crop_top_index = 0
        if crop_top_index is not None and output_height is not None:
            print("Neither crop_top_index or output_height is None, crop_top_index will take over")
        self.crop_top_index = crop_top_index
        self.output_height = output_height
        self.upper = self.crop_top_index

    def __call__(self, left_image, right_image=None, calib=None, disp_P2=None):
        # C * H * W
        height, width = left_image.shape[1:3]

        if self.crop_top_index is not None:
            h_out = height - self.crop_top_index
            upper = self.crop_top_index
        else:
            h_out = self.output_height
            upper = height - self.output_height
        lower = height
        self.upper = upper
        left_image = left_image[:, upper:lower, :]
        # left_image[:, 0:upper, :] = left_image[:, 0:upper, :]*0+1
        if right_image is not None:
            right_image = right_image[:, upper:lower, :]
            # right_image[:, 0:upper, :] = right_image[:, 0:upper, :]*0+1
        # if calib is not None:
        #     calib.P2[1, 2] = calib.P2[1, 2] - upper               # cy' = cy - dv
        #     calib.P2[1, 3] = calib.P2[1, 3] - upper * calib.P2[2, 3] # ty' = ty - dv * tz

        if disp_P2 is not None:
             disp_P2 = disp_P2[:, upper//4:lower//4, :]
        return left_image, right_image, disp_P2


class KITTI_Dataset(data.Dataset):
    def __init__(self, split, cfg):

        # basic configuration
        self.root_dir = cfg.get('root_dir')
        self.train_txt = cfg.get('train_txt', None)
        self.eval_txt = cfg.get('eval_txt', None)
        self.subset = cfg.get('subset', None)
        self.root_dir_eval= cfg.get('root_dir_eval')
        self.get_img_right = cfg.get('get_img_right')
        self.get_disparity = cfg.get('get_disparity')
        self.get_disparity_on_the_fly = cfg.get('get_disparity_on_the_fly')
        if split in ['train']:
            self.repeat_labels = cfg.get('repeat_labels')
        else:
            self.repeat_labels = 1
        self.split = split
        self.num_classes = 3
        self.max_objs = 50
        self.writelist_train = cfg.get('writelist_train')
        self.writelist_Eval =  cfg.get('writelist_Eval')

        self.cls2id = {}
        for i, cls in enumerate(self.writelist_train):
            self.cls2id[cls] = i
        self.resolution_ori = np.array(cfg.get('resolution_ori'))
        self.ori_size_check_filter = cfg.get('resolution_ori_check_filter')
        self.crop_top = cfg.get('crop_top', 0)
        self.resolution_top_croped = np.array([self.resolution_ori[0], 
                                                self.resolution_ori[1]-self.crop_top])  # W * H
        self.resolution_disp_ori = np.array([self.resolution_ori[0]//4, self.resolution_ori[1]//4])
        self.resolution_disp = np.array([self.resolution_ori[0]//4, self.resolution_top_croped[1]//4])
        self.use_3d_center = cfg.get('use_3d_center', True)
        
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        self.use_dontcare = cfg.get('use_dontcare', False)
        self.ingore_far_object = cfg.get('ingore_far_object', True)
        self.ingore_3dcenter_out_of_img = cfg.get('ingore_3dcenter_out_of_img', True)
        self.drop_occupied = cfg.get('drop_occupied', True)
        self.drop_unkown_level = cfg.get('drop_unkown_level', False)
        self.decouple = cfg.get('decouple', False)
        self.get_depth_sample_points = cfg.get('get_depth_sample_points', True)

        if self.class_merging:
            self.writelist_train.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist_train.extend(['DontCare'])

        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test', 'train_self_val']
        print("self.split", self.split)
        if self.split in ['val','test']:
            if self.eval_txt is not None:
                self.split_file = self.eval_txt 
            else:
                self.split_file = os.path.join(self.root_dir_eval, 'ImageSets', self.split + '.txt')
        else:
            if self.train_txt is not None:
                self.split_file = self.train_txt
            else:
                self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        if self.split in ['val','test']:
            self.data_dir = self.root_dir_eval
        else:
            self.data_dir = self.root_dir
  
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.image3_dir = os.path.join(self.data_dir, 'image_3')
        self.disparity_dir = os.path.join(self.data_dir, 'cv2disps')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.aug_pd = cfg.get('aug_pd', False)
        self.aug_crop = cfg.get('aug_crop', False)
        self.aug_calib = cfg.get('aug_calib', False)

        self.random_mixup3d = cfg.get('random_mixup3d', 0)
        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_switch = cfg.get('random_switch', 0)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.stereo_shift_aug = cfg.get('stereo_shift_aug', 0)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

        self.depth_scale = cfg.get('depth_scale', 'normal')

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ],
                                       [1.52563191462 ,1.62856739989, 3.88311640418],
                                       [1.73698127    ,0.59706367   , 1.76282397   ]])
        if not self.meanshape:
            # self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)
            self.cls_mean_size = np.zeros((len(self.writelist_train), 3), dtype=np.float32)

        # others
        self.downsample = 32
        if self.get_img_right:
            self.pd = PhotometricDistortStereo()
            self.pd_stereo = True 
        else:
            self.pd = PhotometricDistort()
            self.pd_stereo = False 
        self.croptop_func = CropTop(self.crop_top)

        self.clip_2d = cfg.get('clip_2d', False)

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file), img_file
        return Image.open(img_file)    # (H, W, 3) RGB mode
    
    def get_image3(self, idx):
        img3_file = os.path.join(self.image3_dir, '%06d.png' % idx)
        assert os.path.exists(img3_file)
        return Image.open(img3_file)    # (H, W, 3) RGB mode
    
    def get_disparity_P2(self, idx):
        disp_file = os.path.join(self.disparity_dir, 'P2%06d.png' % idx)
        if not os.path.exists(disp_file):
            return None
        return Image.open(disp_file)    # (H, W, 3) RGB mode
    
    def get_disparty_P2_onfly(self, img_left, img_right, flip_flag, switch_flag):
        left_image = cv2.cvtColor(np.asarray(img_left),cv2.COLOR_RGB2BGR)
        right_image = cv2.cvtColor(np.asarray(img_right),cv2.COLOR_RGB2BGR)
        gray_image1 = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        matcher = cv2.StereoBM_create(192, 25)
        if switch_flag or flip_flag:
            gray_image1_flip = cv2.flip(gray_image1, 1)
            gray_image2_flip = cv2.flip(gray_image2, 1)
            disparity_left = matcher.compute(gray_image1_flip, gray_image2_flip)
            disparity_left = disparity_left[:,::-1]
            disparity_left[disparity_left < 0] = 0
        else :
            disparity_left = matcher.compute(gray_image1, gray_image2)
            disparity_left[disparity_left < 0] = 0
            
        disparity_left = disparity_left.astype(np.uint16)
        disparity_left = skimage.measure.block_reduce(disparity_left, (4,4), np.max) 
        # disparity_left = Image.fromarray(disparity_left) 
        return disparity_left

    def get_disparity_P3(self, idx):
        disp_file = os.path.join(self.disparity_dir, 'P3%06d.png' % idx)
        assert os.path.exists(disp_file)
        return Image.open(disp_file)    # (H, W, 3) RGB mode

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file), label_file
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)
    

    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)
        test_id = {'Car': 0, 'Pedestrian':1, 'Cyclist': 2, 
                   'Van': 3,  'Person_sitting': 4, 'Truck':5}
        
        logger.info('==> Evaluating (official) ...')
        car_moderate = 0
        for category in self.writelist_Eval:
            results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            if category == 'Car':
                car_moderate = mAP3d_R40
            logger.info(results_str)
        return car_moderate
    
    def get_depth_map(self, img_size_original, objects):
        W, H = img_size_original
        depth_maps = torch.zeros((H, W))
        depth_maps = np.ones((H, W))*120
        # 构造depth map，每个像素点的深度值为离相机的距离，仅保留最近距离
        for obj in objects:
            depth_map_obj = np.ones((H, W))*120
            if obj.cls_type == 'DontCare':
                continue
            x1, y1, x2, y2 = obj.box2d # x1, y1, x2, y2
            depth_obj = obj.pos[-1]
            depth_map_obj[int(y1):int(y2), int(x1):int(x2)] = depth_obj
            # depth_maps = np.max(depth_maps, depth_map_obj)
            depth_maps = np.where(depth_map_obj < depth_maps, depth_map_obj, depth_maps)
        return depth_maps

    def get_sample_point(self, obj, depth_map):
        obj_depth = obj.pos[-1]
        depth_map_filter = np.where(depth_map == obj_depth, obj_depth, 0)
        if np.max(depth_map_filter) == 0:
            return None
        # sum by row计算x采样点
        col_sum = np.sum(depth_map_filter, axis=0)
        max_col = np.max(col_sum)
        col_filter = np.where(col_sum == max_col)[0]
        row_index = np.mean(col_filter)
        row_index = row_index

        # sum by row
        row_sum = np.max(depth_map_filter, axis=1)
        max_row = np.max(row_sum)
        row_filter = np.where(row_sum == max_row)[0]
        col_index = np.mean(row_filter)
        col_index = col_index
        return [row_index, col_index]

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        if self.get_img_right:
            img3 = self.get_image3(index)
        else:
            img3 = None

        if self.get_disparity and not self.get_disparity_on_the_fly:
                disp_P2 = self.get_disparity_P2(index)
        else:
            disp_P2 = None
        img_size_original = np.array(img.size) #W H
        features_size = self.resolution_top_croped // self.downsample    # W * H

        # data augmentation for image
        center = np.array(img_size_original) / 2
        crop_size, crop_scale = img_size_original, 1.0
        random_flip_flag, random_crop_flag, random_switch_flag = False, False, False
        random_mix_flag = False
        calib = self.get_calib(index)

        if self.data_augmentation:
            if np.random.random() < self.random_mixup3d:
                random_mix_flag = True
            if self.aug_pd:
                if self.pd_stereo:
                    img = np.array(img).astype(np.float32)
                    img3 = np.array(img3).astype(np.float32)
                    img, img3 = self.pd(img, img3)
                    img = Image.fromarray(img.astype(np.uint8))
                    if img3 is not None:
                        img3 = Image.fromarray(img3.astype(np.uint8))

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if img3 is not None:
                    img3 = img3.transpose(Image.FLIP_LEFT_RIGHT)
                if disp_P2 is not None:
                    disp_P2 = disp_P2.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_switch and not random_flip_flag:
                random_switch_flag = True
                tem_img = deepcopy(img)
                img = deepcopy(img3)
                img3 = tem_img
                disp_P2 = disp_P2 #TODO trans to p3 @mushiyi 

            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    random_crop_flag = True
                    crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                    crop_size = img_size_original * crop_scale
                    center[0] += img_size_original[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    center[1] += img_size_original[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
            if np.random.random() < self.stereo_shift_aug :
                if img3 is not None:  
                    img3 = np.array(img3)  
                    x_shift = np.random.randint(-1, 2)
                    img3 = np.roll(img3, shift=(0, x_shift, 0), axis=(0, 1, 2))
                    img3 = Image.fromarray(img3)

        if random_mix_flag == True:
            count_num = 0
            random_mix_flag = False
            while count_num < 50:
                count_num += 1
                random_index = int(np.random.choice(self.idx_list))
                calib_temp = self.get_calib(random_index)
                
                if calib_temp.cu == calib.cu and calib_temp.cv == calib.cv and calib_temp.fu == calib.fu and calib_temp.fv == calib.fv:
                    img_temp = self.get_image(random_index)
                    if self.get_img_right:
                        img_temp3 = self.get_image3(random_index)
                    img_size_temp = np.array(img_temp.size)
                    dst_W_temp, dst_H_temp = img_size_temp
                    if dst_W_temp == img_size_original[0] and dst_H_temp == img_size_original[1]:
                        objects_1 = self.get_label(index)
                        objects_2 = self.get_label(random_index)
                        if len(objects_1) + len(objects_2) < self.max_objs: 
                            random_mix_flag = True
                            if random_flip_flag == True:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                                if self.get_img_right:
                                    img_temp3 = img_temp3.transpose(Image.FLIP_LEFT_RIGHT)
                            img_blend = Image.blend(img, img_temp, alpha=0.5)
                            img = img_blend
                            if self.get_img_right:
                                img_blend3 = Image.blend(img3, img_temp3, alpha=0.5)
                                img3 = img_blend3
                            break    

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution_ori, inv=1)
        img = img.transform(tuple(self.resolution_ori.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        if img3 is not None:
            img3 = img3.transform(tuple(self.resolution_ori.tolist()),
                                method=Image.AFFINE,
                                data=tuple(trans_inv.reshape(-1).tolist()),
                                resample=Image.BILINEAR)
        if disp_P2 is not None:
            # 先放大4倍
            disp_P2 = disp_P2.resize(tuple(img_size_original.tolist()))
            disp_P2 = disp_P2.transform(tuple(self.resolution_ori.tolist()),
                                method=Image.AFFINE,
                                data=tuple(trans_inv.reshape(-1).tolist()),
                                resample=Image.Resampling.NEAREST)
            disp_P2 = disp_P2.resize(tuple(self.resolution_disp_ori.tolist()))

        if self.get_disparity and self.get_disparity_on_the_fly:
            
            disp_P2 = self.get_disparty_P2_onfly(img, img3, random_flip_flag, random_switch_flag)

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W
        if img3 is not None:
            img3 = np.array(img3).astype(np.float32) / 255.0
            img3 = (img3 - self.mean) / self.std
            img3 = img3.transpose(2, 0, 1)  # C * H * W

        if disp_P2 is not None:
            disp_P2 = np.array(disp_P2).astype(np.float32) / 16.0
            disp_P2 = np.expand_dims(disp_P2, axis=0)
            if not self.get_disparity_on_the_fly:
                scale_w = self.resolution_ori[0] / img_size_original[0]
                disp_P2 = disp_P2 * scale_w / crop_scale 
    
        
        if self.crop_top > 0:
            img, img3, disp_P2 = self.croptop_func(img, img3, calib, disp_P2)
        img_size_croped = np.array([img.shape[-1],img.shape[-2]]) # w H

        info = {'img_id': index,
                'img_size_original': img_size_original,
                'img_size_croped': img_size_croped,
                "upper": self.croptop_func.upper,
                'bbox_downsample_ratio': img_size_croped / features_size}
        # if self.split in ['test', 'val']:
        if self.split in ['test']:
            if img3 is not None:
                img = np.concatenate([img, img3], axis=0)
            return img, calib.P2, img, info

        #  ============================   get labels   ==============================
        objects = self.get_label(index)
        calib = self.get_calib(index)
        if random_mix_flag == True:
            objects_mix = self.get_label(random_index)
            objects.extend(objects_mix)
        # data augmentation for labels
        if random_flip_flag:
            if self.aug_calib:
                calib.flip(img_size_original)
            for object in objects:
                [x1, _, x2, _] = object.box2d
                object.box2d[0],  object.box2d[2] = img_size_original[0] - x2, img_size_original[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if self.aug_calib:
                    object.pos[0] *= -1
                if object.alpha > np.pi:  object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi: object.alpha += 2 * np.pi
                if object.ry > np.pi:  object.ry -= 2 * np.pi
                if object.ry < -np.pi: object.ry += 2 * np.pi

        if random_switch_flag:
            calib.switch2right()
            for object in objects:
                corners3d = object.generate_corners3d()
                corners3d = corners3d.reshape(-1,8, 3)
                box2d_new , _  = calib.corners3d_to_img_boxes(corners3d)
                object.box2d = box2d_new[0]
                object.box2d[0] = max(1, object.box2d[0])
                object.box2d[1] = max(1, object.box2d[1])
                object.box2d[2] = min(img_size_original[0]-1, object.box2d[2])
                object.box2d[3] = min(img_size_original[1]-1, object.box2d[3])
 
        # labels encoding
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32) 
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)
        sample_points = np.zeros((self.max_objs, 2), dtype=np.float32)
        boxes_ry = np.zeros((self.max_objs, 1), dtype=np.float32)

        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
        if self.get_depth_sample_points is not None:
            depth_map = self.get_depth_map(img_size_original, objects)
        for i in range(object_num):
            if self.ori_size_check_filter == True:
                if img_size_original[0] != 1242:
                    continue
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist_train:
                continue

            # filter inappropriate samples
            if self.drop_unkown_level and objects[i].level_str == 'UnKnown' :
                continue

            if objects[i].pos[-1] < 2:
                continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = 65
            if self.ingore_far_object:
                if objects[i].pos[-1] > threshold:
                    continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()
            if self.get_depth_sample_points is True:
                depth_sample_point = self.get_sample_point(objects[i], depth_map) #已经随2d框完成变换
            else:
                # depth_sample_point = center_3d[:2].copy()
                depth_sample_point = None

            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            if depth_sample_point is not None:
                depth_sample_point = affine_transform(depth_sample_point, trans)
                
            if self.crop_top > 0:
                bbox_2d[1] -= self.croptop_func.upper
                bbox_2d[3] -= self.croptop_func.upper
                if depth_sample_point is not None:
                    depth_sample_point[1] = depth_sample_point[1] - self.croptop_func.upper
         
            bbox_2d = np.maximum(bbox_2d, 0)
            # process 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
            corner_2d = bbox_2d.copy()

            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            

            if random_flip_flag and not self.aug_calib:  # random flip for center3d
                center_3d[0] = img_size_original[0] - center_3d[0]

            center_3d = affine_transform(center_3d.reshape(-1), trans)
            
            if self.crop_top > 0:
                center_3d[1] = center_3d[1] - self.croptop_func.upper
            if depth_sample_point is None:
                depth_sample_point = center_3d[0:2].copy()
            # filter 3d center out of img
            if self.ingore_3dcenter_out_of_img:
                proj_inside_img = True
                if center_3d[0] < 0 or center_3d[0] >= self.resolution_top_croped[0]: 
                    proj_inside_img = False
                if center_3d[1] < 0 or center_3d[1] >= self.resolution_top_croped[1]: 
                    proj_inside_img = False

                if proj_inside_img == False:
                        continue

            # class
            cls_id = self.cls2id[objects[i].cls_type]
            labels[i] = cls_id

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1. * w, 1. * h

            center_2d_norm = center_2d / self.resolution_top_croped
            size_2d_norm = size_2d[i] / self.resolution_top_croped

            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution_top_croped
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution_top_croped
            center_3d_norm = center_3d / self.resolution_top_croped
            depth_sample_point_norm = depth_sample_point / self.resolution_top_croped

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                # else:
                #     continue		

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b
            sample_points[i] = depth_sample_point_norm[0], depth_sample_point_norm[1]
            # encoding depth
            if self.depth_scale == 'normal':
                depth[i] = objects[i].pos[-1] * crop_scale
            
            elif self.depth_scale == 'inverse':
                depth[i] = objects[i].pos[-1] / crop_scale
            
            elif self.depth_scale == 'none':
                depth[i] = objects[i].pos[-1]
            # encoding heading angle
            heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
            boxes_ry[i] = objects[i].ry
            if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi: heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding size_3d
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size
            if self.decouple:
                scale_h1 = self.resolution_ori[1] / img_size_original[1]
                scale_w1 = self.resolution_ori[0] / img_size_original[0]
                size_3d[i][0] = size_3d[i][0] / depth[i] * calib.P2[0, 0] * scale_h1
                size_3d[i][1:] = size_3d[i][1:] / depth[i] * calib.P2[0, 0] * scale_w1
            if self.drop_occupied:
                if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                    mask_2d[i] = 1
                else:
                    mask_2d[i] = 0
            else:
                    mask_2d[i] = 1

            calibs[i] = calib.P2

        # collect return data
        inputs = img
        if img3 is not None:
            inputs = np.concatenate([img, img3], axis=0)
        if disp_P2 is None:
            disp_P2 = np.zeros((1,72,320))
        
        targets = {
                   'calibs': calibs,
                   'indices': indices,
                   'img_size_original': img_size_original, 
                   'img_size_croped': img_size_croped,
                   'labels': labels,
                   'boxes': boxes,
                   'boxes_3d': boxes_3d,
                   'sample_points': sample_points,
                   'depth': depth,
                   'size_2d': size_2d,
                   'size_3d': size_3d,
                   'src_size_3d': src_size_3d,
                   'heading_bin': heading_bin,
                   'boxes_ry': boxes_ry,
                   'heading_res': heading_res,
                   'mask_2d': mask_2d,
                   'disp': disp_P2,
                   "random_flip_flag": random_flip_flag,
                   "random_switch_flag": random_switch_flag,
                   'img_id': index}
        info = {'img_id': index,
                'img_size_original': img_size_original,
                'img_size_croped': img_size_croped,
                "upper": self.croptop_func.upper,
                'bbox_downsample_ratio': img_size_croped / features_size}
        return inputs, calib.P2, targets, info


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'root_dir': '../../../data/KITTI',
           'random_flip': 0.0, 'random_crop': 1.0, 'scale': 0.8, 'shift': 0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False}
    dataset = KITTI_Dataset('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break

    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())

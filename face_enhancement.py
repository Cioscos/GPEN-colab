'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import glob
import time
import numpy as np
from pathlib import Path
from PIL import Image
import __init_paths
from retinaface.retinaface_detection import RetinaFaceDetection
from face_model.face_gan import FaceGAN
from align_faces import warp_and_crop_face, get_reference_facial_points
from skimage import transform as tf
from DFLIMG.DFLJPG import DFLJPG

class FaceEnhancement(object):
    def __init__(self, base_dir='./', size=512, model=None, channel_multiplier=2):
        self.facedetector = RetinaFaceDetection(base_dir)
        self.facegan = FaceGAN(base_dir, size, model, channel_multiplier)
        self.size = size
        self.threshold = 0.9

        # the mask for pasting restored faces back
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)

        self.kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
                (self.size, self.size), inner_padding_factor, outer_padding, default_square)

    def process(self, img):
        facebs, landms = self.facedetector.detect(img)
        
        orig_faces, enhanced_faces = [], []
        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4]<self.threshold: continue
            fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts, crop_size=(self.size, self.size))
            
            # enhance the face
            ef = self.facegan.process(of)
            
            orig_faces.append(of)
            enhanced_faces.append(ef)
            
            tmp_mask = self.mask
            tmp_mask = cv2.resize(tmp_mask, ef.shape[:2])
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

            if min(fh, fw)<100: # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)
            
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask>0)] = tmp_mask[np.where(mask>0)]
            full_img[np.where(mask>0)] = tmp_img[np.where(mask>0)]

        full_mask = full_mask[:, :, np.newaxis]
        img = cv2.convertScaleAbs(img*(1-full_mask) + full_img*full_mask)

        return img, orig_faces, enhanced_faces

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dirs):
    images = []
    assert os.path.isdir(dirs), '%s is not a valid directory' % dirs

    for root, _, fnames in sorted(os.walk(dirs)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


if __name__=='__main__':
    model = {'name': 'GPEN-512', 'size': 512}
    file_path = os.getcwd()
    file_path.replace('/face_enhancement2.py', '')
    indir = os.path.join(file_path, 'examples/imgs')
    outdir = 'examples/outs'
    os.makedirs(outdir, exist_ok=True)

    faceenhancer = FaceEnhancement(size=model['size'], model=model['name'], channel_multiplier=2)

    # files = sorted(glob.glob(os.path.join(indir, '*.*g')))
    imgPaths = make_dataset(indir)
    is_dfl_image = False

    for n, file in enumerate(imgPaths):
        InputDflImg = DFLJPG.load(file)
        if not InputDflImg or not InputDflImg.has_data():
            print('\t################ No landmarks in file {}'.format(file))
            is_dfl_image = False
        else:
            is_dfl_image = True
            Landmarks = InputDflImg.get_landmarks()
            InputData = InputDflImg.get_dict()
            if InputDflImg.has_seg_ie_polys():
                xseg_polys = InputDflImg.get_seg_ie_polys()
                for poly in xseg_polys:
                    poly.set_points(poly.get_pts())

        filename = os.path.basename(file)
        
        im = cv2.imread(file, cv2.IMREAD_COLOR)  # BGR
        if not isinstance(im, np.ndarray): print(filename, 'error'); continue
        im = cv2.resize(im, (0, 0), fx=1, fy=1)

        img, orig_faces, enhanced_faces = faceenhancer.process(im)

        cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1])+'.jpg'), img)

        if is_dfl_image:
            OutputDflImg = DFLJPG.load(os.path.join(outdir, '.'.join(filename.split('.')[:-1])+'.jpg'))
            OutputDflImg.set_dict(InputData)
            OutputDflImg.set_landmarks(Landmarks)
            if InputDflImg.has_seg_ie_polys():
                OutputDflImg.set_seg_ie_polys(xseg_polys)
            OutputDflImg.save()
            
        if n % 10 == 0: print(n, filename)

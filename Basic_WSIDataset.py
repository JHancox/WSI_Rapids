%%time
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import multiprocessing
import os
import torch
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ProcessPoolExecutor, Executor, as_completed
from cucim import CuImage
import time
from multiprocessing import Manager
import matplotlib.pyplot as plt

class WSIDataset(torch.utils.data.IterableDataset):
    
    def process_chunk(self, start_loc, inp_file):

        # Loads a tile of image data and returns
        # the coords if > threshold
        level = self.reduction_level
        slide = CuImage(inp_file)
        low_res_patch_size = self.patch_size // (2 ** level )

        tile = np.array(slide.read_region(start_loc, [low_res_patch_size, low_res_patch_size], level))
        if tile.flatten().var() > 20:
            return start_loc
        
    def get_size(self, image, level=0):
        # returns imaage dimensions at the specified reduction level
        wsi = CuImage(image)

        sizes=wsi.metadata["cucim"]["resolutions"]
        w = sizes["level_dimensions"][level][0]
        h = sizes["level_dimensions"][level][1]
        
        return w,h

    def generate_patch_list(self, image_file):
        # generates a list of patches that are above the threshold
        w,h = self.get_size(image_file, level = 0)

        start_loc_data = [(sx, sy)        
            for sy in range(0, h, self.patch_size )
                for sx in range(0, w, self.patch_size )]

        results=[]

        with ProcessPoolExecutor(max_workers=16) as executor:
            result_futures = {executor.submit(self.process_chunk, loc,image_file): loc for loc in start_loc_data}

            for future in as_completed(result_futures):
                res1 = future.result()
                if res1:
                    results.append(res1)

          
        print("{} Patches found".format(len(results)))
        return results  

    def __init__(self, images=[], reduction_level=0, patch_size=256):
        super(WSIDataset).__init__()
        self.image_list = images
        self.patch_list = []
        self.reduction_level = reduction_level
        self.patch_size = patch_size
        
        for i, image in enumerate(images):
            patches = self.generate_patch_list(image)
            self.patch_list.append(patches)


    def __iter__(self):
        for i, patch in enumerate(self.patch_list):
            image = self.image_list[i]
            wsi = CuImage(image)

            for item in patch:
                img = np.array(wsi.read_region(location=item, size=(256, 256), level=0))
                if img.shape == (256,256,3):
                    nvtx.range_pop()
                    yield np.array(np.moveaxis(img,-1,0)),item
                else:
                    print(img.shape)
                    assert 1==2

 
    def generate_mask(self):
        # for demo purposes
        print("Generating Mask")
        
        w,h = self.get_size(self.image_list[0], level = 0)
                
        mask = np.zeros((h // self.patch_size , w // self.patch_size), dtype=np.uint8)
        patches = self.patch_list[0]

        for patch in patches:
                
            y=patch[0] // self.patch_size 
            x=patch[1] // self.patch_size
            mask[x,y] = 255

        return mask
    
# Create a new Dataset and do the thresolding dynamically        
dataset = WSIDataset(['patient_100_node_0.tif'], patch_size=512,reduction_level=3)

# Creates an image with each pixel representing a patch and its
# threshold status (e.g. above or below)
mask = dataset.generate_mask()

plt.figure(figsize=(10,10))
plt.imshow(mask)
plt.title('mask')
plt.show()

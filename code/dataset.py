import torch.utils.data as data
import util
import numpy as np
import random

class GANstronomyDataset(data.Dataset):
    # data = GANstronomyDataset(DATA_PATH, [0.6, 0.2, 0.2])
    # data.set_split_index(0)
    def __init__(self, data_path, split):
        dataset = util.unpickle(data_path)
        self.data = dataset['data']
        self.class_mapping = dataset['class_mapping'] # Maps raw class to mapped class
        self.inv_class_mapping = {self.class_mapping[c]:c for c in self.class_mapping}
        self.ids = sorted(list(self.data.keys()))
        random.Random(0).shuffle(self.ids)
        self.split = np.array(split) / np.sum(split)
        self.split_sizes = np.floor(self.split * len(self.ids))
        self.split_index = 0
        self.split_starts = np.insert(np.cumsum(self.split_sizes), 0, [0])
        self.split_starts = [int(split_start) for split_start in self.split_starts]
        self.id_sets = []
        for i in range(len(self.split_sizes) - 1):
            self.id_sets.append(set(self.ids[self.split_starts[i]:self.split_starts[i+1]]))
        np.random.seed(0)
        self.noisy_real = np.random.rand(len(self.ids)) * 0.1 + 0.9
        self.noisy_fake = np.random.rand(len(self.ids)) * 0.1
        
    def __getitem__(self, index):
        real_index = self.get_real_index(index)
        item = self.data[self.ids[real_index]]
        recipe_id = item['recipe_id']
        img_id = item['img_id']
        klass = item['class']
        recipe_emb = item['recipe_emb']
        img = item['img_pre'] # The preprocessed image
        noisy_real = self.noisy_real[real_index]
        noisy_fake = self.noisy_fake[real_index]
        return [recipe_id, recipe_emb, img_id, img, klass, noisy_real, noisy_fake]
    
    def __len__(self):
        return int(self.split_sizes[self.split_index])

    def num_classes(self):
        return len(self.class_mapping)

    def set_split_index(self, index):
        self.split_index = index

    def get_real_index(self, index):
        real_index = self.split_starts[self.split_index] + index
        if real_index < self.split_starts[self.split_index] or real_index >= self.split_starts[self.split_index + 1]:
            raise ValueError('real_index %d out of bounds for split %d!' % (real_index, self.split_index))
        return real_index

    def get_recipe_split_index(self, recipe_id):
        for i, id_set in enumerate(self.id_sets):
            if recipe_id in id_set:
                return i
        return None

    def get_class_name(self, recipe_id):
        item = self.data[recipe_id]
        klass = item['class']
        real_class = self.inv_class_mapping[klass]
        class_names = util.load_class_names()
        return class_names[real_class]

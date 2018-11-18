import torch.utils.data as data
import util
import numpy as np

class GANstronomyDataset(data.Dataset):
    # data = GANstronomyDataset(DATA_PATH, [0.6, 0.2, 0.2])
    # data.set_split_index(0)
    def __init__(self, data_path, split):
        dataset = util.unpickle(data_path)
        self.data = dataset['data']
        self.class_mapping = dataset['class_mapping']
        self.ids = list(self.data.keys())
        self.split = np.array(split) / np.sum(split)
        self.split_sizes = np.floor(self.split * len(self.ids))
        self.split_index = 0
        self.split_starts = np.insert(np.cumsum(self.split_sizes)[:-1], 0, [0])
        
    def __getitem__(self, index):
        item = self.data[self.ids[self.get_real_index(index)]]
        recipe_id = item['recipe_id']
        img_id = item['img_id']
        klass = item['class']
        recipe_emb = item['recipe_emb']
        img = item['img_pre'] # The preprocessed image
        return [recipe_id, recipe_emb, img_id, img, klass]
    
    def __len__(self):
        return int(self.split_sizes[self.split_index])

    def num_classes(self):
        return len(self.class_mapping)

    def set_split_index(self, index):
        self.split_index = index

    def get_real_index(self, index):
        return int(self.split_starts[self.split_index]) + index

import torch.utils.data as data
import util

class GANstronomyDataset(data.Dataset):
    def __init__(self, data_path):
        dataset = util.unpickle(data_path)
        self.data = dataset['data']
        self.class_mapping = dataset['class_mapping']
        self.ids = list(self.data.keys())

    def __getitem__(self, index):
        item = self.data[self.ids[index]]
        recipe_id = item['recipe_id']
        img_id = item['img_id']
        klass = item['class']
        recipe_emb = item['recipe_emb']
        img = item['img_pre'] # The preprocessed image
        return [recipe_id, recipe_emb, img_id, img, klass]

    def __len__(self):
        return len(self.ids)

    def num_classes(self):
        return len(self.class_mapping)

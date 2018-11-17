import torch.utils.data as data
import util

class GANstronomyDataset(data.Dataset):
    def __init__(self, data_path):
        self.data = util.unpickle(data_path)
        self.ids = list(self.data.keys())

    def __getitem__(self, index):
        item = self.data[self.ids[index]]
        recipe_id = item['recipe_id']
        img_id = item['img_id']
        klass = item['class']
        recipe_emb = item['recipe_emb']
        img = self.normalize_img(item['img_pre']) # A misnomer - should actually just be img_raw
        return [recipe_id, recipe_emb, img_id, img, klass]

    def __len__(self, index):
        return len(self.ids)

    def normalize_img(img): # Squeezes image into [0, 1] range
        assert img.dtype == np.float32
        return img / 255.0

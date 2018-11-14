import util
import os
import numpy as np

# # Each example is (recipe_id, recipe_embedding, img_id, img, klass)
# def build_dataset(recipe_ids, out_path):
#     classes = util.load_classes()
#     lmdb_data = load_lmdb()
#     recipe2img_id = map_recipe_id_to_img_id(lmdb_data)
#     for recipe_id in recipe_ids:
#         recipe_embedding = None
#         img_id = recipe2img_id[recipe_id][-1] # Pick the last one?
#         img = np.array(get_img_path(img_id, RAW_IMG_PATH))
#         klass = classes[recipe_id]

def preprocess_img(img): # TODO: maybe center mean
    return np.array(img, dtype=np.float32)

def save_img(id, img, raw_img_path):
    out_filename = id + '.png'
    img.save(os.path.join(raw_img_path, out_filename), format='PNG')

def gen_dataset(N, data_path, compute_embed=False):
    data_path = os.path.abspath(data_path)
    raw_img_path = os.path.join(data_path, 'imgs')
    temp_path = os.path.join(data_path, 'temp_lmdb')
    util.create_dir(data_path)
    util.create_dir(raw_img_path)
    util.create_dir(temp_path)
    # Data from the entire raw dataset
    print('Loading data...')
    lmdb_data = util.load_lmdb()
    recipe2img_id = util.map_recipe_id_to_img_id(lmdb_data)
    recipe_classes = util.load_classes()
    # Data to go into our custom set
    print('Sampling recipe ids...')
    recipe_ids = util.sample_ids(lmdb_data, N)
    # recipe_ids_decode = [recipe_id.decode('utf-8') for recipe_id in recipe_ids]
    util.repickle(recipe_ids, os.path.join(data_path, 'temp_keys.pkl'))
    # Fill in everything except embeddings
    print('Filling in everything except embeddings...')
    dataset = {}
    for recipe_id in recipe_ids:
        sample = {}
        sample['recipe_id'] = recipe_id
        sample['img_id'] = recipe2img_id[recipe_id][-1]
        sample['class'] = recipe_classes[recipe_id.decode('utf-8')]
        img = util.resize_crop_img(util.get_img_path(sample['img_id'], util.RAW_IMG_PATH))
        save_img(sample['img_id'], img, raw_img_path)
        sample['img_pre'] = preprocess_img(img)
        dataset[recipe_id] = sample
    # Hacky business to compute embeddings
    if not compute_embed:
        print('Skipping embeddings...')
    else:
        print('Computing embeddings...')
        lmdb_slice = util.slice_lmdb(recipe_ids, lmdb_data)
        util.save_lmdb_data(lmdb_slice, temp_path)
        # Need to call their code to read in sliced lmdb, and compute embeddings on recipes
        # Call python3 test.py [model stuff] [data location] [embed location]
        # Say it saves a pickle file of a dict mapping recipe_id -> embedding in temp_path/embed.pkl
        #embeddings = util.unpickle(os.path.join(temp_path, 'embed.pkl'))
        #for recipe_id in dataset:
        #    dataset[recipe_id]['recipe_emb'] = embeddings[recipe_id]
    print('Saving dataset...')
    util.repickle(dataset, os.path.join(data_path, 'data.pkl'))
    print('...done!')

import util
import os
import numpy as np
import subprocess
import sys

def preprocess_img(img):
    return np.array(img, dtype=np.float32) / 255.0

def save_img(id, img, raw_img_path):
    out_filename = id + '.png'
    img.save(os.path.join(raw_img_path, out_filename), format='PNG')

def gen_dataset(N, data_path, compute_embed=True):
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
    recipe_ids_decode = [recipe_id.decode('utf-8') for recipe_id in recipe_ids]
    util.repickle(recipe_ids_decode, os.path.join(data_path, 'temp_keys.pkl'))
    # Fill in everything except embeddings
    print('Filling in everything except embeddings...')
    dataset = {}
    classes = {} # Maps raw class to the smallest number needed
    for recipe_id in recipe_ids:
        sample = {}
        sample['recipe_id'] = recipe_id.decode('utf-8')
        sample['img_id'] = recipe2img_id[recipe_id][-1]
        sample['class_raw'] = recipe_classes[recipe_id.decode('utf-8')]
        if sample['class_raw'] not in classes:
            classes[sample['class_raw']] = len(classes)
        sample['class'] = classes[sample['class_raw']]
        img = util.resize_crop_img(util.get_img_path(sample['img_id'], util.RAW_IMG_PATH))
        save_img(sample['img_id'], img, raw_img_path)
        sample['img_pre'] = preprocess_img(img)
        dataset[recipe_id.decode('utf-8')] = sample
    # Hacky business to compute embeddings
    if not compute_embed:
        print('Skipping embeddings...')
    else:
        print('Computing embeddings...')
        lmdb_slice = util.slice_lmdb(recipe_ids, lmdb_data)
        util.save_lmdb_data(lmdb_slice, temp_path)
        subprocess.run(['python',
                        util.GEN_EMBEDDINGS_ROOT,
                        '--model_path=%s' % util.MODEL_PATH,
                        '--data_path=%s' % data_path,
                        '--path_results=%s' % data_path])
        # Need to call their code to read in sliced lmdb, and compute embeddings on recipes
        # Call python3 test.py [model stuff] [data location] [embed location]
        # Say it saves a pickle file of a dict mapping recipe_id -> embedding in temp_path/embed.pkl
        embeddings = util.unpickle2(os.path.join(data_path, 'rec_embeds.pkl'))
        embedding_ids = util.unpickle2(os.path.join(data_path, 'rec_ids.pkl'))
        for i, embedding_id in enumerate(embedding_ids):
            assert embedding_id in dataset
            dataset[embedding_id]['recipe_emb'] = embeddings[i]
    print('Saving dataset...')
    util.repickle({'data': dataset, 'class_mapping': classes}, os.path.join(data_path, 'data.pkl'))
    print('...done!')

def main():
    if len(sys.argv) < 3:
        print('Usage: python3 gen_dataset.py [N] [output_path]')
        exit()

    gen_dataset(int(sys.argv[1]), sys.argv[2], compute_embed=True)

if __name__ == '__main__':
    main()

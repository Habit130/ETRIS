import argparse
import json
import os
import os.path as osp


def dumps_pyarrow(obj):
    import pyarrow as pa
    return pa.serialize(obj).to_buffer()


def read_bytes(path):
    with open(path, 'rb') as f:
        return f.read()


def remove_existing_lmdb(lmdb_path):
    if osp.isdir(lmdb_path):
        for root, dirs, files in os.walk(lmdb_path, topdown=False):
            for name in files:
                os.remove(osp.join(root, name))
            for name in dirs:
                os.rmdir(osp.join(root, name))
        os.rmdir(lmdb_path)
    elif osp.exists(lmdb_path):
        os.remove(lmdb_path)
    lock_path = lmdb_path + '-lock'
    if osp.exists(lock_path):
        os.remove(lock_path)


def load_items(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_item(item, data_root, caption_index):
    captions = item.get('caption', [])
    if len(captions) <= caption_index:
        raise ValueError(
            f"Item {item.get('id')} does not have caption[{caption_index}]")
    image_path = osp.join(data_root, item['image'])
    mask_path = osp.join(data_root, item['mask'])
    if not osp.exists(image_path):
        raise FileNotFoundError(image_path)
    if not osp.exists(mask_path):
        raise FileNotFoundError(mask_path)
    return image_path, mask_path, captions[caption_index]


def write_split(items, split_name, data_root, output_dir, caption_index,
                overwrite):
    import lmdb

    split_items = [item for item in items if item['split'] == split_name]
    if not split_items:
        raise ValueError(f'No samples found for split: {split_name}')

    os.makedirs(output_dir, exist_ok=True)
    lmdb_path = osp.join(output_dir, f'{split_name}.lmdb')
    if overwrite:
        remove_existing_lmdb(lmdb_path)
    elif osp.exists(lmdb_path) or osp.exists(lmdb_path + '-lock'):
        raise FileExistsError(
            f'{lmdb_path} already exists. Use --overwrite to rebuild it.')

    db = lmdb.open(lmdb_path,
                   subdir=False,
                   map_size=1099511627776,
                   readonly=False,
                   meminit=False,
                   map_async=True)

    txn = db.begin(write=True)
    for idx, item in enumerate(split_items):
        image_path, mask_path, caption = validate_item(item, data_root,
                                                       caption_index)
        record = {
            'img': read_bytes(image_path),
            'mask': read_bytes(mask_path),
            'cat': item.get('disease_label', ''),
            'seg_id': item['id'],
            'img_name': osp.basename(item['image']),
            'num_sents': 1,
            'sents': [caption]
        }
        txn.put(str(idx).encode('ascii'), dumps_pyarrow(record))
        if idx % 1000 == 0 and idx > 0:
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    keys = [str(k).encode('ascii') for k in range(len(split_items))]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    db.sync()
    db.close()
    print(f'Built {split_name}: {len(split_items)} samples -> {lmdb_path}')


def main():
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    parser = argparse.ArgumentParser(
        description='Convert plantseg main.json into ETRIS LMDB splits.')
    parser.add_argument('--data-root',
                        default=osp.join(repo_root, '..', 'plantseg'),
                        type=str,
                        help='Path to the plantseg directory.')
    parser.add_argument('--json-path',
                        default=None,
                        type=str,
                        help='Path to main.json. Defaults to <data-root>/main.json.')
    parser.add_argument('--output-dir',
                        default=None,
                        type=str,
                        help='Output directory for train/val/test LMDB files.')
    parser.add_argument('--caption-index',
                        default=3,
                        type=int,
                        help='Zero-based caption index to use as the only text prompt.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    data_root = osp.abspath(args.data_root)
    json_path = osp.abspath(args.json_path or osp.join(data_root, 'main.json'))
    output_dir = osp.abspath(args.output_dir or osp.join(data_root, 'lmdb'))

    items = load_items(json_path)
    for split_name in ('train', 'val', 'test'):
        write_split(items, split_name, data_root, output_dir,
                    args.caption_index, args.overwrite)


if __name__ == '__main__':
    main()

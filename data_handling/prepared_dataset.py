from tqdm import tqdm
import os
import tensorflow as tf
import random
from typing import List, Dict

random.seed(42)

MIN_IWI = tf.constant([-1.8341599999999998], dtype=tf.float16)
MAX_IWI = tf.constant([86.065248], dtype=tf.float16)
FRAME_PX_DIAMETER = 224  # Each frame has a 224 x 224 px resolution
N_FRAMES_IN_TS_IMG = 10  # A time-series image has ten frames, each corresponding to a three year period
START_YEAR = 1990  # The first frame represents 1990-1992
END_YEAR = 2019  # The last frame represents 2017-2019
SPAN_LENGTH = 3  # Each frame represents a span of three years

####################################
##### Writing prepared dataset #####
####################################

def store_prepared_data(tfrecord_files: List[str], save_dir: str, n_of_frames: int = 10, bands: List[str] = None, 
                       normalize: bool = True, band_stats: Dict = None, shuffle: bool = True, 
                       samples_per_file: int = 100, filename: str = ''):
    if normalize:
        assert band_stats != None, 'Can not normalize without band_stats'

    # Defaults to all bands
    if bands == None:
        bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']

    # Shuffle files to avoid batches containing only samples from a single country
    if shuffle:
        random.shuffle(tfrecord_files)

    # ds = create_img_dataset(tfrecord_files, bands, cache_file, normalize, band_stats)
    # convert to individual records
    ds = tf.data.TFRecordDataset(
        filenames=tfrecord_files,
        compression_type = 'GZIP',
        buffer_size = 1024 * 1024 * 128,  # 128 MB buffer size
        num_parallel_reads = tf.data.AUTOTUNE)

    # prefetch 100 samples at a time to smooth out the time 
    # taken to load input files as we go through processing
    ds = ds.prefetch(buffer_size = 100)

    ds = ds.map(lambda x: proto_to_img(x, bands, normalize, band_stats), 
                          num_parallel_calls=tf.data.AUTOTUNE)

    converters = [get_converter(i, n_of_frames, len(bands)) for i in range(10)]
    map_fn = lambda x: tf.case(
                pred_fn_pairs=[
                    (tf.equal(x['frame_index'], tf.constant(0)), lambda : converters[0](x['img'], x['iwi'])),
                    (tf.equal(x['frame_index'], tf.constant(1)), lambda : converters[1](x['img'], x['iwi'])),
                    (tf.equal(x['frame_index'], tf.constant(2)), lambda : converters[2](x['img'], x['iwi'])),
                    (tf.equal(x['frame_index'], tf.constant(3)), lambda : converters[3](x['img'], x['iwi'])),
                    (tf.equal(x['frame_index'], tf.constant(4)), lambda : converters[4](x['img'], x['iwi'])),
                    (tf.equal(x['frame_index'], tf.constant(5)), lambda : converters[5](x['img'], x['iwi'])),
                    (tf.equal(x['frame_index'], tf.constant(6)), lambda : converters[6](x['img'], x['iwi'])),
                    (tf.equal(x['frame_index'], tf.constant(7)), lambda : converters[7](x['img'], x['iwi'])),
                    (tf.equal(x['frame_index'], tf.constant(8)), lambda : converters[8](x['img'], x['iwi'])),
                    (tf.equal(x['frame_index'], tf.constant(9)), lambda : converters[9](x['img'], x['iwi']))
                ], exclusive=True
            )

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.unbatch()

    # Shuffle dataset to prevent all configurations from a single location ending up in the same batch
    if shuffle:
        ds = ds.shuffle(500)

    # Save dataset as tfrecords
    n_samples, file_count = save_dataset(ds, n_of_frames, save_dir, filename, samples_per_file)

    print(f'Created {n_samples} samples from {len(tfrecord_files)} clusters and saved them to {file_count} TFRecord files')


def proto_to_img(x, bands, normalize, band_stats):
    """
    Converts a proto from a tfrecord into an image 
    """
    x = parse_single_example(x, labeled=True, bands=bands)
    year = tf.cast(x.get('year', -1), tf.int32)
    frame_index = get_frame_index_for_year(year)
    iwi = tf.cast(x.get('iwi', -1), tf.float16)
    normalized_iwi = (iwi - MIN_IWI) / (MAX_IWI - MIN_IWI)
    img = single_example_to_image(x, bands, normalize, band_stats)
    img = tf.cast(img, tf.float16)
    return {'img': img, 'frame_index': frame_index, 'iwi': normalized_iwi}


def parse_single_example(example_proto, labeled, bands):
    """
    Parses a single tf-record file to readable dict with Tensor band values.
    :param example_proto: tf.train.Example, the tf-record example to parse
    :param bands: list, a list containg names for all the bands to use in image
    :param labeled: bool, should parsed example include "IWI" label?
    :param n_year_composites: int, number of image frames in tf-record
    :return: A dict mapping bands, year and possibly label to Tensors values
    """
    keys_to_features = {}
    for band in bands:
        keys_to_features[band] = tf.io.FixedLenFeature(shape=[FRAME_PX_DIAMETER * FRAME_PX_DIAMETER * N_FRAMES_IN_TS_IMG], dtype=tf.float32)
    scalar_float_keys = ['year', 'iwi'] if labeled else ['year']
    for key in scalar_float_keys:
        keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
    return tf.io.parse_single_example(example_proto, features=keys_to_features)


def single_example_to_image(ex, bands, normalize, band_stats):
    """
    Converts a dict of Tensor band values to a correctly formated Tensor time-series image
    :param x: dict, a dict mapping bands to Tensors values
    :param bands: list, a list containg names for all the bands to use in image
    :param normalize: bool, wheter to normalize the band values or not
    :param band_stats: dict, constants (mean and var) used to normalize band values
    :return: A Tensor representation of the time-series image with the 
        shape (N_FRAMES_IN_TS_IMG, FRAME_PX_DIAMETER, FRAME_PX_DIAMETER, len(bands))
    """
    assert len(bands) > 0, 'Can\'t parse image since no bands where submitted'
    
    img = float('nan')
    for band in bands:
        ex[band] = tf.nn.relu(ex[band])
        ex[band] = tf.reshape(ex[band], (N_FRAMES_IN_TS_IMG, FRAME_PX_DIAMETER, FRAME_PX_DIAMETER))
    if normalize:
        means = band_stats['means']
        std_devs = band_stats['stds']
        for band in bands[:-1]:
            ex[band] = (ex[band] - means[band]) / std_devs[band]
        nl_band = 'NIGHTLIGHTS'
        ex[band] = tf.reshape(ex[nl_band], (N_FRAMES_IN_TS_IMG, FRAME_PX_DIAMETER, FRAME_PX_DIAMETER))
        nl_split = tf.split(ex[band], [8, 2], axis=0)

        split1 = (nl_split[0] - means['DMSP']) / std_devs['DMSP']
        split2 = (nl_split[1] - means['VIIRS']) / std_devs['VIIRS']

        ex[nl_band] = tf.concat([split1, split2], axis=0)

    img = tf.stack([ex[band] for band in bands], axis=3)
    return img


def get_converter(frame_index, n_of_frames, num_bands=8):
    # Get possible start indices in a non-tf format for the given index
    min_possible_i = max(0, frame_index - n_of_frames + 1)
    max_possible_i = min(10 - n_of_frames, frame_index)
    possible_start_indices = list(range(min_possible_i, max_possible_i + 1))

    # Create output masks for all configurations. Note that the masks will be identical 
    # for all samples with the same frame_index
    mask_pos = [frame_index - start_index for start_index in possible_start_indices]
    masks = tf.one_hot(mask_pos, n_of_frames, dtype=tf.float16)

    # We weight samples for clusters with multiple configurations lower, since we 
    # want each cluster to carry the same weight
    n_samples = len(possible_start_indices)
    sample_weights = tf.constant(1 / n_samples, shape=(n_samples), dtype=tf.float16)

    # Get the indices for where to split the (10, 224, 224, num_bands) image such that 
    # we get n_samples of new images with the shape (n_of_frames, 224, 224, num_bands)
    img_split_is = [(i, n_of_frames, 10 - (i + n_of_frames)) for i in possible_start_indices]

    # All values so far will be the same for all samples with the same frame_index. To 
    # convert the rest of the variables, we declare a converter function that will be 
    # maped over all image-iwi pairs with this frame_index
    def converter(img, iwi):
        # Ensure the image has the correct shape
        img = tf.ensure_shape(img, (10, 224, 224, num_bands))
        # Split the image into the configurations as specified by 'img_split_is'
        img_configs = [tf.split(img, img_split_i, axis=0)[1] for img_split_i in img_split_is]

        # Stack the new images
        inputs = tf.stack(img_configs)

        # Get the one-hot encoded labels
        labels = iwi * masks
        return {'input': inputs, 'mask': masks, 'label': labels, 'sample_weight': sample_weights}
    
    return converter


def save_dataset(ds, n_of_frames, save_dir, filename, samples_per_file):
    shard_count = 1
    file_name_base = os.path.join(save_dir, '{:04d}_{}.tfrecord')
    current_shard_name = file_name_base.format(shard_count, filename)
    writer = tf.io.TFRecordWriter(current_shard_name)
    file_is_empty = True

    pbar = tqdm()
    n_samples = 0

    for i, sample in ds.enumerate():
        current_shard_count = (i % samples_per_file) + 1

        # Write sample to tfrecord
        out = parse_single_sample(sample, n_of_frames)
        writer.write(out.SerializeToString())
        file_is_empty = False
        n_samples += 1

        # If current shard is full
        if current_shard_count == samples_per_file:

            # Save current shard file and start with a new one
            writer.close()
            pbar.update()
            shard_count += 1
            current_shard_name = file_name_base.format(shard_count, filename)
            writer = tf.io.TFRecordWriter(current_shard_name)
            file_is_empty = True
    
    # Save final shard
    writer.close()
    pbar.close()

    # In case the last file is empty, delete it
    if file_is_empty:
        os.remove(current_shard_name)

    return n_samples, shard_count


def parse_single_sample(sample, n_of_frames):

    data = {
        'input': _bytes_feature(tf.io.serialize_tensor(sample['input'])),
        'sample_weight': _bytes_feature(tf.io.serialize_tensor(sample['sample_weight'])),
        'label': _bytes_feature(tf.io.serialize_tensor(sample['label']))
    }

    if n_of_frames > 1:
        data['mask'] = _bytes_feature(tf.io.serialize_tensor(sample['mask']))
        
    #create an Example, wrapping the single features
    return tf.train.Example(features=tf.train.Features(feature=data))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

####################################
##### Reading prepared dataset #####
####################################

def read_tfrecord_ds(tfrecord_files: List[str], is_single_frame: bool = False, batch_size: int = 32, 
                     max_epochs: int = 200, shuffle: bool = True):

    # Shuffle files to avoid batches containing only samples from a single country
    if shuffle:
        random.shuffle(tfrecord_files)

    # convert to individual records
    ds = tf.data.TFRecordDataset(
        filenames=tfrecord_files,
        buffer_size = 1024 * 1024 * 128,  # 128 MB buffer size
        num_parallel_reads = tf.data.AUTOTUNE)

    ds.prefetch(tf.data.AUTOTUNE)

    ds = ds.map(lambda x: parse_tfrecord(x, is_single_frame), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048)

    ds = ds.batch(batch_size)

    ds = ds.repeat(max_epochs)

    ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


def parse_tfrecord(x, is_single_frame):
    keys_to_features = {
        'input': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'sample_weight': tf.io.FixedLenFeature([], tf.string)
    }

    if not is_single_frame:
        keys_to_features['mask'] = tf.io.FixedLenFeature([], tf.string)
    
    x = tf.io.parse_single_example(x, features=keys_to_features)

    inp = tf.io.parse_tensor(x['input'], out_type=tf.float16)
    label = tf.io.parse_tensor(x['label'], out_type=tf.float16)
    sample_weight = tf.io.parse_tensor(x['sample_weight'], out_type=tf.float16)

    if not is_single_frame:
        mask = tf.io.parse_tensor(x['mask'], out_type=tf.float16)
        mask = tf.reshape(mask, (5, 1)) ########################################## <- Here's tha problam!
        return {'model_input': inp, 'outputs_mask': mask}, {'masked_outputs': label}, sample_weight
        # return {'input': inp, 'mask': mask}, label, sample_weight
    else:
        return inp, label, sample_weight


def get_frame_index_for_year(year):
    """
    Finds the frame index (0-9) which corresponds to the given year.
    """
    # assert START_YEAR <= year and year <= END_YEAR, f'The year ({year} is not in the range ({START_YEAR}-{END_YEAR})'
    index = (year - START_YEAR) // SPAN_LENGTH
    return index

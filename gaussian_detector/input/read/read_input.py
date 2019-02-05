import tensorflow as tf

from typing import List, Callable


def read_input(train_files: List[str], batch_size: int, epochs: int, map_func: Callable[[str], tf.Tensor] = None,
               map_func_parallel_calls=4, shuffle=False, prefetch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices(train_files)
    # skip header line
    dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(1))
    if map_func:
        dataset = dataset.map(map_func, num_parallel_calls=map_func_parallel_calls)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    if shuffle:
        dataset = dataset.shuffle(batch_size)
    dataset = dataset.prefetch(prefetch_size)
    iterator = dataset.make_initializable_iterator()

    return iterator

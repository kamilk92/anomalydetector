import tensorflow as tf


def slice_line_and_transform_to_float(line, slice_cols: int):
    line_splitted = tf.string_split([line], ",")
    str_data = tf.convert_to_tensor(line_splitted.values, dtype=tf.string)
    str_data = tf.slice(str_data, [0], slice_cols)
    float_data = tf.string_to_number(str_data, out_type=tf.float32)

    return float_data

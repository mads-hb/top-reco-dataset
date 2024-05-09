#!/usr/bin/env python3

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import h5py
import hdf5plugin
from collections.abc import Mapping, MutableMapping
import json


class CompressionWrapper:
    def __init__(self, group, compression, compression_opts=None):
        self.group = group
        self.compression = compression
        self.compression_opts = compression_opts

    def __setitem__(self, name, obj):
        if np.asarray(obj).nbytes < 16:
            # If the object size is very small, compression won't have much
            # effect. In fact the Blosc implementation raises warnings for
            # sizes < 16 bytes. Skip compression in this case
            self.group[name] = obj
            return
        ds = self.group.create_dataset(None, data=obj,
                                       compression=self.compression,
                                       compression_opts=self.compression_opts)
        self.group[name] = ds


@ak.mixin_class(ak.behavior)
class Dataframe:
    @ak.mixin_class_method(np.ufunc)
    def ufunc(ufunc, method, args, kwargs):  # noqa: N805
        fields = set(ak.fields(args[0]))
        for i in range(1, len(args)):
            if np.isscalar(args[i]):
                args[i] = ak.full_like(args[0], args[i])
            fields &= set(ak.fields(args[i]))
        out = {}
        func = getattr(ufunc, method)
        for field in fields:
            out[field] = func(*[arg[field] for arg in args], **kwargs)
        return ak.Array(out, with_name="Dataframe")

    def _runak(self, func, *args, **kwargs):
        out = {}
        for field in ak.fields(self):
            out[field] = [func(self[field], *args, **kwargs)]
        return ak.Array(out, with_name="Dataframe")

    def mean(self, *args, **kwargs):
        return self._runak(ak.mean, *args, **kwargs)

    def std(self, *args, **kwargs):
        return self._runak(ak.std, *args, **kwargs)


class HDF5File(MutableMapping):
    def __init__(self, file, mode=None, compression=hdf5plugin.Blosc(),
                 packed=True):
        """Create or open an HDF5 file storing awkward arrays

        Arguments:
        file -- Filename as string or Python file object or h5py file object
        mode -- Determines whether to read ('r') or to write ('w') in case
                `file` is a string
        compression -- If the file is opened for writing, determines the
                       compression used for writing. If None, compression
                       is disabled. Either the value for `compression` in
                       `h5py.Group.create_dataset` or a mapping with keys
                       `compression` and `compression_opts`. See
                       `h5py.Group.create_dataset` for details.
        packed -- Minimize size that awkward arrays will take using ak.packed.
        """
        if isinstance(file, str):
            if mode is None:
                raise ValueError("If file is a str, mode can not be None")
            file = h5py.File(file, mode)
            self._should_close = True
        else:
            self._should_close = False
        self._file = file
        self.compression = compression
        self.packed = packed

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            try:
                value = ak.Array({k: [v] for k, v in value.items()})
            except RuntimeError:
                raise ValueError("Only dicts with string keys and simple "
                                 "values are supported")
            convertto = "dict"
        elif isinstance(value, list):
            value = ak.Array(value)
            convertto = "list"
        elif isinstance(value, str):
            value = ak.Array([value])
            convertto = "str"
        elif isinstance(value, tuple):
            value = ak.Array(value)
            convertto = "tuple"
        elif isinstance(value, np.ndarray):
            value = ak.Array(value)
            convertto = "numpy"
        elif not isinstance(value, ak.Array):
            raise ValueError(f"Invalid type for writing to HDF5: {value}")
        else:
            convertto = "None"
        group = self._file.create_group(key)
        if self.compression is not None:
            if isinstance(self.compression, Mapping):
                container = CompressionWrapper(group, **self.compression)
            else:
                container = CompressionWrapper(group, self.compression)
        else:
            container = group
        if self.packed:
            value = ak.packed(value)
        form, length, container = ak.to_buffers(value, container=container)
        group.attrs["form"] = form.tojson()
        group.attrs["length"] = json.dumps(length)
        group.attrs["parameters"] = json.dumps(ak.parameters(value))
        group.attrs["convertto"] = convertto
        self._file.attrs["version"] = 1

    def __getitem__(self, key):
        group = self._file[key]
        form = ak.forms.Form.fromjson(group.attrs["form"])
        length = json.loads(group.attrs["length"])
        parameters = json.loads(group.attrs["parameters"])
        convertto = group.attrs["convertto"]
        data = {k: np.asarray(v) for k, v in group.items()}
        value = ak.from_buffers(form, length, data)
        for parameter, param_value in parameters.items():
            value = ak.with_parameter(value, parameter, param_value)
        if convertto == "numpy":
            value = np.asarray(value)
        elif convertto == "str":
            value = value[0]
        elif convertto == "tuple":
            value = tuple(value.tolist())
        elif convertto == "list":
            value = value.tolist()
        elif convertto == "dict":
            value = {field: value[field][0] for field in ak.fields(value)}
        return value

    def __delitem__(self, key):
        del self._file[key]

    def __len__(self, key):
        return len(self._file)

    def __iter__(self):
        for key in self._file.keys():
            yield self[key]

    def __repr__(self):
        return f"<AkHdf5 ({self._file.filename})>"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close:
            self._file.close()
        return False

    def keys(self):
        return self._file.keys()

    def close(self):
        self._file.close()


def feat(particle, coord="both"):
    if coord == "both":
        return [f"{particle}_{x}" for x in
                ["x", "y", "z", "t", "pt", "phi", "eta", "mass"]]
    elif coord == "pt":
        return [f"{particle}_{x}" for x in
                ["pt", "phi", "eta", "mass"]]


def table_to_numpy(table, fields):
    cols = []
    # Not using ak.fields here to enforce a specific order
    for field in fields:
        col = table[field]
        if col.ndim == 2:
            col = ak.pad_none(col, ak.max(ak.num(col)))
        col = ak.fill_none(col, 0.)
        cols.append(np.asarray(col))
    return np.stack(cols, axis=-1)


def plot_loss_history(history, filename):
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Val. loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.margins(x=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def load_normalization(path):
    with HDF5File(path, "r") as f:
        return ak.to_list(f["offset"])[0], ak.to_list(f["scale"])[0]


def accuracy_max(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.reduce_all(tf.cast(tf.reduce_max(y_pred, axis=-1, keepdims=True) == y_pred, float) == y_true, axis=-1), float))


class TitledModel(keras.models.Model):
    def __init__(self, input_titles=None, output_titles=None, **kwargs):
        super().__init__(**kwargs)
        self.input_titles = input_titles
        self.output_titles = output_titles
        if hasattr(self, "_layers"):
            try:
                self._layers.remove(self.input_titles)
                self._layers.remove(self.output_titles)
            except ValueError:
                pass

    def get_config(self):
        config = {
            "input_titles": self.input_titles,
            "output_titles": self.output_titles}
        config.update(super().get_config())
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, **kwargs):
        obj = super().from_config(config, custom_objects)
        obj.input_titles = config.get("input_titles")
        obj.output_titles = config.get("output_titles")
        if hasattr(cls, "_layers"):
            try:
                cls._layers.remove(cls.input_titles)
                cls._layers.remove(cls.output_titles)
            except ValueError:
                pass
        return obj

    def get_output_by_name(self, names):
        if names == self.output_titles:
            return self.output
        for name in names:
            idx = self.output_titles.index(name)
            output = self.output[:, idx]
        return tf.stack(output, axis=1)


class LayerFullLorentz(keras.layers.Layer):
    def __init__(self, input_parts, output_parts, norm, epsilon=0.001, should_normalize=True, should_denormalize=True, **kwargs):
        super().__init__(**kwargs)
        self.input_parts = input_parts
        self.output_parts = output_parts
        self.norm = norm
        self.epsilon = epsilon
        self.should_normalize = should_normalize
        self.should_denormalize = should_denormalize
        self._cache = {}

    def get_config(self):
        config = {
            "input_parts": self.input_parts,
            "output_parts": self.output_parts,
            "norm": self.norm,
            "epsilon": self.epsilon,
            "should_normalize": self.should_normalize,
            "should_denormalize": self.should_denormalize
        }
        config.update(super().get_config())
        return config

    def normalize(self, names, vec):
        offset = np.array([self.norm[0][n] for n in names])
        scale = np.array([self.norm[1][n] for n in names])
        return (vec - offset) / scale

    def denormalize(self, names, vec):
        offset = np.array([self.norm[0][n] for n in names])
        scale = np.array([self.norm[1][n] for n in names])
        return vec * scale + offset

    def cached(self, key):
        if key not in self._cache:
            self._cache[key] = getattr(self, key)
        return self._cache[key]

    @property
    def x(self):
        return tf.cos(self.cached("phi")) * self.cached("pt")

    @property
    def y(self):
        return tf.sin(self.cached("phi")) * self.cached("pt")

    @property
    def z(**kwargs):
        raise NotImplementedError("z")

    @property
    def t2(self):
        if "t" in self._cache:
            return self.cached("t") ** 2
        return self.cached("mass2") + self.cached("pt2") + self.cached("z") ** 2

    @property
    def t(self):
        return tf.sqrt(self.cached("t2"))

    @property
    def pt2(self):
        if "pt" in self._cache:
            return self.cached("pt") ** 2
        return self.cached("x") ** 2 + self.cached("y") ** 2

    @property
    def pt(self):
        return tf.sqrt(self.cached("pt2"))

    @property
    def eta(self):
        return tf.asinh(self.cached("z") / self.cached("pt"))

    @property
    def phi(self):
        return tf.atan2(self.cached("y"), self.cached("x"))

    @property
    def mass2(self):
        if "mass" in self._cache:
            return self.cached("mass") ** 2
        return self.cached("t2") - self.cached("pt2") - self.cached("z") ** 2

    @property
    def mass(self):
        return tf.sqrt(self.cached("mass2"))

    def call(self, inputs):
        self._cache = {}
        if self.should_denormalize:
            input_denorm = self.denormalize(self.input_parts, inputs)
        else:
            input_denorm = inputs
        for part_name in self.input_parts:
            idx = self.input_parts.index(part_name)
            self._cache[part_name] = input_denorm[:, idx]
        output = []

        for part_name in self.output_parts:
            if part_name in self.input_parts:
                idx = self.input_parts.index(part_name)
                output.append(input_denorm[:, idx])
            else:
                output.append(self.cached(part_name))
        output = tf.stack(output, axis=1)
        if self.should_normalize:
            output = self.normalize(self.output_parts, output)
        return output


class MaxPassLayer(keras.layers.Layer):
    def __init__(self, num_picks, **kwargs):
        super().__init__(**kwargs)
        self.num_picks = num_picks

    def get_config(self):
        config = {
            "num_picks": self.num_picks
        }
        config.update(super().get_config())
        return config

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("MaxPassLayer needs two element list of inputs.")
        probabilities, features = inputs
        num_steps = features.shape[1]  # = num jets
        is_max = tf.reduce_max(probabilities, axis=1, keepdims=True) == probabilities
        is_max = tf.reshape(is_max, (-1,) + (num_steps,) * self.num_picks)
        outputs = []
        for i in range(self.num_picks):
            mask = tf.reduce_any(is_max, axis=-i - 1)
            outputs.append(tf.reduce_sum(features * tf.cast(mask[..., None], float), axis=1))
        return tf.concat(outputs, axis=1)


class SwitchLayer(keras.layers.Layer):
    def __init__(self, num_picks, **kwargs):
        super().__init__(**kwargs)
        self.num_picks = num_picks

    def get_config(self):
        config = {
            "num_picks": self.num_picks
        }
        config.update(super().get_config())
        return config

    def build(self, input_shape):
        self.w = self.add_weight(name="switch", shape=(self.num_picks, 1), trainable=False, initializer=tf.constant_initializer(1.))
        self.built = True

    def call(self, inputs):
        return tf.reduce_sum(tf.reshape(inputs, (-1, self.num_picks, inputs.shape[1] // self.num_picks)) * self.w, axis=1)

    def switch_to_single(self, index):
        if not self.built:
            raise RuntimeError("SwitchLayer needs to be built first")
        val = np.zeros(self.w.shape)
        val[index] = 1
        self.w.assign(val)


class NormalizationLayerMasked(keras.layers.Normalization, keras.layers.Masking):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        outputs = keras.layers.Masking.call(self, inputs)
        mask = outputs._keras_mask
        outputs = keras.layers.Normalization.call(self, outputs)
        outputs = outputs * tf.cast(mask[..., None], tf.float32)
        outputs._keras_mask = mask
        return outputs


def switch_charge_label(label):
    if label.startswith("alep_"):
        return "lep_" + label[len("alep_"):]
    elif label.startswith("lep_"):
        return "alep_" + label[len("lep_"):]
    elif label.startswith("abot_"):
        return "bot_" + label[len("abot_"):]
    elif label.startswith("bot_"):
        return "abot_" + label[len("bot_"):]
    elif label.startswith("wminus_"):
        return "wplus_" + label[len("wminus_"):]
    elif label.startswith("wplus_"):
        return "wminus_" + label[len("wplus_"):]
    elif label.startswith("atop_"):
        return "top_" + label[len("atop_"):]
    elif label.startswith("top_"):
        return "atop_" + label[len("top_"):]
    else:
        return label


custom_objects = {"accuracy_max": accuracy_max, "TitledModel": TitledModel, "LayerFullLorentz": LayerFullLorentz, "MaxPassLayer": MaxPassLayer, "SwitchLayer": SwitchLayer}

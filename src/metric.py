# from https://github.com/bioinf-jku/TTUR

import os
import shutil
import urllib
import pathlib
import warnings
import numpy as np
import gzip, pickle
import tensorflow as tf

from tqdm import tqdm
from PIL import Image
from scipy import linalg
from torchvision.utils import save_image

from training import generate
from params import DATA_PATH


WEIGHTS_PATH = "../input/classify_image_graph_def.pb"

model_params = {
    "Inception": {
        "name": "Inception",
        "imsize": 64,
        "output_layer": "Pretrained_Net/pool_3:0",
        "input_layer": "Pretrained_Net/ExpandDims:0",
        "output_shape": 2048,
        "cosine_distance_eps": 0.1,
    }
}


class KernelEvalException(Exception):
    pass


def create_model_graph(pth):
    with tf.io.gfile.GFile(pth, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="Pretrained_Net")


def _get_model_layer(sess, model_name):
    layername = model_params[model_name]["output_layer"]
    layer = sess.graph.get_tensor_by_name(layername)
    ops = layer.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return layer


def get_activations(images, sess, model_name, batch_size=50, verbose=False):
    inception_layer = _get_model_layer(sess, model_name)
    n_images = images.shape[0]
    if batch_size > n_images:
        print(
            "warning: batch size is bigger than the data size. setting batch size to data size"
        )
        batch_size = n_images
    n_batches = n_images // batch_size + 1
    pred_arr = np.empty((n_images, model_params[model_name]["output_shape"]))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size
        if start + batch_size < n_images:
            end = start + batch_size
        else:
            end = n_images

        batch = images[start:end]
        pred = sess.run(
            inception_layer, {model_params[model_name]["input_layer"]: batch}
        )
        pred_arr[start:end] = pred.reshape(-1, model_params[model_name]["output_shape"])
    if verbose:
        print(" done")
    return pred_arr


def normalize_rows(x: np.ndarray):
    return np.nan_to_num(x / np.linalg.norm(x, ord=2, axis=1, keepdims=True))


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        # covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
    images, sess, model_name, batch_size=50, verbose=False
):
    act = get_activations(images, sess, model_name, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


def _handle_path_memorization(path, sess, model_name, is_checksize, is_check_png):
    path = pathlib.Path(path)
    files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
    imsize = model_params[model_name]["imsize"]

    # In production we don't resize input images. This is just for demo purpose.
    x = np.array(
        [
            np.array(img_read_checks(fn, imsize, is_checksize, imsize, is_check_png))
            for fn in files
        ]
    )
    m, s, features = calculate_activation_statistics(x, sess, model_name)
    del x  # clean up memory
    return m, s, features


def img_read_checks(
    filename, resize_to, is_checksize=False, check_imsize=64, is_check_png=False
):
    im = Image.open(str(filename))
    if is_checksize and im.size != (check_imsize, check_imsize):
        raise KernelEvalException("The images are not of size " + str(check_imsize))

    if is_check_png and im.format != "PNG":
        raise KernelEvalException("Only PNG images should be submitted.")

    if resize_to is None:
        return im
    else:
        return im.resize((resize_to, resize_to), Image.ANTIALIAS)


def calculate_kid_given_paths(paths, model_name, model_path, feature_path=None):
    """ Calculates the KID of two paths. """
    tf.compat.v1.reset_default_graph()
    create_model_graph(str(model_path))
    with  tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        m1, s1, features1 = _handle_path_memorization(
            paths[0], sess, model_name, is_checksize=True, is_check_png=True
        )
        if feature_path is None:
            m2, s2, features2 = _handle_path_memorization(
                paths[1], sess, model_name, is_checksize=False, is_check_png=False
            )
        else:
            with np.load(feature_path) as f:
                m2, s2, features2 = f["m"], f["s"], f["features"]

        return calculate_frechet_distance(m1, s1, m2, s2)


def compute_fid(generated_path, real_path, graph_path, model_params, eps=10e-15):
    return calculate_kid_given_paths([generated_path, real_path], "Inception", graph_path)


def tmp_eval(generator, folder='../output/tmp_images', n_images=10000, im_batch_size=50, latent_dim=128):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)

    for i_b in range(0, n_images, im_batch_size):
        gen_images = generate(generator, n=im_batch_size, latent_dim=latent_dim)
        for i_img in range(gen_images.size(0)):
            save_image(gen_images[i_img, :, :, :], os.path.join(folder, f'img_{i_b+i_img}.png'))
    
    if len(os.listdir(folder)) != n_images:
        print(len(os.listdir(folder)))
        
    fid = compute_fid(folder, DATA_PATH, WEIGHTS_PATH, model_params)
    shutil.rmtree(folder, ignore_errors=True)
    return fid
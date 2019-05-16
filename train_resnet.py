"""
Trains a modified Resnet to generate approximate dlatents using examples from a trained StyleGAN.
Props to @SimJeg on GitHub for the original code this is based on, from this thread: https://github.com/Puzer/stylegan-encoder/issues/1#issuecomment-490469454
"""
import os
import math
import numpy as np
import pickle
import cv2
#from google.colab.patches import cv2_imshow

import dnnlib
import config
import dnnlib.tflib as tflib

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import LocallyConnected1D, Reshape, Permute, Conv2D
from keras.models import Sequential, load_model

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'

tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

def load_Gs():
    tflib.init_tf()
    return Gs_network

def generate_dataset_main(n=10000, save_path=None, seed=None, model_res=1024, image_size=256, minibatch_size=32):
    """
    Generates a dataset of 'n' images of shape ('size', 'size', 3) with random seed 'seed'
    along with their dlatent vectors W of shape ('n', 512)

    These datasets can serve to train an inverse mapping from X to W as well as explore the latent space

    More variation added to latents; also, negative truncation added to balance these examples.
    """

    n = n // 2 # this gets doubled because of negative truncation below
    model_scale = int(2*(math.log(model_res,2)-1)) # For example, 1024 -> 18

    Gs = load_Gs()
    if (model_scale % 3 == 0):
        mod_l = 3
    else:
        mod_l = 2
    if seed is not None:
        b = bool(np.random.RandomState(seed).randint(2))
        Z = np.random.RandomState(seed).randn(n*mod_l, Gs.input_shape[1])
    else:
        b = bool(np.random.randint(2))
        Z = np.random.randn(n*mod_l, Gs.input_shape[1])
    if b:
        mod_l = model_scale // 2
    mod_r = model_scale // mod_l
    if seed is not None:
        Z = np.random.RandomState(seed).randn(n*mod_l, Gs.input_shape[1])
    else:
        Z = np.random.randn(n*mod_l, Gs.input_shape[1])
    W = Gs.components.mapping.run(Z, None, minibatch_size=minibatch_size) # Use mapping network to get unique dlatents for more variation.
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]
    W = (W[np.newaxis] - dlatent_avg) * np.reshape([1, -1], [-1, 1, 1, 1]) + dlatent_avg # truncation trick and add negative image pair
    W = np.append(W[0], W[1], axis=0)
    W = W[:, :mod_r]
    W = W.reshape((n*2, model_scale, 512))
    X = Gs.components.synthesis.run(W, randomize_noise=False, minibatch_size=minibatch_size, print_progress=True,
                                    output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
    X = np.array([cv2.resize(x, (image_size, image_size), interpolation = cv2.INTER_AREA) for x in X])
    X = preprocess_input(X)
    return W, X

def generate_dataset(n=10000, save_path=None, seed=None, model_res=1024, image_size=256, minibatch_size=16):
    """
    Use generate_dataset_main() as a helper function.
    Divides requests into batches to save memory.
    """
    batch_size = 16
    inc = n//batch_size
    left = n-((batch_size-1)*inc)
    W, X = generate_dataset_main(inc, save_path, seed, model_res, image_size, minibatch_size)
    for i in range(batch_size-2):
        aW, aX = generate_dataset_main(inc, save_path, seed, model_res, image_size, minibatch_size)
        W = np.append(W, aW, axis=0)
        aW = None
        X = np.append(X, aX, axis=0)
        aX = None
    aW, aX = generate_dataset_main(left, save_path, seed, model_res, image_size, minibatch_size)
    W = np.append(W, aW, axis=0)
    aW = None
    X = np.append(X, aX, axis=0)
    aX = None

    if save_path is not None:
        prefix = '_{}_{}'.format(seed, n)
        np.save(os.path.join(os.path.join(save_path, 'W' + prefix)), W)
        np.save(os.path.join(os.path.join(save_path, 'X' + prefix)), X)

    return W, X

def is_square(n):
  return (n == int(math.sqrt(n) + 0.5)**2)
  
def get_resnet_model(save_path, model_res=1024, image_size=256):
    # Build model
    if os.path.exists(save_path):
        print('Loading existing model')
        model = load_model(save_path)
    else:
        print('Building model')
        model_scale = int(2*(math.log(model_res,2)-1)) # For example, 1024 -> 18
        resnet = ResNet50(include_top=False, pooling=None, weights='imagenet', input_shape=(image_size, image_size, 3))
        model = Sequential()
        model.add(resnet)
        model.add(Conv2D(model_scale*8, 1)) # scale down to correct # of parameters
        layer_size = model_scale*8*8*8
        if is_square(layer_size): # work out layer dimensions
          layer_l = int(math.sqrt(layer_size)+0.5)
          layer_r = layer_l
        else:
          layer_m = math.log(math.sqrt(layer_size),2)
          layer_l = 2**math.ceil(layer_m)
          layer_r = layer_size // layer_l
        layer_l = int(layer_l)
        layer_r = int(layer_r)
        model.add(Reshape((layer_l, layer_r))) # See https://github.com/OliverRichter/TreeConnect/blob/master/cifar.py - TreeConnect inspired layers instead of dense layers.
        model.add(LocallyConnected1D(layer_r, 1, activation='elu'))
        model.add(Permute((2, 1)))
        model.add(LocallyConnected1D(layer_l, 1, activation='elu'))
        model.add(Permute((2, 1)))
        model.add(LocallyConnected1D(layer_r, 1, activation='elu'))
        model.add(Permute((2, 1)))
        model.add(LocallyConnected1D(layer_l, 1, activation='elu'))
        model.add(Reshape((model_scale, 512))) # train against all dlatent values

    model.compile(loss='logcosh', metrics=[], optimizer='adam') # Adam optimizer, logcosh used for loss.
    model.summary()
    return model

def finetune_resnet(model, save_path, model_res=1024, image_size=256, batch_size=10000, test_size=1000, n_epochs=10, max_patience=5, seed=0):
    """
    Finetunes a resnet to predict W from X
    Generate batches (X, W) of size 'batch_size', iterates 'n_epochs', and repeat while 'max_patience' is reached
    on the test set. The model is saved every time a new best test loss is reached.
    """
    assert image_size >= 224

    # Create a test set
    print('Creating test set:')
    np.random.seed(seed)
    W_test, X_test = generate_dataset(n=test_size, model_res=model_res, image_size=image_size, seed=seed)

    # Iterate on batches of size batch_size
    print('Generating training set:')
    patience = 0
    best_loss = np.inf
    #loss = model.evaluate(X_test, W_test)
    #print('Initial test loss : {:.5f}'.format(loss))
    while (patience <= max_patience):
        W_train = X_train = None
        W_train, X_train = generate_dataset(batch_size, model_res=model_res, image_size=image_size, seed=seed)
        model.fit(X_train, W_train, epochs=n_epochs, batch_size=16, verbose=True)
        loss = model.evaluate(X_test, W_test)
        if loss < best_loss:
            print('New best test loss : {:.5f}'.format(loss))
            patience = 0
            best_loss = loss
        else:
            print('Test loss : {:.5f}'.format(loss))
            patience += 1
        if (patience > max_patience): # When done with test set, train with it and discard.
            print('Done with current test set.')
            model.fit(X_test, W_test, epochs=n_epochs, verbose=True)
        model.save(save_path)

model = get_resnet_model('data/finetuned_resnet.h5', model_res=1024)

while True:
    finetune_resnet(model, 'data/finetuned_resnet.h5', model_res=1024, image_size=256, batch_size=2048, test_size=512, max_patience=2, n_epochs=2, seed=None)

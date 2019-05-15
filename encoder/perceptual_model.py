import PIL.Image
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
import keras.backend as K

def load_images(images_list, image_size=256):
    loaded_images = list()
    for img_path in images_list:
      img = PIL.Image.open(img_path).convert('RGB').resize((image_size,image_size),PIL.Image.LANCZOS)
      img = np.array(img)
      img = np.expand_dims(img, 0)
      loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    return loaded_images

def tf_custom_l1_loss(img1,img2):
  return tf.math.reduce_mean(tf.math.abs(img2-img1), axis=None)

def tf_custom_logcosh_loss(img1,img2):
  return tf.math.reduce_mean(tf.keras.losses.logcosh(img1,img2))

# This is the perceptual model included with StyleGAN; remove to have one less dependency.
#with dnnlib.util.open_url('https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2', cache_dir=config.cache_dir) as f:
#  perc_model =  pickle.load(f)
#def compare_images(img1,img2):
#  return perc_model.get_output_for(tf.transpose(img1, perm=[0,3,2,1]), tf.transpose(img2, perm=[0,3,2,1]))

class PerceptualModel:
    def __init__(self, img_size, layer=9, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size
        self.ref_img = None
        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None

    def build_perceptual_model(self, generator):
        generated_image_tensor = generator.generated_image
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
        generated_image = tf.image.resize_nearest_neighbor(generated_image_tensor,
                                                                  (self.img_size, self.img_size), align_corners=True)
        generated_img_features = self.perceptual_model(generated_image)
        self.ref_img = tf.get_variable('ref_img', shape=generated_image.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.sess.run([self.features_weight.initializer, self.features_weight.initializer])

        # L1 loss on VGG16 features
        self.loss = tf_custom_l1_loss(self.features_weight * self.ref_img_features, self.features_weight * generated_img_features) * 1.5
        # + logcosh loss on image pixels
        self.loss += tf_custom_logcosh_loss(self.ref_img,generated_image)
        # + MS-SIM loss on image pixels
        self.loss += tf.math.reduce_mean(1-tf.image.ssim_multiscale(self.ref_img,generated_image,1)) * 75
        # + extra perceptual loss on image pixels
        #self.loss += compare_images(self.ref_img, generated_image)*50
        # + L1 penalty on dlatent weights
        self.loss += tf.math.reduce_sum(tf.math.abs(generator.dlatent_variable))/15

    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        image_features = self.perceptual_model.predict_on_batch(preprocess_input(loaded_image))

        # in case if number of images less than actual batch size
        # can be optimized further
        weight_mask = np.ones(self.features_weight.shape)
        if len(images_list) != self.batch_size:
            features_space = list(self.features_weight.shape[1:])
            existing_features_shape = [len(images_list)] + features_space
            empty_features_shape = [self.batch_size - len(images_list)] + features_space

            existing_examples = np.ones(shape=existing_features_shape)
            empty_examples = np.zeros(shape=empty_features_shape)
            weight_mask = np.vstack([existing_examples, empty_examples])

            image_features = np.vstack([image_features, np.zeros(empty_features_shape)])

        self.sess.run(tf.assign(self.features_weight, weight_mask))
        self.sess.run(tf.assign(self.ref_img_features, image_features))
        self.sess.run(tf.assign(self.ref_img, loaded_image))

    def optimize(self, vars_to_optimize, iterations=200, learning_rate=0.01):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9, beta2=0.999, epsilon=1e-08)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        self.sess.run(tf.variables_initializer(optimizer.variables()))
        for _ in range(iterations):
            _, loss = self.sess.run([min_op, self.loss])
            yield loss
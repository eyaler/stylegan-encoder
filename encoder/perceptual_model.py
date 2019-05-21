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

class PerceptualModel:
    def __init__(self, args, batch_size=1, perc_model=None, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.epsilon = 0.00000001
        self.img_size = args.image_size
        self.layer = args.use_vgg_layer
        self.vgg_loss = args.use_vgg_loss
        if (self.layer <= 0 or self.vgg_loss <= self.epsilon):
            self.vgg_loss = None
        self.pixel_loss = args.use_pixel_loss
        if (self.pixel_loss <= self.epsilon):
            self.pixel_loss = None
        self.mssim_loss = args.use_mssim_loss
        if (self.mssim_loss <= self.epsilon):
            self.mssim_loss = None
        self.lpips_loss = args.use_lpips_loss
        if (self.lpips_loss <= self.epsilon):
            self.lpips_loss = None
        self.l1_penalty = args.use_l1_penalty
        if (self.l1_penalty <= self.epsilon):
            self.l1_penalty = None
        self.batch_size = batch_size
        if perc_model is not None and self.lpips_loss is not None:
            self.perc_model = perc_model
        else:
            self.perc_model = None
        self.ref_img = None
        self.ref_weight = None
        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None

    def compare_images(self,img1,img2):
        if self.perc_model is not None:
            return self.perc_model.get_output_for(tf.transpose(img1, perm=[0,3,2,1]), tf.transpose(img2, perm=[0,3,2,1]))
        return 0

    def build_perceptual_model(self, generator):
        generated_image_tensor = generator.generated_image
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
        generated_image = tf.image.resize_nearest_neighbor(generated_image_tensor,
                                                                  (self.img_size, self.img_size), align_corners=True)
        generated_img_features = self.perceptual_model(preprocess_input(generated_image))
        self.ref_img = tf.get_variable('ref_img', shape=generated_image.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.ref_weight = tf.get_variable('ref_weight', shape=generated_image.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.sess.run([self.features_weight.initializer, self.features_weight.initializer])

        self.loss = 0
        # L1 loss on VGG16 features
        if (self.vgg_loss is not None):
            self.loss += self.vgg_loss * tf_custom_l1_loss(self.features_weight * self.ref_img_features, self.features_weight * generated_img_features)
        # + logcosh loss on image pixels
        if (self.pixel_loss is not None):
            self.loss += self.pixel_loss * tf_custom_logcosh_loss(self.ref_weight * self.ref_img, self.ref_weight * generated_image)
        # + MS-SIM loss on image pixels
        if (self.mssim_loss is not None):
            self.loss += self.mssim_loss * tf.math.reduce_mean(1-tf.image.ssim_multiscale(self.ref_weight * self.ref_img, self.ref_weight * generated_image, 1))
        # + extra perceptual loss on image pixels
        if self.perc_model is not None and self.lpips_loss is not None:
            self.loss += self.lpips_loss * tf.math.reduce_mean(self.compare_images(self.ref_weight * self.ref_img, self.ref_weight * generated_image))
        # + L1 penalty on dlatent weights
        if self.l1_penalty is not None:
            self.loss += self.l1_penalty * 512 * tf.math.reduce_mean(tf.math.abs(generator.dlatent_variable-tf.math.reduce_mean(generator.dlatent_variable)))

    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        image_features = self.perceptual_model.predict_on_batch(preprocess_input(loaded_image))
        #image_features = self.perceptual_model.predict_on_batch(loaded_image)

        # in case if number of images less than actual batch size
        # can be optimized further
        weight_mask = np.ones(self.features_weight.shape)
        image_mask = np.ones(self.ref_weight.shape)
        if len(images_list) != self.batch_size:
            features_space = list(self.features_weight.shape[1:])
            existing_features_shape = [len(images_list)] + features_space
            empty_features_shape = [self.batch_size - len(images_list)] + features_space

            images_space = list(self.ref_weight.shape[1:])
            existing_images_space = [len(images_list)] + images_space
            empty_images_space = [self.batch_size - len(images_list)] + images_space

            existing_examples = np.ones(shape=existing_features_shape)
            empty_examples = np.zeros(shape=empty_features_shape)
            weight_mask = np.vstack([existing_examples, empty_examples])

            existing_images = np.ones(shape=existing_images_space)
            empty_images = np.zeros(shape=empty_images_space)
            image_mask = np.vstack([existing_images, empty_images])

            image_features = np.vstack([image_features, np.zeros(empty_features_shape)])
            loaded_image = np.vstack([loaded_image, np.zeros(empty_images_space)])

        self.sess.run(tf.assign(self.features_weight, weight_mask))
        self.sess.run(tf.assign(self.ref_img_features, image_features))
        self.sess.run(tf.assign(self.ref_weight, image_mask))
        self.sess.run(tf.assign(self.ref_img, loaded_image))

    def optimize(self, vars_to_optimize, iterations=200, learning_rate=0.01):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9, beta2=0.999, epsilon=1e-08)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        self.sess.run(tf.variables_initializer(optimizer.variables()))
        for _ in range(iterations):
            _, loss = self.sess.run([min_op, self.loss])
            yield loss

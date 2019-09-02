#embed and check
#embed non-face and check
#create n images by intepolating m verctors with uniform wwights but non higher than x

import math
import pickle
import PIL.Image
import config
import dnnlib
import dnnlib.tflib as tflib
from encoder.generator_model import Generator
import numpy as np
import os
import cv2

steps = 300
spherical = True

interpolated_images_dir='inter2_images'
dlatent_dir='latent_representations_old'
latent1 = 'faiglin1_01.npy'
latent2 = 'bibi1_01.npy'

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

def interp(steps, low, high, spherical=False):
    vals = [step/(steps+1) for step in range(steps+2)]
    ret = np.asarray([(1.0 - val) * low + val * high for val in vals])  # L'Hopital's rule/LERP
    if not spherical:
        return ret
    omega = np.arccos(np.clip(np.sum(low/np.linalg.norm(low,axis=-1,keepdims=True)*high/np.linalg.norm(high,axis=-1,keepdims=True),axis=-1,keepdims=True), -1, 1))
    so = np.sin(omega)
    ind = so[:,0]!=0
    ret[:,ind] = np.asarray([np.sin((1.0-val)*omega[ind]) / so[ind] * low[ind] + np.sin(val*omega[ind]) / so[ind] * high[ind] for val in vals])
    return ret


generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

model_res = 1024
model_scale = int(2*(math.log(model_res,2)-1))

def generate_raw_image(latent_vector):
    latent_vector = latent_vector.reshape((1, model_scale, 512))
    generator.set_dlatents(latent_vector)
    return generator.generate_images()[0]

def generate_image(latent_vector):
    img_array = generate_raw_image(latent_vector)
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img

latent1 = np.load(os.path.join(dlatent_dir,latent1))
latent2 = np.load(os.path.join(dlatent_dir,latent2))

video_out = cv2.VideoWriter('inter2.avi',cv2.VideoWriter_fourcc(*'MJPG'), 30, (model_res,model_res))

vectors = interp(steps, latent1, latent2, spherical=spherical)
vectors = np.vstack([vectors, vectors[-2:0:-1]])
for i,vector in enumerate(vectors):
    print('%d/%d'%(i+1,len(vectors)))
    np.expand_dims(vector,axis=0)
    img = generate_image(vector)
    #img.save(os.path.join(interpolated_images_dir,str(i)+'.png'), 'PNG')
    video_out.write(cv2.cvtColor(np.array(img).astype('uint8'), cv2.COLOR_RGB2BGR))

video_out.release()


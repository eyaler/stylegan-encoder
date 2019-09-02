import math
import pickle
import PIL.Image
import config
import dnnlib
import dnnlib.tflib as tflib
from encoder.generator_model import Generator
import numpy as np
import os

fix_norm = False
num_outputs = 100
min_comps = 3
max_comps = 5
min_weight = 0.1
max_weight = 0.6

interpolated_images_dir='interpolated_images'
dlatent_dir='latent_representations_old'

np.random.seed(42)

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

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

latent_vectors = np.asarray([np.load(os.path.join(dlatent_dir,file)) for file in os.listdir(dlatent_dir)])

for i in range(num_outputs):
    comps = np.random.randint(max(2,min_comps),1+min(max_comps,len(latent_vectors)))
    inds = np.random.choice(len(latent_vectors), comps, replace=False)
    have_weights = False
    while not have_weights:
        weights = np.random.uniform(0,1,comps)
        weights /= np.sum(weights)
        if (not max_weight or np.max(weights)<=max_weight) and (not min_weight or np.min(weights)>=min_weight):
            have_weights = True
    inter = np.sum(weights[:,np.newaxis,np.newaxis]*latent_vectors[inds],axis=0)
    if fix_norm:
        norm = np.sum(weights[:,np.newaxis,np.newaxis]*np.linalg.norm(latent_vectors[inds],axis=-1,keepdims=True),axis=0)
        inter *= norm/np.linalg.norm(inter,axis=-1,keepdims=True)
    print('%d/%d %.2f %.2f' % (i + 1, num_outputs, np.mean(np.linalg.norm(inter, axis=-1, keepdims=True)), np.max(np.abs(inter))),
          sorted(zip(np.round(weights, 2), inds), key=lambda x: -x[0]))

    img = generate_image(np.expand_dims(inter,axis=0))
    img.save(os.path.join(interpolated_images_dir,str(i+1)+'.png'), 'PNG')

'''
# more stupid latent tricks
lr = ((np.arange(1,model_scale+1)/model_scale)**0.75).reshape((model_scale,1))
rl = 1-lr
display(generate_image(lr*s1+rl*s2).resize((256,256),PIL.Image.LANCZOS))

lr = ((np.arange(1,model_scale+1)/model_scale)**0.25).reshape((model_scale,1))
rl = 1-lr
display(generate_image(lr*s2+rl*s1).resize((256,256),PIL.Image.LANCZOS))

display(generate_image(-0.5*s1).resize((256,256),PIL.Image.LANCZOS))
display(generate_image(-0.5*s1+(lr*s2+rl*s1)).resize((256,256),PIL.Image.LANCZOS))
'''
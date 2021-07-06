# -*- coding: utf-8 -*-
"""
Created on Fri May 22 19:00:22 2020

@author: Cansel
"""
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from matplotlib import pyplot
import time

start_time = time.time()

def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# 
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# 
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# 
	model = Model([in_src_image, in_target_image], patch_out)
	# 
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])  
	model.summary()
	return model
  

def define_generator(image_shape=(256,256,3)):
	# başlangıç ağırlıkları
   init = RandomNormal(stddev=0.02)
	# giriş imgesi
   in_image = Input(shape=image_shape)
    
   e1 = Conv2D(64,(4,4),strides=(2,2), padding='same',kernel_initializer=init)(in_image)
   e1 = LeakyReLU(alpha=0.2)(e1)
    
   e2 = Conv2D(64,(3,3), strides=(1,1), padding='same', kernel_initializer=init)(e1)
   e2 = BatchNormalization()(e2, training=True)
   e2 = LeakyReLU(alpha=0.2)(e2)
    
   e3 = Conv2D(64,(3,3), strides=(1,1), padding='same', kernel_initializer=init)(e2)
   e3 = BatchNormalization()(e3, training=True)
   e3 = LeakyReLU(alpha=0.2)(e3)
    
  # g = Concatenate()([e3, e1])
    
   d = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e3)
    
   out_image = Activation('tanh')(d)
    
    # modeli tanımla
   model = Model(in_image, out_image)
    
   return model



# gan modelinin oluşturulması
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# source imge
	in_src = Input(shape=image_shape)
	# source imgeyi genaretore yolla
	gen_out = g_model(in_src)
	# 
	dis_out = d_model([in_src, gen_out])
	# src imgesi giriş, üretilmiş ve model çıkışı 
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# load and prepare training images
def load_real_samples(filename):
	# dataseti yükle
	data = load(filename)
	# ikiye böl
	X1, X2 = data['arr_0'], data['arr_1']
	# scale et [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]


def generate_real_samples(dataset, n_samples, patch_shape):
	# dataseti ayır
	trainA, trainB = dataset
	# random değer seç
	ix = randint(0, trainA.shape[0], n_samples)
	# görüntüyü ve etiketini ata
	X1, X2 = trainA[ix], trainB[ix]
	# real= 1 olarak etiketle
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# fake (0) etiketi oluştur
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y


def summarize_performance(step, g_model, dataset, n_samples=3):
	# hedef imgeden input al sample 
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# train pix2pix model
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)

# load image data
dataset = load_real_samples('5Kimg_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)
print("--- %s seconds ---" % (time.time() - start_time))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
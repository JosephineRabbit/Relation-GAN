from .frechet_kernel_Inception_distance import *
from .inception_score import *
from glob import glob
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = False

class Inception_Score():
    def __init__(self,batch_size):
        inception_images = tf.placeholder(tf.float32, [batch_size, 3, None, None])
        logits = inception_logits(inception_images)
        self.batch_size = batch_size
        self.model_op = logits
        self.image_op = inception_images

    def get_score(self,images):
        IS = get_inception_score(self.batch_size, images, self.image_op, self.model_op, splits=5)
        return IS

class KID_Score():
    def __init__(self,batch_size):
        inception_images = tf.placeholder(tf.float32, [batch_size, 3, None, None])
        real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
        fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

        # start_time = time.time()
        self.kcd_mean, self.kcd_stddev = kernel_classifier_distance_and_std_from_activations(real_activation, fake_activation,
                                                                                   max_block_size=10)
        activations = inception_activations(inception_images)


        self.batch_size = batch_size
        self.model_op = activations
        self.image_op = real_activation
        self.fake_op = fake_activation
        self.inception_images_op = inception_images

    def get_score(self,real_images,fake_images,act_img=None):
        if act_img is None:
            act_img = get_inception_activations(self.batch_size, real_images, self.inception_images_op, self.model_op)
        act_fake = get_inception_activations(self.batch_size, fake_images, self.inception_images_op, self.model_op)
        KID_mean = get_kid_value(self.kcd_mean, act_img, act_fake, self.image_op, self.fake_op)
        KID_stddev = get_kid_value(self.kcd_stddev, act_img, act_fake, self.image_op, self.fake_op)

        return {"KID_mean": KID_mean * 100, "KID_stddev": KID_stddev * 100, 'act_img': act_img}

def inception_score(images,BATCH_SIZE) :
    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE,3,None, None])
    logits = inception_logits(inception_images)
    IS = get_inception_score(BATCH_SIZE, images, inception_images, logits, splits=5)

    return IS


def kernel_inception_distance(real_images,fake_images,BATCH_SIZE,act_img=None) :

    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    # start_time = time.time()
    kcd_mean, kcd_stddev = kernel_classifier_distance_and_std_from_activations(real_activation, fake_activation, max_block_size=10)
    # print(time.time() - start_time)

    activations = inception_activations(inception_images)
    if act_img is None:
        act_img = get_inception_activations(BATCH_SIZE, real_images, inception_images, activations)
    act_fake = get_inception_activations(BATCH_SIZE, fake_images, inception_images, activations)
    KID_mean = get_kid_value(kcd_mean, act_img,act_fake, real_activation, fake_activation)
    KID_stddev = get_kid_value(kcd_stddev, act_img,act_fake, real_activation, fake_activation)

    return {"KID_mean": KID_mean * 100,"KID_stddev":KID_stddev * 100,'act_img':act_img}

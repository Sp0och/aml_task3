# In[1]:
import matplotlib.pyplot as plt
import numpy as np

# In[1]:

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))

# In[3]:

import pickle
import gzip
import numpy as np
import os
import cv2
import cv2 as cv
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# In[3]:


from tensorflow.keras import layers
from sklearn.model_selection import KFold


from tensorflow import keras
from PIL import Image as im


from skimage.measure import label, regionprops
import random
# installation: pip install elasticdeform
# import elasticdeform


# In[3]:
if gpus:
    gpu_num = 2  # Number of the GPU to be used
    try:
        # tf.config.set_visible_devices([], 'GPU')
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)


# tf.config.get_visible_devices('GPU')
# ### Helper functions

# In[3]:


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


# In[4]:


def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)


# In[5]:


def evaluate(predictions, targets):
    ious = []
    for p, t in zip(predictions, targets):
        assert p['name'] == t['name']
        prediction = np.array(p['prediction'], dtype=bool)
        target = np.array(t['label'], dtype=bool)

        assert target.shape == prediction.shape
        overlap = prediction * target
        union = prediction + target

        ious.append(overlap.sum() / float(union.sum()))

    print("Median IOU: ", np.median(ious))


# In[6]:


def double_conv_block(x, n_filters):
    x = layers.Conv2D(n_filters, 3, padding="same")(x)  # kernel_initializer = "he_normal"
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters, 3, padding="same")(x)  # kernel_initializer = "he_normal"
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


# In[7]:


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    # p = layers.Dropout(0.3)(p)
    return f, p


# In[8]:


def upsample_block(x, conv_features, n_filters):
    x = layers.Conv2DTranspose(n_filters, 2, 2, padding="valid")(x)
    x = layers.concatenate([x, conv_features])
    # x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x


# In[9]:


def get_model(img_size):
    inputs = layers.Input(shape=img_size + (1,))

    f1, p1 = downsample_block(inputs, 16)
    f2, p2 = downsample_block(p1, 32)
    f3, p3 = downsample_block(p2, 64)
    # f4, p4 = downsample_block(p3, 256)

    # bottleneck = double_conv_block(p4, 512)
    bottleneck = double_conv_block(p3, 128)

    # u6 = upsample_block(bottleneck, f4, 256)
    u7 = upsample_block(bottleneck, f3, 64)
    u8 = upsample_block(u7, f2, 32)
    u9 = upsample_block(u8, f1, 16)

    outputs = layers.Conv2D(1, 1, padding="valid", activation="sigmoid")(u9)

    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


# ### Load data, make predictions and save prediction in correct format

# In[10]:
# load data
train_data = load_zipped_pickle("train.pkl")
test_data = load_zipped_pickle("test.pkl")
samples = load_zipped_pickle("sample.pkl")


# # Visualizations
# solution taken from https://stackoverflow.com/questions/43445103/inline-animations-in-jupyter
#
# also see http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/

# In[11]:

#
# from matplotlib import animation
# from IPython.display import HTML
#
#
# # %matplotlib inline
#
# def play_video(img_list):
#     def init():
#         img.set_data(img_list[0])
#         return (img,)
#
#     def animate(i):
#         img.set_data(img_list[i])
#         return (img,)
#
#     fig = plt.figure()
#     ax = fig.gca()
#     img = ax.imshow(img_list[0], cmap='gray', vmin=0, vmax=255)
#     anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                    frames=len(img_list), interval=20, blit=True)
#     return anim


# In[14]:

#
# train_sample_id = 33
# crop = False
#
# video = np.copy(train_data[train_sample_id]['video'])
# labels = train_data[train_sample_id]['label']
# box = train_data[train_sample_id]['box']
# labeled_frames = train_data[train_sample_id]['frames']
#
# # first overlay the box and the labels of mitral valve wherever available
# alpha = 0.2  # alpha blending parameter to overlay the box
#
# for i in range(video.shape[2]):
#     frame = video[:, :, i]
#     # add the labels
#     frame[labels[:, :, i]] = 255
#
# # for i in range(video.shape[2]):
# #     # overlay the box on all images
# #     video[:,:,i] = (1-alpha)*video[:,:,i] + alpha*255*box
#
# print("Showing the training sample " + str(train_sample_id) + ":")
#
# if crop:
#     boxnz = np.nonzero(box)
#     w = boxnz[1][-1] - boxnz[1][0]
#     y0 = boxnz[0][-1] - w
#     y1 = boxnz[0][-1]
#     x0 = boxnz[1][0]
#     x1 = boxnz[1][-1]
# else:
#     y0 = 0
#     y1 = video.shape[0]
#     x0 = 0
#     x1 = video.shape[1]
#
# imgs = [video[y0:y1, x0:x1, i] for i in range(video.shape[2])]
#
# HTML(play_video(imgs).to_jshtml())

# In[12]:


for i in range(len(test_data)):
    print(test_data[i]['video'].shape)

# ## If we want to use 'featurewise_std_normalization' we should use the fit method: (skip for now)

# In[ ]:


# we create two instances with the same arguments
# data_gen_args = dict(featurewise_center=True,
#                      featurewise_std_normalization=True,
#                      rotation_range=90,
#                      width_shift_range=0.1,
#                      height_shift_range=0.1,
#                      zoom_range=0.2)
# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)
# # Provide the same seed and keyword arguments to the fit and flow methods
# seed = 1
# image_datagen.fit(images, augment=True, seed=seed)
# mask_datagen.fit(masks, augment=True, seed=seed)
# image_generator = image_datagen.flow_from_directory(
#     'data/images',
#     class_mode=None,
#     seed=seed)
# mask_generator = mask_datagen.flow_from_directory(
#     'data/masks',
#     class_mode=None,
#     seed=seed)
# # combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)
# model.fit(
#     train_generator,
#     steps_per_epoch=2000,
#     epochs=50)


# # Data augmentation using ImageDataGenerator

# In[13]:
# data_gen_args = dict(featurewise_center=True,
#                      featurewise_std_normalization=True,
def augment_data(image, mask):
    seed = 1
    data_gen_args = dict(horizontal_flip=True,
                         rotation_range=10,
                         shear_range=10,
                         zoom_range=[0.8, 1.1],
                         height_shift_range=0.1,
                         width_shift_range=0.1,
                         brightness_range=(0.3, 1))

    frame_augmentor = ImageDataGenerator(**data_gen_args)
    label_augmentor = ImageDataGenerator(**data_gen_args)

    myimg = image.reshape((1,) + image.shape + (1,))
    mask = mask.reshape((1,) + mask.shape + (1,))

    aug_frames = frame_augmentor.flow(myimg, seed=seed, batch_size=1)
    aug_labels = label_augmentor.flow(mask, seed=seed, batch_size=1)

    return aug_frames, aug_labels


# In[16]:

def post_process(y_pred, threshold, num_blobs):
    pruned_pred = np.zeros_like(y_pred)
    THRESHOLD_CENTROID = 1/4 #based on experiments
    for i in range(y_pred.shape[0]):
        pp = y_pred[i,:,:]
        pp = pp > threshold
        lab = label(pp)
        rps = regionprops(lab)
        area_idx = np.argsort([r.area for r in rps])[::-1]
        new_pp = np.zeros_like(pp)
        for j in area_idx[:num_blobs]:
            if np.linalg.norm(np.asarray(rps[area_idx[0]].centroid) - np.asarray(rps[j].centroid))/y_pred.shape[1] < THRESHOLD_CENTROID:
                new_pp[tuple(rps[j].coords.T)] = True
        pruned_pred[i,:,:] = new_pp

    return pruned_pred

# In[14]:
def compute_IoU(y_test, pred):
    fehlt = 0
    IoU_score = 0
    intersection_acc = 0
    union_acc = 0
    for i in range(pred.shape[0]):
        gt = np.squeeze(y_test[i] > 0)
        new_pred_i = pred[i]

        fehlt += np.count_nonzero(np.logical_and(gt, np.logical_not(new_pred_i)))
        intersection = np.count_nonzero(np.logical_and(gt, new_pred_i))
        union = np.count_nonzero(np.logical_or(gt, new_pred_i))
        intersection_acc += intersection
        union_acc += union
        IoU_score += intersection/union
    return IoU_score/pred.shape[0]
# In[14]:

train_sample_id = 50
labeled_frame_idx = 1

video = np.copy(train_data[train_sample_id]['video'])
labels = train_data[train_sample_id]['label']
box = train_data[train_sample_id]['box']
labeled_frames = train_data[train_sample_id]['frames']

aug_frames, aug_labels = augment_data(video[:,:,labeled_frames[labeled_frame_idx]], 255*labels[:,:,labeled_frames[labeled_frame_idx]].astype(np.ubyte))

nrow = 4
ncol = 4
# generate samples and plot
fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 15 * nrow / ncol))

for ox in ax.reshape(-1):
    # convert to unsigned integers
    image = next(aug_frames)[0].astype('uint8')
    mask = next(aug_labels)[0].astype('uint8')
    ox.imshow(image)
    ox.imshow(mask, alpha=0.5)
    ox.axis('off')

plt.show()

## Deformation (doesn't work yet)

# In[17]:


# X = np.zeros((200, 300))
# X[::10, ::10] = 1
# train_sample_id = 33
# labeled_frame_idx = 1
# crop = False
# seed = 1
#
# video = np.copy(train_data[train_sample_id]['video'])
# labels = train_data[train_sample_id]['label']
# box = train_data[train_sample_id]['box']
# labeled_frames = train_data[train_sample_id]['frames']
#
# X = video[:, :, labeled_frames[labeled_frame_idx]]
#
# Y = labels[:, :, labeled_frames[labeled_frame_idx]]
#
# # apply deformation with a random 3 x 3 grid
# X_deformed = elasticdeform.deform_random_grid(X, sigma=10, points=3)
# # [X_deformed, Y_deformed] = elasticdeform.deform_random_grid([X, Y],  order=[5, 5])
# # [X_deformed, Y_deformed] = elasticdeform.deform_random_grid([X, Y],  sigma=0.01, points=1)
#
#
# # In[18]:
#
#
# plt.imshow(X)
# plt.show

# # the U-net

# In[15]:

N_AUG_PER_SAMPLE = 20
x_train = []
y_train = []
for d in train_data:
    for i in d["frames"]:
        image = d["video"][:, :, i]
        mask = 255 * d["label"][:, :, i].astype(np.ubyte)
        x_train.append(cv2.resize(image, dsize=(360, 360)))
        y_train.append(cv2.resize(mask, dsize=(360, 360)))
        aug_images, aug_masks = augment_data(image, mask)
        for tt in range(N_AUG_PER_SAMPLE):
            x_train.append(cv2.resize(next(aug_images)[0], dsize=(360, 360)))
            y_train.append(cv2.resize(next(aug_masks)[0], dsize=(360, 360)))

## randomly shuffle the training data:
# combined_list = list(zip(x_train, y_train))
# random.shuffle(combined_list)
# x_train, y_train = zip(*combined_list)
# x_train, y_train = list(x_train), list(y_train)

# In[16]:
x_train = np.expand_dims(np.array(x_train, dtype=np.single), 3)
y_train = np.expand_dims(np.array(y_train, dtype=np.single), 3)

# In[16]:
## Custom K-fold split: we want to use only expert labels and their augmentations for the validation
N_SPLITS = 5
amateur_idx = []
expert_idx = []
for ii in range(len(train_data)):
    if train_data[ii]['dataset'] == 'amateur':
        amateur_idx.append(ii)
    else:
        expert_idx.append(ii)

# this only works for N_SPLITS = 5:
train_list = []
test_list = []
N_SAMPLES_PER_VIDEO = 3*(N_AUG_PER_SAMPLE+1)
for fold_no in range(N_SPLITS):
    fold_train = amateur_idx[3:] + expert_idx[10:]
    fold_test = amateur_idx[:3] + expert_idx[:10]

    random.shuffle(expert_idx)
    random.shuffle(amateur_idx)
    fold_train_ext = np.zeros(N_SAMPLES_PER_VIDEO*len(fold_train))
    fold_test_ext = np.zeros(N_SAMPLES_PER_VIDEO*len(fold_test))
    for k in range(len(fold_train)):
        fold_train_ext[N_SAMPLES_PER_VIDEO*k:N_SAMPLES_PER_VIDEO*(k+1)] = fold_train[k]*N_SAMPLES_PER_VIDEO + np.arange(N_SAMPLES_PER_VIDEO)

    for k in range(len(fold_test)):
        fold_test_ext[N_SAMPLES_PER_VIDEO*k:N_SAMPLES_PER_VIDEO*(k+1)] = fold_test[k]*N_SAMPLES_PER_VIDEO + np.arange(N_SAMPLES_PER_VIDEO)

    TR_L = list(fold_train_ext.astype(int))
    random.shuffle(TR_L)
    TE_L = list(fold_test_ext.astype(int))
    random.shuffle(TE_L)

    train_list.append(TR_L)
    test_list.append(TE_L)
# In[17]:
EPOCHS = 10
BATCH_SIZE = 10
fold_no = 0
seed = 1
scores = []
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# for train_idx, test_idx in KFold(n_splits=5, shuffle=True).split(y_train):
for jj in range(len(train_list)):
    train_idx = train_list[jj]
    test_idx = test_list[jj]
    fold_no += 1
    print('------------------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    print('------------------------------------------------------------------------------------')

    keras.backend.clear_session()
    model = get_model((360, 360))
    #model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        # loss=keras.losses.CategoricalCrossentropy(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name='accuracy')]
        #loss=tf.keras.metrics.MeanIoU(num_classes=2),
        #metrics=[tf.keras.metrics.MeanIoU(num_classes=2)]
    )

    history = model.fit(
        x_train[train_idx],
        y_train[train_idx],
        # validation_data=(x_train[test_idx], y_train[test_idx]),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2
    )

    pred = model.predict(x_train[test_idx])
    pred = np.squeeze(pred)

    new_pred = post_process(pred, threshold=0.999, num_blobs=2)
    scores.append(compute_IoU(y_train[test_idx], new_pred))

    print(" -- fold %d score: %d", fold_no, scores[fold_no-1])

print("Average IoU over folds: %d", np.mean(scores))
#model.save('./save_model')
# In[16]:
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# In[16]:
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[20]:

keras.backend.clear_session()
model = get_model((360, 360))
#model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(name='accuracy')]
)

history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2
)

# In[]
#model.save('./save_model')

# Prediction and post processing:

x_test = []
for i in range(len(test_data)):
    for j in range(test_data[i]['video'].shape[2]):
        x_test.append(cv2.resize(test_data[i]['video'][:,:,j], dsize=(360, 360)))
x_test = np.expand_dims(np.array(x_test, dtype=np.single), 3)

test_pred = model.predict(x_test)
test_pred = np.squeeze(test_pred)
test_pred = post_process(test_pred, threshold=0.999, num_blobs=2)

# In[]:
# put predictions in submission format
predictions = []
idx = 0
for d in test_data:
    prediction = np.array(np.zeros_like(d['video']), dtype=np.bool)
    height = prediction.shape[0]
    width = prediction.shape[1]
    for kk in range(prediction.shape[2]):
        mask = (cv2.resize(test_pred[idx], dsize=(width, height)) > 0)
        prediction[:,:,kk] = mask
        idx += 1
    # DATA Strucure
    predictions.append({
        'name': d['name'],
        'prediction': prediction
    }
    )

# In[ ]:


# save in correct format
save_zipped_pickle(predictions, 'my_predictions.pkl')
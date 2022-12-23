# In[1]:
import matplotlib.pyplot as plt
## if plots are not shown in Pycharm, use this:
#import matplotlib
#matplotlib.use('module://backend_interagg')

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
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
# In[3]:

from tensorflow.keras import layers
from sklearn.model_selection import KFold

from tensorflow import keras

from skimage.measure import label, regionprops
import random
# installation: pip install elasticdeform
import elasticdeform

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
   x = layers.Conv2D(n_filters, 3, padding="same")(x)#kernel_initializer = "he_normal"
   x = layers.BatchNormalization()(x)
   x = layers.ReLU()(x)
   x = layers.Conv2D(n_filters, 3, padding="same")(x)#kernel_initializer = "he_normal"
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
    x = layers.Conv2DTranspose(n_filters, 2, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    # x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x


# In[9]:
def get_model(img_size):
    inputs = layers.Input(shape=img_size + (1,))
    m = 2
    f1, p1 = downsample_block(inputs, 16*m)
    f2, p2 = downsample_block(p1, 32*m)
    f3, p3 = downsample_block(p2, 64*m)
    f4, p4 = downsample_block(p3, 128*m)

    bottleneck = double_conv_block(p4, 256*m)

    u6 = upsample_block(bottleneck, f4, 128*m)
    u7 = upsample_block(u6, f3, 64*m)
    u8 = upsample_block(u7, f2, 32*m)
    u9 = upsample_block(u8, f1, 16*m)

    outputs = layers.Conv2D(1, 1, padding="valid", activation="sigmoid")(u9)
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model

# In[]:
# def get_model(img_size):
#     inputs = layers.Input(shape=img_size + (1,))
#     m = 2
#     f1, p1 = downsample_block(inputs, 16*m)
#     f2, p2 = downsample_block(p1, 32*m)
#     f3, p3 = downsample_block(p2, 64*m)
#
#     bottleneck = double_conv_block(p3, 128*m)
#
#     u7 = upsample_block(bottleneck, f3, 64*m)
#     u8 = upsample_block(u7, f2, 32*m)
#     u9 = upsample_block(u8, f1, 16*m)
#
#     outputs = layers.Conv2D(1, 1, padding="valid", activation="sigmoid")(u9)
#
#     unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
#
#     return unet_model


# In[]:
def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    # https://github.com/karolzak/keras-unet/tree/master/keras_unet
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    jac = (intersection + 1.) / (union - intersection + 1.)
    return K.mean(jac)

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# In[10]:
# load data
train_data = load_zipped_pickle("train.pkl")
test_data = load_zipped_pickle("test.pkl")
samples = load_zipped_pickle("sample.pkl")


# In[13]:
# ## If we want to use 'featurewise_std_normalization' we should use the fit method
# # Data augmentation using ImageDataGenerator
# data_gen_args = dict(featurewise_center=True,
#                      featurewise_std_normalization=True,
def augment_data(image, mask):
    seed = 1
    # I don't do horizontal_flip bc the mitral valve is in the same side in all videos
    data_gen_args = dict(rotation_range=10,
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
def show_train_sample(train_sample_id, labeled_frame_idx):
    video = np.copy(train_data[train_sample_id]['video'])
    labels = train_data[train_sample_id]['label']
    labeled_frames = train_data[train_sample_id]['frames']
    X = 0.5*video[:,:,labeled_frames[labeled_frame_idx]] + 0.5*(255*labels[:,:,labeled_frames[labeled_frame_idx]])
    plt.imshow(X)
    plt.show()

def show_test_sample(test_sample_id, frame_idx):
    video = np.copy(test_data[test_sample_id]['video'])
    X = video[:,:,frame_idx]
    plt.imshow(X)
    plt.show()

###### VISUALIZE THE AUGMENTED FRAMES AND LABELS: ######

# train_sample_id = 1
# labeled_frame_idx = 1
# video = np.copy(train_data[train_sample_id]['video'])
# labels = train_data[train_sample_id]['label']
# labeled_frames = train_data[train_sample_id]['frames']
# aug_frames, aug_labels = augment_data(video[:,:,labeled_frames[labeled_frame_idx]], 255*labels[:,:,labeled_frames[labeled_frame_idx]].astype(np.ubyte))
#
# nrow = 4
# ncol = 4
# # generate samples and plot
# fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 15 * nrow / ncol))
#
# for ox in ax.reshape(-1):
#     # convert to unsigned integers
#     image = next(aug_frames)[0].astype('uint8')
#     mask = next(aug_labels)[0].astype('uint8')
#     ox.imshow(image)
#     ox.imshow(mask, alpha=0.5)
#     ox.axis('off')
#
# plt.show()


# In[17]:
###### Visualize elastic deformation: ######

train_sample_id = 1
labeled_frame_idx = 1
crop = False
seed = 1

video = np.copy(train_data[train_sample_id]['video'])
labels = train_data[train_sample_id]['label']
box = train_data[train_sample_id]['box']
labeled_frames = train_data[train_sample_id]['frames']

X = video[:,:,labeled_frames[labeled_frame_idx]]

Y = labels[:,:,labeled_frames[labeled_frame_idx]]

[X_deformed, Y_deformed] = elasticdeform.deform_random_grid([X, Y], sigma=8,  order=1, points=3)

plt.imshow(X_deformed)
plt.imshow(Y_deformed, alpha=0.5)
plt.show()

# In[]:
def show_example(i):
    plt.imshow(x_train[i])
    plt.imshow(y_train[i], alpha=0.5)
    plt.show()

# In[15]:
## Data augmentation:

img_height = 192
img_width = 192
DO_PREPROCESSING = True

N_AUG_SIMPLE_PER_SAMPLE = 10   # number of augmentations using ImageDataGeneratir
N_AUG_DEFORM_PER_SAMPLE = 20   # number of augmentations using elasticdeform
N_AUG_PER_SAMPLE = N_AUG_SIMPLE_PER_SAMPLE + N_AUG_DEFORM_PER_SAMPLE
x_train0 = []
y_train = []
for d in train_data:
    for i in d["frames"]:
        image = d["video"][:, :, i]
        mask = 1 * d["label"][:, :, i].astype(np.ubyte)
        # imgR = cv2.resize(image, dsize=(img_height, img_width))
        # imgRS = 255*np.floor((imgR - np.min(imgR))/(np.max(imgR) - np.min(imgR)))
        # maskR = cv2.resize(mask, dsize=(img_height, img_width))
        x_train0.append(cv2.resize(image, dsize=(img_height, img_width)))
        y_label = 1 * (cv2.resize(mask, dsize=(img_height, img_width)) > 0)
        y_train.append(y_label)
        aug_images, aug_masks = augment_data(image, mask)
        for tt in range(N_AUG_SIMPLE_PER_SAMPLE):
            x_train0.append(cv2.resize(next(aug_images)[0], dsize=(img_height, img_width)))
            aug_label = 1*(cv2.resize(next(aug_masks)[0], dsize=(img_height, img_width)) > 0)
            y_train.append(aug_label)
        for tt in range(N_AUG_DEFORM_PER_SAMPLE):
            [image_deformed, mask_deformed] = elasticdeform.deform_random_grid([image, mask], sigma=8, order=1, points=3)
            x_train0.append(cv2.resize(image_deformed, dsize=(img_height, img_width)))
            aug_label = 1*(cv2.resize(mask_deformed, dsize=(img_height, img_width)) > 0)
            y_train.append(aug_label)

# In[]:
## Preprocessing: (it helps according to K-fold cross validation results; average score got improved from 0.31 to 0.38)
#x_train = (x_train - np.mean(x_train))/np.std(x_train)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

x_train = x_train0.copy()

if DO_PREPROCESSING:
    for i in range(len(x_train)):
        XX = x_train[i]
        XY = cv2.normalize(XX, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #X_eq = clahe.apply(XY)
        X_eq = XY
        x_train[i] = (X_eq - np.mean(X_eq))/np.std(X_eq)

# In[16]:
x_train = np.expand_dims(np.array(x_train, dtype=np.single), 3)
y_train = np.expand_dims(np.array(y_train, dtype=np.single), 3)

# In[16]:
## Custom K-fold split: the goal is to use 3 amatuer videos and 10 expert videos and the augmentations thereof, for the validation
## and for the training set, we use 43 amateur and 9 expert videos and the augmentations thereof

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
BATCH_SIZE = 20
fold_no = 0
seed = 1
scores = []
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

history_l = []
# for train_idx, test_idx in KFold(n_splits=5, shuffle=True).split(y_train):
for jj in range(len(train_list)):
    train_idx = train_list[jj]
    test_idx = test_list[jj]
    fold_no += 1
    print('------------------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    print('------------------------------------------------------------------------------------')

    keras.backend.clear_session()
    model = get_model((img_height, img_width))
    #model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        # loss=keras.losses.BinaryCrossentropy(),
        # metrics=[jaccard_coef],
        loss=dice_coef_loss,
        metrics=[dice_coef]
    )

    history = model.fit(
        x_train[train_idx],
        y_train[train_idx],
        # validation_data=(x_train[test_idx], y_train[test_idx]),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2
    )
    history_l.append(history)
    pred = model.predict(x_train[test_idx])
    pred = np.squeeze(pred)

    new_pred = post_process(pred, threshold=0.999, num_blobs=2)
    score = compute_IoU(y_train[test_idx], new_pred)
    scores.append(score)

    print(" -- fold %d score: %f" %(fold_no, score))
    #break
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
model = get_model((img_height, img_width))
#model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    # loss=keras.losses.BinaryCrossentropy(),
    # metrics=[jaccard_coef]
    loss=dice_coef_loss,
    metrics=[dice_coef]
)

history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2
)

# In[]
model.save('./model10')

# In[]
# Prediction and post processing:

x_test = []
for i in range(len(test_data)):
    for j in range(test_data[i]['video'].shape[2]):
        x_test.append(cv2.resize(test_data[i]['video'][:,:,j], dsize=(img_height, img_width)))

# Preprocessing:

if DO_PREPROCESSING:
    for i in range(len(x_test)):
        XX = x_test[i]
        XY = cv2.normalize(XX, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #X_eq = clahe.apply(XY)
        X_eq = XY
        x_test[i] = (X_eq - np.mean(X_eq))/np.std(X_eq)

x_test = np.expand_dims(np.array(x_test, dtype=np.single), 3)

test_pred = model.predict(x_test)
test_pred = np.squeeze(test_pred)
test_pred = post_process(test_pred, threshold=0.999, num_blobs=2)

# In[]:
# put predictions in submission format
predictions = []
idx = 0
for d in test_data:
    prediction = np.array(np.zeros_like(d['video']), dtype=bool)
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
save_zipped_pickle(predictions, 'seyed_predictions10.pkl')

# In[ ]:
ii = 6
fr = 30
plt.imshow(test_data[ii]['video'][:,:,fr])
plt.imshow(predictions[ii]['prediction'][:,:,fr], alpha=0.5)
plt.show()
if predictions[ii]['prediction'].shape != test_data[ii]['video'].shape:
    print('ERROR')

# In[ ]:
kk = 10

plt.imshow(y_train[test_idx[kk]], cmap='plasma')
plt.imshow(new_pred[kk], cmap='magma', alpha=0.5)
plt.show()
# importing libraries
import glob 
from PIL import Image
import glob
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle




# getting list of masked subjects
masked_name_list = []
for name in glob.glob('/content/sample_data/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset/*'):
  masked_name_list.append(name.split('/')[-1])

# getting list of unmasked subjects
unmasked_name_list = []
for name in glob.glob('/content/sample_data/self-built-masked-face-recognition-dataset/AFDB_face_dataset/*'):
  unmasked_name_list.append(name.split('/')[-1])

# getting the common subjects
masked_name_list = set(masked_name_list)
unmasked_name_list = set(unmasked_name_list)
common_names = masked_name_list.intersection(unmasked_name_list)
common_names = list(common_names)

# sorting list for proper data division
common_names.sort()   


# setting ceiling to number of picks (masked and unmasked) per subject
max_pick = 20    
# choosing 80% of the total nubmer of subjects for creation of train and validation datasets
no_of_subjects =  int(len(common_names)*0.8)  

masked_image_list = []
for name in common_names[:no_of_subjects]:  
  count=0
  for filename in glob.glob('/content/sample_data/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset/'+name+'/*.jpg'): 
      masked_image_list.append(filename)
      count+=1
      if count==max_pick:
        break

unmasked_image_list = []
for name in common_names[:no_of_subjects]: 
  count=0
  for filename in glob.glob('/content/sample_data/self-built-masked-face-recognition-dataset/AFDB_face_dataset/'+name+'/*.jpg'): 
      unmasked_image_list.append(filename)
      count+=1
      if count==max_pick:
        break



# shuffling the masked, unmasked images list
random.shuffle(masked_image_list)
random.shuffle(unmasked_image_list)

# combining
fin_series = []
label_series = []
for mi1 in unmasked_image_list:
  for mi2 in masked_image_list:
    if mi1.split('/')[-2]==mi2.split('/')[-2]:
      lab=1.  # genuine pair
    else:
      lab=0. # impostor pair
    fin_series.append([mi1, mi2, lab])

# creating DataFrame with masked face path, unmasked face path and label
comb_ = pd.DataFrame(fin_series, columns = ['mi1', 'mi2', 'label'])
comb_same = comb_[comb_.label==1].reset_index(drop=True)
comb_diff = comb_[comb_.label==0].reset_index(drop=True)


# splitting the created DataFrame into train and validation sets
comb_same_train, comb_same_val = train_test_split(comb_same, 
                                         random_state=2021,
                                         test_size = 0.2)

comb_diff_train, comb_diff_val = train_test_split(comb_diff, 
                                         random_state=2021,
                                         test_size = 0.2)



# extracting the data such that no. of Genuine and Impostor pairs are equal in strength
comb_train = comb_same_train.append(comb_diff_train[:len(comb_same_train)]).reset_index(drop=True)
comb_val = comb_same_val.append(comb_diff_val[:len(comb_same_val)]).reset_index(drop=True)


# shuffling the obtained dataset
comb_train = shuffle(comb_train, random_state=2021).reset_index(drop=True)
comb_val = shuffle(comb_val, random_state=2021).reset_index(drop=True)



# creating the test dataset
masked_image_list_test = []
for name in common_names[no_of_subjects:]:  
  count=0
  for filename in glob.glob('/content/sample_data/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset/'+name+'/*.jpg'): 
      masked_image_list_test.append(filename)
      count+=1
      if count==max_pick:
        break

unmasked_image_list_test = []
for name in common_names[no_of_subjects:]: 
  count=0
  for filename in glob.glob('/content/sample_data/self-built-masked-face-recognition-dataset/AFDB_face_dataset/'+name+'/*.jpg'): 
      unmasked_image_list_test.append(filename)
      count+=1
      if count==max_pick:
        break


# combining
fin_series_test = []
for mi1 in unmasked_image_list_test:
  for mi2 in masked_image_list_test:
    if mi1.split('/')[-2]==mi2.split('/')[-2]:
      lab=1.  # same
    else:
      lab=0. # different

    fin_series_test.append([mi1, mi2, lab])


comb_test = pd.DataFrame(fin_series_test, columns = ['mi1', 'mi2', 'label'])
comb_same_test = comb_test[comb_test.label==1].reset_index(drop=True)
comb_diff_test = comb_test[comb_test.label==0].reset_index(drop=True)
comb_test_final = comb_same_test.append(comb_diff_test[:len(comb_same_test)]).reset_index(drop=True)



def _parse_function(filename_m, filename_um, label):
  '''
    - This is parser function to create a data generator function using the TensorFlow data API.
  
    - Here, we are not using preprocess_input(samples, version=2) here as the same is just centering the face, 
      which may be specific to dataset.
  
  '''
  image_string_M = tf.io.read_file(filename_m)
  image_string_UM = tf.io.read_file(filename_um)

  image_decoded_M = tf.image.decode_jpeg(image_string_M, channels=3)
  image_decoded_UM = tf.image.decode_jpeg(image_string_UM, channels=3)

  image_decoded_M = tf.image.resize(image_decoded_M, (224, 224))
  image_decoded_UM = tf.image.resize(image_decoded_UM, (224, 224))

  image_M = tf.cast(image_decoded_M, tf.float32)
  image_UM = tf.cast(image_decoded_UM, tf.float32)
  label = tf.cast(label, tf.float32)

  # model specific preprocessing [utils.preprocess_input(x, version=2) for RESNET50 or SENET50]
  xx = tf.constant([91.4953,103.8827,131.0912], dtype=tf.float32)
  yy = tf.broadcast_to(xx, shape=(224,224,3))
  image_M -= yy
  image_UM -= yy

  return (image_M, image_UM), label

train_dataset = train_dataset.map(_parse_function, 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32)



# Creating tf batch Train dataset object

comb_train_m = tf.Variable(comb_train.mi1)
comb_train_um = tf.Variable(comb_train.mi2)
comb_train_label = tf.Variable(comb_train.label.astype('float32'))
train_dataset = tf.data.Dataset.from_tensor_slices((comb_train_m, 
                                                     comb_train_um, 
                                                     comb_train_label))



# Creating tf batch Validation dataset object

comb_val_m = tf.Variable(comb_val.mi1)
comb_val_um = tf.Variable(comb_val.mi2)
comb_val_label = tf.Variable(comb_val.label.astype('float32'))
val_dataset = tf.data.Dataset.from_tensor_slices((comb_val_m, 
                                                     comb_val_um, 
                                                     comb_val_label))


val_dataset = val_dataset.map(_parse_function, 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32)



# Creating tf batch Test dataset object

comb_test_m = tf.Variable(comb_test_final.mi1)
comb_test_um = tf.Variable(comb_test_final.mi2)
comb_test_label = tf.Variable(comb_test_final.label.astype('float32'))
test_dataset = tf.data.Dataset.from_tensor_slices((comb_test_m, 
                                                     comb_test_um, 
                                                     comb_test_label))


test_dataset = test_dataset.map(_parse_function, 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32)






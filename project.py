import tensorflow as tf
from tensorflow.keras import layers,models,Model,losses,optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader,dataset

import cv2

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from sklearn.model_selection import train_test_split

import glob
import os

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Is TensorFlow using GPU?", tf.test.is_gpu_available())

# Detailed GPU information
print("GPU Details:")
print(tf.config.experimental.list_physical_devices('GPU'))

##PART 2

rgb_folder=r"D:\capstone_project\dataset\rgb"
ir_folder=r"D:\capstone_project\dataset\ir"

rgb_files=sorted([f for f in os.listdir(rgb_folder) if f.startswith('254p RGB') and f.endswith('.jpg')])
ir_files=sorted([f for f in os.listdir(ir_folder) if f.startswith('254p Thermal') and f.endswith('.jpg')])

image_pairs = [(os.path.join(rgb_folder, rgb), os.path.join(ir_folder, ir)) 
               for rgb, ir in zip(rgb_files, ir_files)]
plt.figure(figsize=(10,5))
'''for i,(rgb_path,ir_path) in enumerate (image_pairs):
    if i>=10:
        break

    rgb_image=Image.open(rgb_path)
    ir_image=Image.open(ir_path)
 
    plt.subplot(1,2,1)
    plt.imshow(rgb_image)
    plt.title(f'{os.path.basename(rgb_path)}')
    plt.axis('off')


    plt.subplot(1,2,2)
    plt.imshow(ir_image)
    plt.title(f'{os.path.basename(ir_path)}')

    plt.axis('off')
    #plt.show(block=False)
    plt.pause(0.2)
    plt.close()
'''
def load_image_pair(rgb_path,ir_path):
    rgb_image=tf.image.decode_jpeg(tf.io.read_file(rgb_path))
    ir_image=tf.image.decode_jpeg(tf.io.read_file(ir_path))
    rgb_image=tf.image.resize(rgb_image,(256,256))/255.0
    ir_image=tf.image.resize(ir_image,(256,256))/255.0
    return rgb_image,ir_image
# Load the image pairs
rgb_files = sorted(tf.io.gfile.glob(r"D:/capstone_project/dataset/rgb/*.jpg"))
ir_files = sorted(tf.io.gfile.glob(r"D:/capstone_project/dataset/ir/*.jpg"))
print(len(rgb_files), len(ir_files))
#print("RGB Folder Contents:", os.listdir(rgb_folder))
#print("IR Folder Contents:", os.listdir(ir_folder))

train_rgb,temp_rgb,train_ir,temp_ir=train_test_split(rgb_files,ir_files,test_size=0.15,train_size=0.85,random_state=42)
val_rgb,test_rgb,val_ir,test_ir=train_test_split(temp_rgb,temp_ir,test_size=0.5,random_state=42)

def create_dataset(rgb_paths,ir_paths):
    dataset=tf.data.Dataset.from_tensor_slices((rgb_paths,ir_paths))
    dataset=dataset.map(load_image_pair,num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


train_dataset=create_dataset(train_rgb,train_ir).shuffle(1000).batch(2 ).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset=create_dataset(val_rgb,val_ir).batch(2).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset=create_dataset(test_rgb,test_ir).batch(2).prefetch(buffer_size=tf.data.AUTOTUNE)
####
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
#Part 3
def build_efficient_unet_generator(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    noisy_inputs = layers.GaussianNoise(stddev=0.05)(inputs)

    # --- Downsampling Path ---
    # D1 Block
    d1 = layers.Conv2D(32, (4, 4), padding='same')(noisy_inputs)
    d1 = layers.LeakyReLU()(d1)
    d2 = layers.Conv2D(32, (4, 4), padding='same')(d1)
    d2 = layers.LeakyReLU()(d2)
    p1 = layers.MaxPooling2D((2, 2))(d2)  # (128, 128, 32)
    
    # D2 Block
    d3 = layers.Conv2D(64, (4, 4), padding='same')(p1)
    d3 = layers.LeakyReLU()(d3)
    d4 = layers.Conv2D(64, (4, 4), padding='same')(d3)
    d4 = layers.LeakyReLU()(d4)
    p2 = layers.MaxPooling2D((2, 2))(d4)  # (64, 64, 64)

    # D3 Block
    d5 = layers.Conv2D(128, (4, 4), padding='same')(p2)
    d5 = layers.LeakyReLU()(d5)
    d6 = layers.Conv2D(128, (4, 4), padding='same')(d5)
    d6 = layers.LeakyReLU()(d6)
    p3 = layers.MaxPooling2D((2, 2))(d6)  # (32, 32, 128)

    # D4 Block
    d7 = layers.Conv2D(256, (4, 4), padding='same')(p3)
    d7 = layers.LeakyReLU()(d7)

    # --- Skip Connections ---
    # Skip S3
    s3_deconv = layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(d7)
    s3_concat = layers.Concatenate()([s3_deconv, d6])  # (64, 64, 256)
    s3 = layers.Conv2D(128, (4, 4), padding='same')(s3_concat)
    s3 = layers.LeakyReLU()(s3)

    # Skip S2
    s2_deconv = layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(s3)
    s2_concat = layers.Concatenate()([s2_deconv, d4])  # (128, 128, 128)
    s2 = layers.Conv2D(64, (4, 4), padding='same')(s2_concat)
    s2 = layers.LeakyReLU()(s2)

    # Skip S1
    s1_deconv = layers.Conv2DTranspose(32, (2, 2), strides=2, padding='same')(s2)
    s1_concat = layers.Concatenate()([s1_deconv, d2])  # (256, 256, 64)
    s1 = layers.Conv2D(32, (4, 4), padding='same')(s1_concat)
    s1 = layers.LeakyReLU()(s1)

    # --- Upsampling Path ---
    # U4 Block
    u4 = layers.Conv2D(256, (4, 4), padding='same')(d7)

    # U3 Block
    u3_deconv = layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(u4)
    u3_concat = layers.Concatenate()([u3_deconv, s3])  # (64, 64, 256)
    u3 = layers.Conv2D(128, (4, 4), padding='same')(u3_concat)
    u3 = layers.LeakyReLU()(u3)

    # U2 Block
    u2_deconv = layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(u3)
    u2_concat = layers.Concatenate()([u2_deconv, s2])  # (128, 128, 128)
    u2 = layers.Conv2D(64, (4, 4), padding='same')(u2_concat)
    u2 = layers.LeakyReLU()(u2)

    # U1 Block
    u1_deconv = layers.Conv2DTranspose(32, (2, 2), strides=2, padding='same')(u2)
    u1_concat = layers.Concatenate()([u1_deconv, s1])  # (256, 256, 64)
    u1 = layers.Conv2D(32, (4, 4), padding='same')(u1_concat)
    u1 = layers.LeakyReLU()(u1)

    # Final Output
    output = layers.Conv2D(3, (4, 4), activation='sigmoid', padding='same')(u1)

    return Model(inputs, output, name="Efficient_U-Net_Generator")

# Instantiate and display the model summary
generator = build_efficient_unet_generator()
#efficient_unet_generator.summary()

#Part 4
def build_mapper_module(input_shape=(256,256,3)):
    inputs = layers.Input(shape=input_shape)

    #7 color spaces
    lab_map=layers.Conv2D(3,(1,1),padding='same')(inputs)
    ycrcb_map=layers.Conv2D(3,(1,1),padding='same')(inputs)
    heat_map=layers.Conv2D(3,(1,1),padding='same')(inputs)
    hot_map=layers.Conv2D(3,(1,1),padding='same')(inputs)
    gray_map=layers.Conv2D(3,(1,1),padding='same')(inputs)
    hsl_map=layers.Conv2D(3,(1,1),padding='same')(inputs)
    hsv_map=layers.Conv2D(3,(1,1),padding='same')(inputs)

    combined_output=layers.Add()([
        0.14*lab_map,
        0.14*ycrcb_map,
        0.14*heat_map,
        0.14*hot_map,
        0.14*gray_map,
        0.14*hsl_map,
        0.14*hsv_map
    ])

    final_output=layers.Activation('sigmoid')(combined_output)

    return Model(inputs,final_output,name="Mapper_Module")

mapper = build_mapper_module(input_shape=(256, 256, 3))
input_image = np.random.rand(1, 256, 256, 3)  # Batch of 1 image with shape (256, 256, 3)
output_image = mapper(input_image)  # Pass the image through the generator
print(output_image.shape)

#mapper.summary()

#Part 5
def build_discriminator(input_shape=(256, 256, 6)):
    inputs = layers.Input(shape=input_shape)

    # First convolution block
    x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Second convolution block
    x = layers.Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Third convolution block
    x = layers.Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Fourth convolution block (reduce downsampling)
    x = layers.Conv2D(512, (4, 4), strides=1, padding='same')(x)  # Use strides=1
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Output convolution layer (PatchGAN output) with appropriate padding
    x = layers.Conv2D(1, (4, 4), strides=1, padding='same')(x)  # Change strides to 1
    output = layers.Activation('sigmoid')(x)  # Use sigmoid for binary classification

    return Model(inputs, output, name="PatchGAN_Discriminator")
discriminator = build_discriminator(input_shape=(256, 256, 6))
#discriminator.summary()
print("done1")


#Part 6
def integration(generator,mapper,discriminator):
    rgb_input=layers.Input(shape=(256,256,3),name="RGB_Input")
    generated_ir=generator(rgb_input)
    mapped_ir=mapper(generated_ir)
    discriminator_output = discriminator(tf.concat([mapped_ir, rgb_input], axis=-1))
    print(discriminator_output.shape)
    #discriminator_output = discriminator(mapped_ir)

    ic_gan=Model(inputs=rgb_input,outputs=[discriminator_output,mapped_ir],name="IC-GAN")
    return ic_gan

ic_gan_model=integration(generator,mapper,discriminator)
#ic_gan_model.summary()
print("done123")

#Part7
bce=tf.keras.losses.BinaryCrossentropy(from_logits=True)
def gan_loss(generator,discriminator,real_rgb,real_ir):
    generated_ir=generator([real_rgb],training=True)

    real_output=discriminator(tf.concat([real_ir,real_ir],axis=-1),training=True)
    fake_output=discriminator(tf.concat([generated_ir,real_ir],axis=-1),training=True)

    disc_loss=bce(tf.ones_like(real_output),real_output)+bce(tf.zeros_like(fake_output),fake_output)

    gen_loss=bce(tf.ones_like(fake_output),fake_output)

    return gen_loss,disc_loss

def mapper_loss(mapped_ir,real_ir):
    return tf.reduce_mean(tf.abs(mapped_ir-real_ir))

def l1_loss(generated_ir,real_ir):
    return tf.reduce_mean(tf.abs(generated_ir-real_ir))

def total_loss(generator, mapper, discriminator, real_rgb, real_ir, lambda_value=10):

    generated_ir=generator([real_rgb],training=True)
    mapped_ir=mapper(generated_ir,training=True)

    gen_loss,disc_loss=gan_loss(generator,discriminator,real_rgb,real_ir)
    map_loss=mapper_loss(mapped_ir,real_ir)
    l1_loss_val=l1_loss(generated_ir,real_ir)

    total_gen_loss=gen_loss+lambda_value*(map_loss+l1_loss_val)

    return total_gen_loss,disc_loss,map_loss,l1_loss_val

generated_images="D:\capstone_project\generated_images"
##
def save_and_visualize_images(generator, mapper, val_rgb, val_ir, epoch, save_dir='generated_images'):
    # Pick a sample from validation data
    sample_rgb = val_rgb[0:1]  # Take the first batch (1 image)
    sample_ir = val_ir[0:1]    # Take the corresponding IR image

    # Generate IR image from the RGB input
    generated_ir = generator(sample_rgb, training=False)
    
    # Map the generated IR image to the mapped IR using the mapper
    mapped_ir = mapper(generated_ir, training=False)

    # Visualize the generated and mapped IR images
    plt.figure(figsize=(10, 5))

    # Plot the generated IR image
    plt.subplot(1, 2, 1)
    plt.imshow(generated_ir[0])  # [0] to remove batch dimension
    plt.title(f"Generated IR - Epoch {epoch}")
    plt.axis('off')

    # Plot the mapped IR image
    plt.subplot(1, 2, 2)
    plt.imshow(mapped_ir[0])  # [0] to remove batch dimension
    plt.title(f"Mapped IR - Epoch {epoch}")
    plt.axis('off')

    # Save the images to disk
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"{save_dir}/epoch_{epoch}_generated_mapped_ir.png")
    plt.close()

##
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('"D:\capstone_project\model_saved_loss_improves/my_model_checkpoint.h5',   # Path to save the model
    save_weights_only=False,                       # Save both model and weights
    save_best_only=True,                           # Save the model only if it has improved
    monitor='val_loss',                           # Monitor validation loss (or any other metric)
    verbose=1                                      # Print a message when the model is saved
)
#quantitative analysis
def compute_metrics(generated_ir,real_ir):

    mse=np.mean((generated_ir-real_ir)**2)
    psnr=-10*np.log10(mse) if mse!=0 else float('inf')
    win_size=3
    ssim_value=ssim(generated_ir,real_ir,data_range=generated_ir.max()-generated_ir.min(),win_size=3)
    return mse,psnr,ssim_value
#Part 8
def train_step(generator, mapper, discriminator, real_rgb, real_ir, lambda_value=10):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_ir = generator(real_rgb ,training=True)
        mapped_ir = mapper(generated_ir, training=True)
        


        real_output = discriminator(tf.concat([real_ir, real_ir], axis=-1), training=True)
        fake_output = discriminator(tf.concat([mapped_ir, real_ir], axis=-1), training=True)

        gen_loss, disc_loss = gan_loss(generator, discriminator, real_rgb, real_ir)
        map_loss = mapper_loss(mapped_ir, real_ir)
        l1_loss_value = l1_loss(generated_ir, real_ir)
       
       
        total_gen_loss = gen_loss + lambda_value * (map_loss + l1_loss_value)

    gen_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables + mapper.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables + mapper.trainable_variables))

        # Update Discriminator
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return total_gen_loss, disc_loss, map_loss, l1_loss_value

#Part 9
def validation_step(generator,mapper,discriminator,real_rgb,real_ir,lambda_value=10):
    generated_ir=generator(real_rgb,training=False)
    mapped_ir=mapper(generated_ir,training=False)
    real_output=discriminator(tf.concat([real_ir,real_ir],axis=-1),training=False)
    fake_output=discriminator(tf.concat([mapped_ir,real_ir],axis=-1),training=False)

    gen_loss, disc_loss = gan_loss(generator, discriminator, real_rgb, real_ir)
    map_loss = mapper_loss(mapped_ir, real_ir)
    l1_loss_value = l1_loss(generated_ir, real_ir)
    total_gen_loss = gen_loss + lambda_value * (map_loss + l1_loss_value)


    return total_gen_loss, disc_loss, map_loss, l1_loss_value

gen_optimizer=tf.keras.optimizers.Adam(1e-4)
disc_optimizer=tf.keras.optimizers.Adam(1e-4)

EPOCHS=50
lambda_value=10
for epoch in range(EPOCHS):
    epoch_gen_loss=0
    epoch_disc_loss=0
    epoch_map_loss=0
    epoch_l1_loss=0
    batch_count=0
    for real_rgb,real_ir in train_dataset:



        total_gen_loss,disc_loss,map_loss,l1_loss_value=train_step(
            generator,mapper,discriminator,real_rgb,real_ir,lambda_value)
        epoch_gen_loss+=total_gen_loss
        epoch_disc_loss+=disc_loss
        epoch_map_loss+=map_loss
        epoch_l1_loss+=l1_loss_value
        batch_count+=1
        print("training batch",batch_count)


    val_gen_loss=0
    val_disc_loss=0
    val_map_loss=0
    val_l1_loss=0
    val_mse = 0
    val_psnr = 0
    val_ssim = 0
    val_batch_count=0
    for val_rgb,val_ir in val_dataset:
        
        total_gen_loss,disc_loss,map_loss,l1_loss_value=validation_step(generator,mapper,discriminator,val_rgb,val_ir,lambda_value)
        val_gen_loss+=total_gen_loss
        val_disc_loss+=disc_loss
        val_map_loss += map_loss
        val_l1_loss += l1_loss_value
        val_batch_count += 1
        print("validation batch",batch_count)

        #qualitative analysis
        generated_ir = generator(val_rgb, training=False).numpy()
        val_ir = val_ir.numpy()
        
        mse, psnr, ssim_value = compute_metrics(generated_ir[0], val_ir[0])
        val_mse += mse
        val_psnr += psnr
        val_ssim += ssim_value
    
    avg_mse = val_mse / val_batch_count
    avg_psnr = val_psnr / val_batch_count
    avg_ssim = val_ssim / val_batch_count
        
    metrics_path = "D:\capstone_project\saved_metrics"
    with open (f"{metrics_path}/metrics_epoch_{epoch+1}.txt","w") as f:
        f.write(f"Epoch {epoch+1}\n")
        f.write(f"Validation MSE: {avg_mse:.4f}\n")
        f.write(f"Validation PSNR: {avg_psnr:.4f}\n")
        f.write(f"Validation SSIM: {avg_ssim:.4f}\n")



    generator.save(r"D:\capstone_project\model_saved_after_epoch\my_model_generator.h5")
    mapper.save(r"D:\capstone_project\model_saved_after_epoch\my_model_mapper.h5")
    discriminator.save(r"D:\capstone_project\model_saved_after_epoch\my_model_discriminator.h5")

    ic_gan_model.compile(
    optimizer='adam',  # You can choose a different optimizer if needed
    loss='mean_squared_error',  # Use a loss function suitable for your task
    metrics=['accuracy']  # You can add other metrics like 'mse' or 'ssim' if needed
)
    history = ic_gan_model.fit(
    train_dataset,
    epochs=50,
    batch_size=2,
    validation_data=val_dataset,
    callbacks=[checkpoint_callback])   # Add the checkpoint callback here

   
    print(f"Epoch {epoch+1}/{EPOCHS}\n,Gen Loss:{total_gen_loss.numpy()},"
         f"Disc loss:{disc_loss.numpy()},Mapper Loss:{map_loss.numpy()},Batch Number:{batch_count}" )
    
    if (epoch + 1) % 5 == 0:
        # Pick a batch from the validation dataset
        val_rgb_batch, val_ir_batch = next(iter(val_dataset))  # Get a batch of validation data
        save_and_visualize_images(generator, mapper, val_rgb_batch, val_ir_batch, epoch)
#Part 10
output_path_generated="D:\capstone_project\output_generated"
output_path_mapped="D:\capstone_project\output_mapped"

def test_model(generator,test_rgb_path,output_path):
    test_image=tf.image.decode_jpeg(tf.io.read_file(test_rgb_path))
    test_image = tf.image.resize(test_image, (256, 256)) / 255.0
    test_image = tf.expand_dims(test_image, axis=0)

    
    generated_ir = generator([test_image], training=False)

    mapped_ir = mapper(generated_ir, training=False)

    generated_ir = tf.squeeze(generated_ir, axis=0) 
    mapped_ir = tf.squeeze(mapped_ir, axis=0)

    tf.keras.preprocessing.image.save_img(output_path_generated, generated_ir)
    tf.keras.preprocessing.image.save_img(output_path_mapped, mapped_ir)
    print(f"Generated IR saved to: {output_path_generated}")
    print(f"Mapped IR saved to: {output_path_mapped}")
'''   
def qualitative_visual():

    epochs = list(range(1, EPOCHS + 1))
    mse_values = [...]  # Populate with saved MSE values
    psnr_values = [...]  # Populate with saved PSNR values
    ssim_values = [...]  # Populate with saved SSIM values

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mse_values, label="MSE")
    plt.plot(epochs, psnr_values, label="PSNR")
    plt.plot(epochs, ssim_values, label="SSIM")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Performance Metrics Over Epochs")
    plt.legend()
    plt.savefig("D:/capstone_project/metrics/metrics_plot.png")
    plt.show()

qualitative_visual()
'''
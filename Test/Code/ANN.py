import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_resnet50(input_shape):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    return base_model

def build_multimodal_model(input_shape):
    # 输入层
    rgb_input = layers.Input(shape=input_shape, name='rgb_input')
    depth_input = layers.Input(shape=input_shape, name='depth_input')
    # 特征提取
    resnet_rgb = build_resnet50(input_shape)
    resnet_depth = build_resnet50(input_shape)
    rgb_features = resnet_rgb(rgb_input)
    depth_features = resnet_depth(depth_input)
    # 展平层
    rgb_features = layers.Flatten()(rgb_features)
    depth_features = layers.Flatten()(depth_features)
    # L2 归一化
    rgb_features = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1)
        )(rgb_features)
    depth_features = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1)
        )(depth_features)
    # 合并特征
    combined_features = layers.Concatenate()([rgb_features, depth_features])
    # 全连接层
    x = layers.Dense(1024, activation='relu')(combined_features)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    # 输出层 (x, y, h, w, θ)
    output = layers.Dense(5, activation='linear')(x)
    # 构建模型
    model = models.Model(inputs=[rgb_input, depth_input], outputs=output)
    return model

input_shape = (224, 224, 3)

model = build_multimodal_model(input_shape)

model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss=MeanSquaredError())

model.summary()



model.fit(train_generator, epochs=30, steps_per_epoch=100)

datagen = ImageDataGenerator(rescale=1./255)

rgb_dir = 'path_to_rgb_images'
depth_dir = 'path_to_depth_images'

rgb_generator = datagen.flow_from_directory(rgb_dir, target_size=(224, 224), batch_size=32, class_mode=None, seed=42)
depth_generator = datagen.flow_from_directory(depth_dir, target_size=(224, 224), batch_size=32, class_mode=None, seed=42)

def combined_generator(rgb_gen, depth_gen):
    while True:
        rgb_images = rgb_gen.next()
        depth_images = depth_gen.next()
        yield [rgb_images, depth_images], grasp_labels


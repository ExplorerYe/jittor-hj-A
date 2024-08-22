import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
import random

jt.flags.use_cuda = 1

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='A')
    return parser.parse_args()

def load_model_and_classes():
    model, preprocess = clip.load("ViT-B-32.pkl")
    classes = open('Dataset/classes.txt').read().splitlines()
    new_classes = process_classes(classes)
    text = clip.tokenize(new_classes)
    with jt.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return model, preprocess, text_features

def process_classes(classes):
    new_classes = []
    for c in classes:
        c = c.split(' ')[0]
        c = c.replace('Animal', '').replace('Thu-dog', '').replace('Caltech-101', '').replace('Food-101', '')
        c = 'a photo of ' + c.strip()
        new_classes.append(c)
    return new_classes

def load_data(imgs_dir, file_path):
    labels = open(file_path).read().splitlines()
    imgs = [os.path.join(imgs_dir, l.split(' ')[0]) for l in labels]
    labels = [int(l.split(' ')[1]) for l in labels]
    return imgs, labels


#随机选6张，4张用作训练，2张用作验证
def select_images_per_class(imgs, labels, num_per_class=6, num_train=4):
    from collections import defaultdict
    class_dict = defaultdict(list)
    for img, label in zip(imgs, labels):
        class_dict[label].append(img)
    
    train_imgs = []
    train_labels = []
    val_imgs = []
    val_labels = []
    for label, img_list in class_dict.items():
        if len(img_list) >= num_per_class:
            chosen_imgs = random.sample(img_list, num_per_class)
        else:
            chosen_imgs = img_list
        
        train_imgs.extend(chosen_imgs[:num_train])
        train_labels.extend([label] * num_train)
        val_imgs.extend(chosen_imgs[num_train:])
        val_labels.extend([label] * (len(chosen_imgs) - num_train))
    
    return train_imgs, train_labels, val_imgs, val_labels

def extract_features(model, preprocess, images):
    features = []
    with jt.no_grad():
        for img_path in tqdm(images):
            image = Image.open(img_path)
            image = preprocess(image).unsqueeze(0)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
    return jt.cat(features).numpy()

def train_classifier(train_features, train_labels):
    classifier = LogisticRegression(random_state=0, C=8.960, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    return classifier


def evaluate_model(classifier, features, labels):
    predictions = classifier.predict(features)
    return accuracy_score(labels, predictions)

def load_and_process_test_data(imgs_dir, model, preprocess):
    test_imgs = [f for f in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print('Testing data processing:')
    test_features = []
    with jt.no_grad():
        for img in tqdm(test_imgs):
            img_path = os.path.join(imgs_dir, img)
            try:
                image = Image.open(img_path)
                image = preprocess(image).unsqueeze(0)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                test_features.append(image_features)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    return test_imgs, jt.cat(test_features).numpy()

def predict_and_save_results(classifier, test_features, test_imgs, result_file):
    predictions = classifier.predict_proba(test_features)
    with open(result_file, 'w') as save_file:
        for i, prediction in enumerate(predictions):
            prediction = np.asarray(prediction)
            top5_idx = prediction.argsort()[-1:-6:-1]
            save_file.write(test_imgs[i] + ' ' + ' '.join(str(idx) for idx in top5_idx) + '\n')

def main():
    args = parse_arguments()
    model, preprocess, _ = load_model_and_classes()
    imgs, labels = load_data('Dataset/', 'Dataset/train.txt')
    train_imgs, train_labels, val_imgs, val_labels = select_images_per_class(imgs, labels, num_per_class=6, num_train=4)

    train_features = extract_features(model, preprocess, train_imgs)
    val_features = extract_features(model, preprocess, val_imgs)
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    classifier = train_classifier(train_features, train_labels)
    print("Training complete.")
    val_accuracy = evaluate_model(classifier, val_features, val_labels)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # 加载测试数据
    test_imgs_dir = 'Dataset/TestSet' + args.split
    test_imgs, test_features = load_and_process_test_data(test_imgs_dir, model, preprocess)
    
    #保存
    result_file = 'result.txt'
    predict_and_save_results(classifier, test_features, test_imgs, result_file)
    print("Testing complete and results saved.")

    
    
if __name__ == '__main__':
    main()
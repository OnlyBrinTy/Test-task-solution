from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from keras import Model
from keras.models import load_model
from keras.utils import image_utils
from tqdm import tqdm
import numpy as np
import os

from file_utils import load_csv


class DistilBertModel:
    def __init__(self, path_to_model, labels_num):
        self.load_model(f"{path_to_model}")
        self.num_labels = labels_num

    def load_model(self, path_to_model):
        print("Loading model...")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(path_to_model)
        self.model = DistilBertForSequenceClassification.from_pretrained(path_to_model, num_labels=self.num_labels)

        print("Loaded!")

    def get_embeddings(self, texts: list) -> list:
        tokenized_text = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        outputs = self.model(**tokenized_text, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        return last_hidden_states[:, 0, :]


def generate_image_embs(split):
    data_dir = f'../images/{split}'
    save_dir = f'data/image_embs/{split}'

    model = load_model('models/vgg16_model2.h5')
    # set penultimate layer of neural network as output
    embed_model = Model(inputs=model.input, outputs=model.get_layer('dense_6').output)

    # list of pictures paths
    file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
    for file in tqdm(file_list):
        basename = os.path.basename(file)
        save_path = os.path.join(save_dir, f"{basename}.npy")

        if os.path.exists(save_path):
            continue

        image = image_utils.load_img(file, color_mode='rgb', target_size=(224, 224))
        arr_image = image_utils.img_to_array(image)
        arr_image = np.expand_dims(arr_image, axis=0)
        model_output = embed_model.predict(arr_image, verbose=0)

        np.save(save_path, model_output[0])


def generate_embeddings_on_batch(model, batch: list, save_dir):
    product_ids = [row['product_id'] for row in batch]
    texts = [row['text'] for row in batch]
    embeddings = model.get_embeddings(texts)

    for i in range(len(batch)):
        save_path = os.path.join(save_dir, f'{product_ids[i]}.npy')
        np.save(save_path, embeddings[i].detach().cpu().numpy())


def generate_text_embs(split):
    df = load_csv(f'./data/{split}_processed.csv')
    save_dir = f'data/text_embs/{split}'

    model = DistilBertModel("./models/distilb", 874)
    batch_size = 8

    batch = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        product_id = row['product_id']
        save_path = os.path.join(save_dir, f"{product_id}.npy")

        if os.path.exists(save_path):
            continue
        elif len(batch) == batch_size:
            generate_embeddings_on_batch(model, batch, save_dir)
            batch.clear()

        batch.append(row)

    if batch:
        generate_embeddings_on_batch(model, batch, save_dir)


def main():
    generate_image_embs('train')
    generate_text_embs('train')
    generate_image_embs('test')
    generate_text_embs('test')


if __name__ == "__main__":
    main()

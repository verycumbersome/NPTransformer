import os
import fasttext.util

import numpy as np

import nn


def get_embeddings(languages):
    # Download and prep language word embeddings
    embeddings = {}
    for lang in languages:
        embed_path = "cc."+lang+".300.bin"

        if not os.path.exists(embed_path):
            fasttext.util.download_model(lang, if_exists="ignore")

        # Move embeddings to data folder and reduce embeddings to dim 50
        os.rename(embed_path, "data/" + embed_path)
        os.remove(embed_path + ".gz")

        embeddings[lang] = fasttext.load_model("data/" + embed_path)
        fasttext.util.reduce_model(embeddings[lang], 50)

    return embeddings


class Model(nn.Net):
    def __init__(self):
        self.L1 = nn.LinearLayer(784, 50)
        self.L2 = nn.LinearLayer(50, 10)

        self.layers = [
            self.L1,
            self.L2,
        ]

    def forward(self, x):
        x = x.reshape(784)

        x = self.L1.calc(x)
        x = self.L2.calc(x)

        x = nn.softmax(x)

        return x

if __name__=="__main__":
    net = Model()

    embeddings = get_embeddings(["en", "fr"])

    print(en.get_word_vector("hello"))
    print(fr.get_word_vector("bonjour"))

    # nn.train_model(net, train_data, test_data, num_epochs=20)

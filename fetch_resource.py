#!/usr/bin/env python3

from pathlib import Path
import urllib.request


def retrieve_model():
    model_dir = Path(__file__).parent / "./model"
    if not model_dir.exists():
        model_dir.mkdir()
    print("Downloading VGG16 ONNX file...")
    urllib.request.urlretrieve(
        "https://www.dropbox.com/s/bjfn9kehukpbmcm/VGG16.onnx?dl=1",
        str(model_dir / "VGG16.onnx"))

    print("Downloading VGG16 category list...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt",
        str(model_dir / "synset_words.txt"))


def retrieve_data():
    data_dir = Path(__file__).parent / "./data"
    if not data_dir.exists():
        data_dir.mkdir()

    print("Downloading hen image...")
    urllib.request.urlretrieve(
        "https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg",
        str(data_dir / "Light_sussex_hen.jpg")
    )

    print("Downloading cat image...")
    urllib.request.urlretrieve(
        "https://upload.wikimedia.org/wikipedia/commons/9/97/Feral_cat_Virginia_crop.jpg",
        str(data_dir / "Feral_cat_Virginia_crop.jpg")
    )


if __name__ == "__main__":
    retrieve_model()
    retrieve_data()

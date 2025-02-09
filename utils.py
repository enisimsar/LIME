import io
import cv2
import torch
import string
import nltk
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# from sklearn.cluster import KMeans
from cuml import KMeans
from cuml.cluster import KMeans

import cudf, cupy
cupy.cuda.Device(0).use()

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")


def plot_array(title, arr, cmap='viridis'):
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=24)
    ax.imshow(arr, cmap=cmap)
    ax.axis('off')

def locate_points(prompt, attention_store, tokenizer, top_k=100):
    attn = attention_store.aggregate_attention(from_where=[
        "down_cross", 
        "up_cross",
        "mid_cross",
    ])

    tokens = tokenizer(prompt)["input_ids"]
    decoder = tokenizer.decode

    words = [decoder(token) for token in tokens]
    tags = nltk.pos_tag(words)

    selected_indices = []
    stop_words = set(stopwords.words('english'))

    for i, (word, pos) in enumerate(tags):
        if i == 0:
            continue
        if word in stop_words or word in string.punctuation or pos.startswith("VB"):
            continue
        selected_indices.append(i)

    if selected_indices:
        avg_attn = attn[:, :, selected_indices].mean(dim=2)
    else:
        print("No tokens selected for averaging.")
        avg_attn = None

    fig, axes = plt.subplots(1, len(tokens) + 1, figsize=(4 * len(tokens) + 1, 3))

    for idx in range(len(tokens)):
        channel_image = attn[:, :, idx].detach().cpu().float().numpy()
        im = axes[idx].imshow(channel_image, cmap='viridis')
        axes[idx].set_title(f"{decoder(int(tokens[idx]))}", fontsize=24)
        axes[idx].axis('off')
        fig.colorbar(im, ax=axes[idx])

    if avg_attn is not None:
        avg_attn_np = avg_attn.detach().cpu().float().numpy()
        im = axes[-1].imshow(avg_attn_np, cmap='viridis')
        axes[-1].set_title("Average", fontsize=24)
        axes[-1].axis('off')
        fig.colorbar(im, ax=axes[-1])
    else:
        axes[-1].set_title("Average (None)", fontsize=24)
        axes[-1].axis('off')

    plt.tight_layout()

    if avg_attn is not None:
        rows, cols = avg_attn.shape
        values, indices = torch.topk(avg_attn.view(-1), top_k)
        coordinates = [(int(index / cols), int(index % cols)) for index in indices]
    else:
        coordinates = []

    return avg_attn, coordinates

def get_segmentation_map(attention_store, out_res, num_clusters=8):
    attn = attention_store.aggregate_attention(from_where=[
        "down_feats", 
        "up_feats",
        "mid_feats",
    ])
    arr = attn.cpu().float().numpy().reshape(-1, attn.shape[-1])
    arr = normalize(arr, axis=1, norm='l2')

    # kmeans = KMeans(n_clusters=num_clusters, n_init=10).fit(arr)
    # labels = kmeans.labels_

    cudf_arr = cudf.DataFrame(arr, )

    kmeans_float = KMeans(n_clusters=num_clusters, n_init=1)
    kmeans_float.fit(cudf_arr)

    labels = kmeans_float.labels_.values.astype(np.uint8).get()
    labels_spatial = labels.reshape(attn.shape[0], attn.shape[1])
    labels_spatial = cv2.resize(labels_spatial, dsize=(out_res, out_res), interpolation=cv2.INTER_NEAREST)

    plot_array("Segmentation Map", labels_spatial, cmap='viridis')

    return labels_spatial

def extract_binary_mask(labels_spatial, coordinates):
    labels = list(set([labels_spatial[coord[0], coord[1]] for coord in coordinates]))
    mask = np.isin(labels_spatial, labels)

    plot_array("Binary Mask", mask, cmap="gray")

    return mask

def get_effective_tokens(prompt, tokenizer):
    input_ids = tokenizer(prompt)["input_ids"]
    str_tokens = [tokenizer.decode(int(token)) for token in input_ids]

    effective_tokens = [] 
    for i, word in enumerate(str_tokens):
        if i == 0 or word in set(stopwords.words('english')): 
            continue
        effective_tokens.append(i)

    return effective_tokens

# LIME: Localized Image Editing via Attention Regularization in Diffusion Models [WACV 2025]

[![Project Website](https://img.shields.io/badge/Project-Website-green)](https://enis.dev/LIME/) [![arXiv](https://img.shields.io/badge/arXiv-2312.06059-b31b1b.svg)](https://arxiv.org/abs/2312.09256)

><p align="center">

[Enis Simsar](https://enis.dev/), [Alessio Tonioni](https://alessiotonioni.github.io/), [Yongqin Xian](https://xianyongqin.github.io/), [Thomas Hofmann](https://da.inf.ethz.ch/), [Federico Tombari](https://federicotombari.github.io/)

></p>
>
> Diffusion models (DMs) have gained prominence due to their ability to generate high-quality varied images with recent advancements in text-to-image generation. The research focus is now shifting towards the controllability of DMs. A significant challenge within this domain is localized editing, where specific areas of an image are modified without affecting the rest of the content. This paper introduces LIME for localized image editing in diffusion models. LIME does not require user-specified regions of interest (RoI) or additional text input, but rather employs features from pre-trained methods and a straightforward clustering method to obtain precise editing mask. Then, by leveraging cross-attention maps, it refines these segments for finding regions to obtain localized edits. Finally, we propose a novel cross-attention regularization technique that penalizes unrelated cross-attention scores in the RoI during the denoising steps, ensuring localized edits. Our approach, without re-training, fine-tuning and additional user inputs, consistently improves the performance of existing methods in various editing benchmarks.

## Setup

### Environment
To set up the environment, please run:
``` bash
conda create -n lime python=3.11
pip install -r requirements.txt
```

### Hugging Face Diffusers Library
Our code relies also on Hugging Face's [diffusers](https://github.com/huggingface/diffusers) library for downloading the Stable Diffusion v1.5 model. 


## Usage

To explore LIME, you can use the jupyter notebook `lime.ipynb`.

## Citation

If you find our work useful, please consider citing our paper:

```
@misc{simsar2023lime,
      title={LIME: Localized Image Editing via Attention Regularization in Diffusion Models}, 
      author={Enis Simsar and Alessio Tonioni and Yongqin Xian and Thomas Hofmann and Federico Tombari},
      year={2023},
      eprint={2312.09256},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<h1 align="center">
  <i>Low-Rank Expert Merging for Multi-Source Domain Adaptation in Person Re-Identification</i>
</h1>

<p align="center">
  <a href="https://www.linkedin.com/in/taha-mustapha-nehdi-240585203/" target='_blank'>Taha Mustapha Nehdi</a>,&nbsp;
  <a href="https://www.linkedin.com/in/nairouz-mrabah-b48162101/" target='_blank'>Nairouz Mrabah</a>,&nbsp;
  <a href="https://www.linkedin.com/in/atif-belal-15779821a/" target='_blank'>Atif Belal</a>,&nbsp;
  <a href="https://www.linkedin.com/in/marco-pedersoli-50677321b/" target='_blank'>Marco Pedersoli</a>,&nbsp;
  <a href="https://www.linkedin.com/in/eric-granger-4062324/" target='_blank'>Eric Granger </a>,&nbsp;
  <br>
  Quebec University <br>
  École de technologie supérieure (ÉTS) <br>
  📧 Primary Contact: taha-mustapha.nehdi.1@ens.etsmtl.ca
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2508.06831" target='_blank'>
    <img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2403.16848-b31b1b?style=flat-square">
  </a>
  <a href="">
    <img alt="Static Badge" src="https://img.shields.io/badge/WACV%202026-%F0%9F%8C%B5-%235E86C1?style=flat-square">
  </a>
</p>

## :mag: Overview

**TL; DR.** We propose SAGE-reID, a source-free multi-source domain adaptation framework for person re-identification based on gated LoRA experts. It first learns lightweight source-specific LoRA adapters without accessing source data during adaptation, then uses a small gating network to dynamically merge these experts while keeping the backbone fixed, achieving state-of-the-art accuracy with <2% extra parameters on Market-1501, DukeMTMC-reID, and MSMT17.

![Overview](./assets/wacv2026pap.png)


## :fire: News
- <span style="font-variant-numeric: tabular-nums;">**2025.10.06**</span>: Our paper is accepted by WACV 2026 :tada: :tada:. The revised paper and a more efficient codebase will be released in December. Almost there :nerd_face: ~

- <span style="font-variant-numeric: tabular-nums;">**2025.08.09**</span>: The first version of our paper is released at [arXiv:2508.06831v1](https://arxiv.org/abs/2508.06831)  :pushpin:.

## :dash: Quick Start
- See [INSTALL.md](./docs/INSTALL.md) for instructions of installing required components.
- See [DATASET.md](./docs/DATASET.md) for datasets download and preparation.
- See [GET_STARTED.md](./docs/GET_STARTED.md) for how to get started with our SAGE-reID, including pre-training, adaptation, and Low-Rank Merging.

## :bouquet: Acknowledgements

This project is built upon [UDAStrongBaseline](https://github.com/zkcys001/UDAStrongBaseline), [LoRA](https://github.com/microsoft/LoRA) .

## :pencil2: Citation

If you think this project is helpful, please feel free to leave a :star: and cite our paper:

```tex
@article{nehdi2025low,
  title={Low-Rank Expert Merging for Multi-Source Domain Adaptation in Person Re-Identification},
  author={Nehdi, Taha Mustapha and Mrabah, Nairouz and Belal, Atif and Pedersoli, Marco and Granger, Eric},
  journal={arXiv preprint arXiv:2508.06831},
  year={2025}
}
```

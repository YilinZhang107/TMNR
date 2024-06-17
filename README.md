# TMNR
This repository contains the code of our paper 
```
Cai Xu, Yilin Zhang, Ziyu Guan, and Wei Zhao. "Trusted Multi-view Learning with Label Noise."
In Proceedings of the 33rd International Joint Conference on Artificial Intelligence, IJCAI. 2024.
```

# Methodology for Generating IDN Labels
Section B.1 of the Appendix within the repository describes how new labels are selected for samples that will have their labels destroyed.

The process of selecting the samples of to-be-destroyed labels is as follows: firstly, the entire clean dataset is predicted using the evidence classifier $g(\cdot)$ trained on the clean dataset mentioned in section B.1 of Appendix. The prediction uncertainty $u$ is obtained for each sample, and after that, using uncertainty as a weight (which indicates that a high level of uncertainty is more likely to be destructive), different proportions of to-be-destroyed labels needed for the experiment are selected.

In our experiments, using the **UCI** dataset as an example, 93.5% of the corrupted samples in the final generated dataset containing 10% instance-dependent noise labels come from the top 10% of the uncertainty ranking mentioned above, and 72.67% of the corrupted samples in the dataset containing 30% instance-dependent noise labels come from the top 30% of the uncertainty ranking.

# More Datasets
You can find more processed multi-view feature datasets from this repository: https://github.com/YilinZhang107/Multi-view-Datasets

# Citation
If you find our work helpful, please consider citing the following papers
```
@misc{xu2024trusted,
      title={Trusted Multi-view Learning with Label Noise}, 
      author={Cai Xu and Yilin Zhang and Ziyu Guan and Wei Zhao},
      year={2024},
      eprint={2404.11944},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

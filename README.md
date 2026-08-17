> **新接入的机器（site B）请先读交接文档**：
> [`docs/20260817-1440-交接协作文档-siteB接手指南.md`](docs/20260817-1440-交接协作文档-siteB接手指南.md)
> 它自带术语定义、环境准备（27K 数据需自建）、五条纪律、任务队列与回报模板。
> 仓库总规则见 [`AGENTS.md`](AGENTS.md)（§1.1 文档规范 / §2.1 两端协作隔离）。
>
> 下面是原始 KuaiRand-1K 复现说明（历史内容，当前实验阶段已切到 27K）。

Set up environment:
```bash
conda create python=3.13 -n LFM4Ads
conda activate LFM4Ads
pip install torch -i https://download.pytorch.org/whl/cu121
pip install torcheval tqdm pandas pyarrow matplotlib
```

Set up dataset:
```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz
md5sum KuaiRand-1K.tar.gz # 6b0b9c8222d67fcd4c676218edca3f1f
tar -xzvf KuaiRand-1K.tar.gz
python dataset.py
```

Run a trial:
```bash
python main.py cuda:0
```
Modify `cuda:0` to use another device.
More runs for more trials.
All AUCs will be appended to `result.csv`.
The AUC in our paper is averaged over 100 trials.

Visualize the average AUC:
```bash
python plot.py
```
Figures will be saved in `Feature\`, `Module\`, and `Model\`.

Welcome to cite our paper:
```bibtex
@misc{zhang2025largefoundationmodelads,
    title={Large Foundation Model for Ads Recommendation},
    author={Shangyu Zhang and Shijie Quan and Zhongren Wang and Junwei Pan and Tianqu Zhuang and Bo Fu and Yilong Sun and Jieying Lin and Jushuo Chen and Xiaotian Li and Zhixiang Feng and Xian Hu and Huiting Deng and Hua Lu and Jinpeng Wang and Boqi Dai and Xiaoyu Chen and Bin Hu and Lili Huang and Yanwen Wu and Yeshou Cai and Qi Zhou and Huang Tang and Chunfeng Yang and Chengguo Yin and Tingyu Jiang and Lifeng Wang and Shudong Huang and Dapeng Liu and Lei Xiao and Haijie Gu and Shu-Tao Xia and Jie Jiang},
    year={2025},
    eprint={2508.14948},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/2508.14948},
}
```

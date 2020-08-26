# PairTrading

This repository releases the code for Hedging via Opinion-based Pair Trading Strategy.

Ting-Wei Hsu, Chung-Chi Chen, Hen-Hsen Huang, Meng Chang Chen, and Hsin-Hsi Chen. 2020. [Hedging via Opinion-based Pair Trading Strategy](https://dl.acm.org/doi/10.1145/3366424.3382701). *In Companion Proceedings of the Web Conference* .

>Risk is an important component when constructing a trading strategy. However, most of the previous works that make the market movement prediction on the basis of the opinions on social media platforms do not take the risk into consideration. In order to hedge the market- and sector-risk, we propose an idea of an opinion-based pair trading strategy. Comparing with the task setting of the previous works, our experimental results show that the neural network models with the pair-wise task setting perform better in both accuracy and profitability metrics. That introduces a new research direction for future researches on market movement predictions.

In this paper, we experiment on the benchmark dataset, [StockNet](https://github.com/yumoxu/stocknet-dataset), which is collected from Twitter. 


## Code

1. `GenData.py`: Separate the dataset into the training set, validation set, and test set for each sector.

2. Train the models via `NN-PW/train.py` `(NN-IND/train.py)` for pair trading setting (price movement prediction setting).

3. `test.py` provides the predictions of test data.

4. Use `result.py` to evaluate the results.

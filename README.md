# PairTrading

This repository releases the code for Hedging via Opinion-based Pair Trading Strategy.

Ting-Wei Hsu, Chung-Chi Chen, Hen-Hsen Huang, Meng Chang Chen, and Hsin-Hsi Chen. 2020. Hedging via Opinion-based Pair Trading Strategy. In . ACM, New York, NY, USA, 2 pages.

>Risk is an important component when constructing a trading strategy. However, most of the previous works that make the market movement prediction on the basis of the opinions on social media platforms do not take the risk into consideration. In order to hedge the market- and sector-risk, we propose an idea of an opinion-based pair trading strategy. Comparing with the task setting of the previous works, our experimental results show that the neural network models with the pair-wise task setting perform better in both accuracy and profitability metrics. That introduces a new research direction for future researches on market movement predictions.

In this paper, we experiment on the benchmark dataset, [StockNet](https://github.com/yumoxu/stocknet-dataset), which is collected from Twitter. 


## Code

1. Generate the dataset of Train, Validation and Test for proposed sector in `GenData.py`.
2. Train the proposed NN model directly in `NN-PW/train.py` (`NN-IND/train.py`) for pair trading (price movement prediction) task.
3. `test.py` can show the predicted values and predicted labels of the data in Test when creating the final prediction file.
4. Compute the accuracy and cumulative profit in `result.py`

# Data-Mining Project

## Datasets
* Wine Reviews: 它是130k个葡萄酒的评论数据集，其中包含了葡萄酒的品种，价格，产地，酿酒厂和描述等信息。作者大概在三年前最后一次更新它，他的原意是使用深度学习技术通过品酒师的描述来预测葡萄酒的品种，酿酒厂，位置等信息，得到一个表现不错的文本相关的预测模型。这个数据集包含两个csv文件，分别含有130k和150k个葡萄酒评论，平均有2组整数属性，10组字符串属性，和1组时间属性。

* Melbourne Airbnb Open Data: 它收集了墨尔本城市Airbnb的房屋出租的交易数据，里面包含了游客信息，租客信息，房屋的经纬度，价格，评论，出租天数等内容。这个数据集包含7个csv文件，含有22.9k个交易相关数据，我集中分析了listings_summary_dec18.csv数据文件，里面整合了其他数据文件的信息以及包含一些度量指标，便于可视化。

## Instruction
* Running the second cell of data_analysis.ipynb, the result of Wine Reviews will be displayed. Note that if you want to show the similarity strategy used for filling nan value, change the input of nan_choice to 4 and edit data_handle.py, go to Data.relation and use mean value to subtitute the curve_fit method. There are two csv files in Wine Reviews dataset, for simplicity I only document the first csv's result in analytic report.

* Running the third cell of data_analysis.ipynb, the result of Melbourne Airbnb Open Data will be displayed. 

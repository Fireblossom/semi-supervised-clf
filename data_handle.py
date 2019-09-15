import pandas as pd


max_content = 10000

meta = pd.read_table("./ag_news_csv/metadata", sep="\t", header=None)
content = pd.read_table("./ag_news_csv/reviewContent", sep="\t", header=None)


join_data = pd.merge(meta, content, on=[0, 1], how="inner").loc[:max_content, :]
join_data = join_data.iloc[:, [3, -1]]
join_data.to_csv("./balance_reviews.csv", header=None, index=None)
# make imbalance data splition
# sample 80% reviews as unlabel information
unsuper_reviews = join_data.sample(frac=0.8)
unsuper_reviews.to_csv("unsuper_reviews.csv", header=None, index=None)
#
th = ~join_data.index.isin(unsuper_reviews.index)
super_reviews = join_data[th]
real = super_reviews[super_reviews.iloc[:, 0] == 1]
fake = super_reviews[super_reviews.iloc[:, 0] == -1]
real.to_csv("./real_reviews.csv", header=None, index=None)
fake.to_csv("./fake_reviews.csv", header=None, index=None)


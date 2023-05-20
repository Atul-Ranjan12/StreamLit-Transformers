import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# Set up File Paths
articles = "./articles.csv"
interactions = "./users_interactions.csv"


class ContentBasedRecommender:
    def __init__(self, article_path=articles):
        self.df = pd.read_csv(article_path)
        self.initialize_df("url", "title", "text")
        self.count = TfidfVectorizer(analyzer="word",
                                     ngram_range=(1, 2),
                                     min_df=0.003,
                                     max_df=0.5,
                                     max_features=5000,
                                     stop_words="english")
        self.count_matrix = self.fit()
        self.df = self.df.reset_index()
        self.titles = self.df["contentId"]
        self.indices = pd.Series(self.df.index, index=self.df["contentId"])
        # Calculate Cosine Similarity
        self.cosine_similarity = self.calculate_cosine_similarity()

    def initialize_df(self, col1, col2, col3):
        self.df["soup"] = self.df[col1] + self.df[col2] + self.df[col3]

    def fit(self):
        return self.count.fit_transform(self.df["soup"])

    def calculate_cosine_similarity(self):
        return cosine_similarity(self.count_matrix, self.count_matrix)

    def get_recommendations(self, title, topn):
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_similarity[idx]))
        sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:31]
        article_indices = [i[0] for i in sim_scores]
        return list(self.titles.iloc[article_indices])[:topn]

    def get_articles(self, title, topn):
        recommended_articles = self.get_recommendations(title, topn)
        recommended_articles_df = self.df[self.df["contentId"].isin(recommended_articles)]
        recs = []
        for title, text in zip(recommended_articles_df["title"], recommended_articles_df["text"]):
            recs.append({"title": title,
                         "text": text})

        return recs

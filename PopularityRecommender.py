import pandas as pd
import collections

# Set up some file paths
articles = "./articles.csv"
interactions = "./users_interactions.csv"


def count_frequency(arr):
    return collections.Counter(arr)


class PopularityRecommender:
    def __init__(self, articles_path=articles, interactions_path=interactions):
        self.articles = pd.read_csv(articles_path)
        self.interactions = pd.read_csv(interactions_path)
        self.ref_df = self.articles

        self.add_event_strength()
        self.pop_df = self.make_popularity_df()

    def add_event_strength(self):
        event_strength = {
            'VIEW': 1.0,
            'LIKE': 2.0,
            'BOOKMARK': 2.5, 
            'FOLLOW': 3.0,
            'COMMENT CREATED': 4.0,
        }
        self.interactions["eventStrength"] = self.interactions["eventType"].apply(lambda x: event_strength[x])

    def make_popularity_df(self):
        return self.interactions.groupby("contentId")["eventStrength"].sum().sort_values(ascending=False).reset_index()

    def make_recommendations(self, n=25):
        rec_df = self.pop_df.head(n)
        return rec_df.merge(self.ref_df, left_on="contentId", right_on="contentId")


class RecommendationsPopularity:
    def __init__(self, n=25, articles_path=articles, interaction_path=interactions):
        self.engine = PopularityRecommender();
        self.recommendation_df = self.engine.make_recommendations(n)
        self.articles = pd.read_csv(articles_path)
        self.interactions = pd.read_csv(interaction_path)
        self.index = self.prepare_index()

    def prepare_index(self):
        index = {}
        counter = 1
        for item in self.recommendation_df.columns:
            index[item] = counter
            counter += 1

        return index

    def get_items(self):
        recommendations = []
        for row in self.recommendation_df.itertuples():

            article_interactions = self.interactions.loc[self.interactions["contentId"] == row[self.index["contentId"]]]
            interactions_data = dict(count_frequency(article_interactions["eventType"]))

            recommendations.append({
                "contentId": row[self.index["contentId"]],
                "title": row[self.index["title"]],
                "content": row[self.index["text"]],
                "sentiment": row[self.index["sentiment"]],
                "interactions": interactions_data
            })
        return recommendations

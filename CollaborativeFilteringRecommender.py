import pandas as pd
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from PopularityRecommender import count_frequency

# Set up File Paths
articles = "./articles.csv"
interactions = "./users_interactions.csv"


def smooth_user_preference(x):
    return math.log(1+x, 2)


class CollaborativeFilteringRecommender:
    def __init__(self, article_path=articles, interactions_path=interactions):
        self.articles_df = pd.read_csv(article_path)
        self.interactions_df = pd.read_csv(interactions_path)
        self.add_event_strength()
        self.interaction_full_df = self.prepare_interactions_full()
        self.user_item_pivot_matrix_df = self.make_pivot_matrix()

        self.user_item_pivot_matrix = self.user_item_pivot_matrix_df.to_numpy()
        self.user_ids = list(self.user_item_pivot_matrix_df.index)
        self.user_item_pivot_sparse_matrix = csr_matrix(self.user_item_pivot_matrix)
        self.U, self.sigma, self.Vt = self.apply_svd(15)
        self.sigma = np.diag(self.sigma)

        self.all_user_predicted_ratings = np.dot(np.dot(self.U, self.sigma), self.Vt)
        self.user_predicted_ratings_norm = (self.all_user_predicted_ratings - self.all_user_predicted_ratings.min()) / (
                    self.all_user_predicted_ratings.max() - self.all_user_predicted_ratings.min())

        # Prepare Collaborative Filtering Prediction Matrix
        self.cf_predictions_df = pd.DataFrame(self.user_predicted_ratings_norm, columns=self.user_item_pivot_matrix_df.columns, index=self.user_ids).transpose()

    def prepare_interactions_full(self):
        users_interactions_count_df = self.interactions_df.groupby(['personId', 'contentId']).size().groupby(
            'personId').size()
        users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[
            ['personId']]
        # Fix interactions_df
        interactions_from_selected_users_df = self.interactions_df.merge(
            users_with_enough_interactions_df,
            how="right",
            left_on="personId",
            right_on="personId"
        )
        return interactions_from_selected_users_df.groupby(
            ["personId", "contentId"]
        )["eventStrength"].sum().apply(smooth_user_preference).reset_index()

    def add_event_strength(self):
        event_type_strength = {
            'VIEW': 1.0,
            'LIKE': 2.0,
            'BOOKMARK': 2.5,
            'FOLLOW': 3.0,
            'COMMENT CREATED': 4.0,
        }
        self.interactions_df["eventStrength"] = self.interactions_df["eventType"].apply(
            lambda x: event_type_strength[x]
        )

    def make_pivot_matrix(self):
        return self.interaction_full_df.pivot(index="personId", columns="contentId", values="eventStrength").fillna(0)

    def apply_svd(self, number_of_matrix_factors):
        U, sigma, Vt = svds(self.user_item_pivot_sparse_matrix, k=number_of_matrix_factors)
        return U, sigma, Vt

    def check_svd(self):
        print(f"Sigma shape: {self.sigma.shape}   u shape: {self.U.shape}    Vt shape: {self.Vt.shape}")

    def check_cf_matrix(self):
        print(self.cf_predictions_df)

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'recStrength'})

        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
            .sort_values('recStrength', ascending=False) \
            .head(topn)

        return recommendations_df

    def get_userIds(self):
        return self.user_ids


class CFProducts:
    def __init__(self, user_id, articles_path=articles, interactions_path=interactions):
        self.user = user_id
        self.articles_df = pd.read_csv(articles_path)
        self.interactions = pd.read_csv(interactions_path)
        self.recommendation_engine = CollaborativeFilteringRecommender()
        self.recommended_items = self.get_contentIds()
        self.recommendation_df = self.prepare_dataframe()
        self.recommendation_df_index = self.set_up_index()

    def add_event_strength(self):
        event_strength = {
            'VIEW': 1.0,
            'LIKE': 2.0,
            'BOOKMARK': 2.5,
            'FOLLOW': 3.0,
            'COMMENT CREATED': 4.0,
        }
        self.interactions["eventStrength"] = self.interactions["eventType"].apply(lambda x: event_strength[x])

    def get_contentIds(self):
        return list(self.recommendation_engine.recommend_items(self.user)["contentId"])

    def prepare_dataframe(self):
        return self.articles_df.loc[self.articles_df["contentId"].isin(self.recommended_items)]

    def check_recommendations(self):
        print(self.recommendation_df.head())

    def set_up_index(self):
        index = {}
        for i, col in enumerate(self.recommendation_df.columns):
            index[col] = i + 1
        return index

    def recommend_items(self):
        recommendations = []
        for row in self.recommendation_df.itertuples():

            article_interactions = self.interactions.loc[self.interactions["contentId"] == row[self.recommendation_df_index["contentId"]]]
            interactions_data = dict(count_frequency(article_interactions["eventType"]))

            recommendations.append({
                "contentId": row[self.recommendation_df_index["contentId"]],
                "title": row[self.recommendation_df_index["title"]],
                "content": row[self.recommendation_df_index["text"]],
                "sentiment": row[self.recommendation_df_index["sentiment"]],
                "interactions": interactions_data
            })

        return recommendations

    def get_prev_article_titles(self, topn):
        content_ids = list(self.interactions.loc[
            self.interactions["personId"] == self.user
        ]["contentId"])
        titles = list(
            self.articles_df.loc[self.articles_df["contentId"].isin(content_ids)]["title"]
        )
        return titles[:topn]

    def check_titles(self):
        print("Printing Titles:")
        print(self.get_prev_article_titles(10))

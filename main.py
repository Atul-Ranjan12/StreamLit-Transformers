import pandas as pd
from PopularityRecommender import RecommendationsPopularity
from ContentBasedRecommender import ContentBasedRecommender
from CollaborativeFilteringRecommender import CollaborativeFilteringRecommender, CFProducts
import streamlit as st

# Manage Popularity based Recommendations
recommendations = RecommendationsPopularity(50)
recommended_articles = recommendations.get_items()
articles = pd.read_csv("./articles.csv")

# Manage Content based Recommendations
content_based_recommender = ContentBasedRecommender()

# Manage Collaborative Filtering Recommendations:
collaborative_filtering_engine = CollaborativeFilteringRecommender()
user_ids = collaborative_filtering_engine.get_userIds()

# Styles for the streamlit components:
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(204, 49, 49);
    color: white;
}
</style>""", unsafe_allow_html=True)

st.write("""
# CI and T Deskdrop Article Recommendations:
""")


def display_content(content, max_length):
    if len(content) > max_length:
        content = content[:max_length] + "..."
    st.write(content)


def view_more_button_handler(title, content):
    st.empty()
    article_title = f"#### {title}"
    article_content = content
    return article_title, article_content


def display_articles(articles, max_length, key_int, call):
    if call:
        st.write("""
        # Trending Article Recommendations for Today:
        """)
    else:
        st.write("""
        # Recommended Articles For you:
        """)
    for article in articles:
        key_int += 1
        st.markdown(f"### {article['title']}")
        display_content(article["content"], max_length)

        interaction_data = article["interactions"]
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if "VIEW" in interaction_data:
                st.write(f"VIEWS: ", interaction_data["VIEW"])
        with c2:
            if "FOLLOW" in interaction_data:
                st.write(f"FOLLOWS: ", interaction_data["FOLLOW"])
        with c3:
            if "BOOKMARK" in interaction_data:
                st.write(f"BOOKMARKS: ", interaction_data["BOOKMARK"])
        with c4:
            if "LIKE" in interaction_data:
                st.write(f"LIKES: ", interaction_data["LIKE"])
        with c5:
            if "COMMENT CREATED" in interaction_data:
                st.write(f"COMMENTS: ", interaction_data["COMMENT CREATED"])

        col1, col2 = st.columns(2)

        article_title = ""
        article_content = ""

        sentiments = []

        flag = 0
        with col1:
            if st.button("View More", key=f"ViewFullArticle{key_int}"):
                article_title, article_content = view_more_button_handler(article["title"],
                                                                          article["content"])
                flag = 1

        st.write(article_title)
        if flag == 1:
            st.write(article_content)

        similar_article_recommendations = content_based_recommender.get_articles(article["contentId"], 3)

        st.markdown(f"#### Similar articles you might be interested in:")
        col1, col2, col3 = st.columns(3)
        with col1:
            similar_article_title = similar_article_recommendations[0]["title"]
            st.markdown(f"###### {similar_article_title}")
            display_content(similar_article_recommendations[0]["text"], 100)
        with col2:
            similar_article_title2 = similar_article_recommendations[1]["title"]
            st.markdown(f"###### {similar_article_title2}")
            display_content(similar_article_recommendations[1]["text"], 100)
        with col3:
            similar_article_title3 = similar_article_recommendations[2]["title"]
            st.markdown(f"###### {similar_article_title3}")
            display_content(similar_article_recommendations[2]["text"], 100)


def display_personalized_recommendations(uids):
    selected_id = st.selectbox("Select a userID for personalized recommendations:", uids)
    cf_recommender = CFProducts(selected_id)
    prev_articles = cf_recommender.get_prev_article_titles(4)
    st.write("""
    #### Previously Viewed Articles:
    """)

    for title in prev_articles:
        st.markdown(f"###### {title}")

    cf_recommendations = cf_recommender.recommend_items()
    display_articles(cf_recommendations, 200, 5234, False)


display_personalized_recommendations(user_ids)
display_articles(recommended_articles, 200, 0, True)
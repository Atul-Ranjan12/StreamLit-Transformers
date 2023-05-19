import pandas as pd
from PopularityRecommender import RecommendationsPopularity
from ContentBasedRecommender import ContentBasedRecommender
from CollaborativeFilteringRecommender import CollaborativeFilteringRecommender, CFProducts
import streamlit as st
from transformers import pipeline

# Set up Transformers
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model", max_length=2048)
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
analyzer = pipeline(model="roberta-large-mnli")


# Function to set up sentiment analysis for parts of the text
def get_sentiment(text, max_len=512, classifier=analyzer):
    sentiment_array = []
    counter = 0
    for i in range(max_len, len(text), max_len):
        sentiment = classifier(text[counter: i])
        sentiment_array.append({sentiment[0]["label"]: round(sentiment[0]["score"], 4)})
        counter = i
    return sentiment_array


def display_content_with_sentiment(text, sentiment_array, max_len=512):
    counter, index = 0, 0
    for i in range(max_len, len(text), max_len):
        for sentiment in sentiment_array[index].keys():
            if sentiment == "NEUTRAL":
                st.markdown(
                    f'<div><span style="border-radius: 1em 0 1em 0; text-shadow: 1px 1px 1px #fff; background-image: linear-gradient(90deg, rgba(2,0,36,1) 0%, rgba(185,255,169,1) 0%, rgba(220,255,203,1) 100%, rgba(242,255,247,1) 100%);">{text[counter: i]}</span></div>',
                    unsafe_allow_html=True)
                break
            elif sentiment == "ENTAILMENT":
                st.markdown(
                    f'<div><span style="border-radius: 1em 0 1em 0; text-shadow: 1px 1px 1px #fff; background-image: linear-gradient(90deg, rgba(2,0,36,1) 0%, rgba(252,255,169,1) 0%, rgba(255,254,203,1) 100%, rgba(242,255,247,1) 100%);">{text[counter: i]}</span></div>',
                    unsafe_allow_html=True)
                break
            elif sentiment == "CONTRADICTION":
                st.markdown(
                    f'<div><span style="border-radius: 1em 0 1em 0; text-shadow: 1px 1px 1px #fff; background-image: linear-gradient(90deg, rgba(2,0,36,1) 0%, rgba(255,225,225,1) 0%, rgba(255,229,229,1) 100%, rgba(242,255,247,1) 100%);">{text[counter: i]}</span></div>',
                    unsafe_allow_html=True)
                break
        counter = i
        index += 1


# Function to summarize the articles
def summarize_article(text):
    result = summarizer(text)
    return result[0]["summary_text"]


# Manage Popularity based Recommendations
recommendations = RecommendationsPopularity(50)
recommended_articles = recommendations.get_items()
articles = pd.read_csv("/Users/atulranjan/PycharmProjects/streamLitRecommenderSystem/articles.csv")

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
    sentiments = get_sentiment(article_content)
    return article_title, article_content, sentiments


def summary_button_handler(content):
    st.empty()
    article_title = f"#### Summary of the Article:"
    article_content = summarize_article(content)
    sentiments = get_sentiment(article_content)
    return article_title, article_content, sentiments


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
                article_title, article_content, sentiments = view_more_button_handler(article["title"],
                                                                                      article["content"])
                flag = 1
        with col2:
            if st.button("View Summary", key=f"Summarize{key_int}"):
                article_title, article_content, sentiments = summary_button_handler(article["content"])
                flag = 2

        st.write(article_title)
        if flag == 1:
            display_content_with_sentiment(article_content, sentiment_array=sentiments)
        elif flag == 2:
            st.write(article_content)

        st.markdown(f"#### Ask Questions about the article:")
        question = st.text_input("Enter your question about the article", key=f"question{key_int}")
        if question:
            answer = qa_model(question=question, context=article["content"])
            question_answer = answer["answer"]
            st.write(f"""
            #### Answer to the question:
            {question_answer}
            """)

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

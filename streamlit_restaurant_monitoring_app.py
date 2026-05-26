import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).with_name("Restaurant_Reviews.tsv")
MODEL_PATH = Path(__file__).with_name("restaurant_sentiment_pipeline.joblib")

CUSTOM_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "is", "am", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "at", "by", "from", "this", "that", "these", "those", "it", "its",
    "as", "so", "very", "just", "than", "then", "there", "here", "i", "we", "you", "he", "she", "they", "them",
    "my", "our", "your", "his", "her", "their", "me", "us", "do", "does", "did", "have", "has", "had"
}
NEGATION_WORDS = {"no", "not", "nor", "never", "none", "cannot", "can't", "dont", "don't", "didnt", "didn't", "isnt", "isn't", "wasnt", "wasn't", "wont", "won't"}
STOPWORDS = CUSTOM_STOPWORDS - NEGATION_WORDS


def clean_review(text: str) -> str:
    "Clean review text while preserving negation words that matter for sentiment."
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z' ]", " ", text)
    tokens = [token.strip(string.punctuation) for token in text.split()]
    tokens = [token for token in tokens if token and token not in STOPWORDS and len(token) > 1]
    return " ".join(tokens)


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, sep="\t")
    data["clean_review"] = data["Review"].apply(clean_review)
    data["review_length"] = data["Review"].astype(str).str.split().str.len()
    return data


@st.cache_resource
def train_or_load_pipeline():
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data["Review"], data["Liked"], test_size=0.2, random_state=42, stratify=data["Liked"]
    )
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(preprocessor=clean_review, max_features=1500, ngram_range=(1, 2))),
        ("model", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test).max(axis=1)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Avg Confidence": float(y_prob.mean()),
        "False Positives": int(fp),
        "False Negatives": int(fn),
    }
    baseline = {
        "train_avg_length": float(data["review_length"].mean()),
        "train_p95_length": float(data["review_length"].quantile(0.95)),
        "positive_rate": float(data["Liked"].mean()),
    }
    return pipeline, metrics, baseline


def prediction_frame(pipeline, reviews):
    probabilities = pipeline.predict_proba(reviews)
    predictions = pipeline.predict(reviews)
    rows = []
    for review, pred, prob in zip(reviews, predictions, probabilities):
        confidence = float(prob[pred])
        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "review": review,
            "clean_review": clean_review(review),
            "prediction": "Positive" if pred == 1 else "Negative",
            "predicted_class": int(pred),
            "confidence": round(confidence, 4),
            "review_length": len(str(review).split()),
            "needs_human_review": confidence < 0.65,
        })
    return pd.DataFrame(rows)


def top_terms(texts, n=12):
    tokens = " ".join(clean_review(text) for text in texts).split()
    return pd.DataFrame(Counter(tokens).most_common(n), columns=["term", "count"])


def show_metric_cards(metrics, baseline):
    cols = st.columns(4)
    cols[0].metric("Model accuracy", f"{metrics['Accuracy']:.1%}")
    cols[1].metric("F1 score", f"{metrics['F1 Score']:.1%}")
    cols[2].metric("Average confidence", f"{metrics['Avg Confidence']:.1%}")
    cols[3].metric("Training positive rate", f"{baseline['positive_rate']:.1%}")


def monitoring_dashboard(log_df: pd.DataFrame, baseline: dict):
    if log_df.empty:
        st.info("No live predictions yet. Enter a review or upload a small batch to populate the monitoring dashboard.")
        return

    positive_rate = log_df["predicted_class"].mean()
    low_conf_rate = log_df["needs_human_review"].mean()
    avg_length = log_df["review_length"].mean()
    drift_flag = abs(avg_length - baseline["train_avg_length"]) > 0.5 * baseline["train_avg_length"]

    cols = st.columns(4)
    cols[0].metric("Live predictions", len(log_df))
    cols[1].metric("Live positive rate", f"{positive_rate:.1%}")
    cols[2].metric("Low confidence rate", f"{low_conf_rate:.1%}")
    cols[3].metric("Avg review length", f"{avg_length:.1f} words")

    if drift_flag:
        st.warning("Input length drift detected: recent reviews are much shorter or longer than the training baseline.")
    if low_conf_rate > 0.25:
        st.warning("More than 25% of recent predictions need human review. This is a useful retraining or threshold-tuning signal.")

    left, right = st.columns(2)
    with left:
        st.subheader("Prediction mix")
        mix = log_df["prediction"].value_counts().rename_axis("sentiment").reset_index(name="count")
        st.bar_chart(mix, x="sentiment", y="count")
    with right:
        st.subheader("Confidence over time")
        confidence_trend = log_df.reset_index().rename(columns={"index": "request_number"})[["request_number", "confidence"]]
        st.line_chart(confidence_trend, x="request_number", y="confidence")

    st.subheader("Recent prediction log")
    st.dataframe(log_df.tail(20).sort_index(ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Top words in recent traffic")
    st.dataframe(top_terms(log_df["review"].tail(50)), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="Restaurant Review Sentiment Monitor", page_icon=":knife_fork_plate:", layout="wide")
    st.title("Restaurant Review Sentiment Monitor")
    st.caption("A deployment-style Streamlit app with live prediction logging, quality checks, and monitoring signals.")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}. Keep Restaurant_Reviews.tsv in the same folder as this app.")
        st.stop()

    pipeline, model_metrics, training_baseline = train_or_load_pipeline()
    show_metric_cards(model_metrics, training_baseline)

    if "prediction_log" not in st.session_state:
        st.session_state.prediction_log = pd.DataFrame(columns=[
            "timestamp", "review", "clean_review", "prediction", "predicted_class", "confidence", "review_length", "needs_human_review"
        ])

    st.divider()
    left_panel, right_panel = st.columns([1, 1])

    with left_panel:
        st.subheader("Single review prediction")
        sample_review = st.text_area("Customer review", "The food was delicious, but the service was very slow.", height=120)
        if st.button("Predict review", type="primary"):
            result = prediction_frame(pipeline, [sample_review])
            st.session_state.prediction_log = pd.concat([st.session_state.prediction_log, result], ignore_index=True)
            label = result.loc[0, "prediction"]
            confidence = result.loc[0, "confidence"]
            st.success(f"Prediction: {label} with {confidence:.1%} confidence")
            if result.loc[0, "needs_human_review"]:
                st.warning("Confidence is below 65%, so this prediction should be reviewed by a human.")

    with right_panel:
        st.subheader("Batch scoring")
        uploaded = st.file_uploader("Upload CSV/TSV with a Review column", type=["csv", "tsv"])
        if uploaded is not None:
            sep = "\t" if uploaded.name.endswith(".tsv") else ","
            batch = pd.read_csv(uploaded, sep=sep)
            if "Review" not in batch.columns:
                st.error("The uploaded file must contain a column named Review.")
            else:
                scored = prediction_frame(pipeline, batch["Review"].astype(str).tolist())
                st.session_state.prediction_log = pd.concat([st.session_state.prediction_log, scored], ignore_index=True)
                st.dataframe(scored, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download scored predictions",
                    scored.to_csv(index=False).encode("utf-8"),
                    "scored_restaurant_reviews.csv",
                    "text/csv"
                )

    st.divider()
    st.header("Monitoring dashboard")
    monitoring_dashboard(st.session_state.prediction_log, training_baseline)

    with st.expander("Model and monitoring notes"):
        st.write(
            "This app retrains a compact logistic-regression pipeline on startup, saves a joblib artifact, "
            "and tracks live predictions in the Streamlit session. In production, the prediction log would be "
            "written to a database or observability platform, and the drift checks would compare live traffic "
            "against a stable training baseline."
        )


if __name__ == "__main__":
    main()


# print("Run this command in a terminal:")
# print("streamlit run streamlit_restaurant_monitoring_app.py")
#or
print("python -m streamlit run streamlit_restaurant_monitoring_app.py")
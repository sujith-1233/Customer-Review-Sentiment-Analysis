import streamlit as st
import pandas as pd
import plotly.express as px

from src.pipeline import (
    train_and_evaluate,
    predict_reviews,
    summarize_reviews,
    load_demo_data
)

# Page Configuration
st.set_page_config(
    page_title="ReviewSense | Customer Sentiment",
    page_icon="💬",
    layout="wide"
)

st.title("💬 ReviewSense — Customer Review Sentiment Analysis")
st.caption("NLP + Machine Learning using TF-IDF, Word2Vec and Sentiment Classification.")

# Sidebar
with st.sidebar:
    st.header("Project Controls")

    page = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Analyze Reviews",
            "Model Evaluation",
            "Project Guide"
        ]
    )

    st.markdown("---")
    st.info("CSV must contain a review column and optionally a sentiment column.")

# Load demo dataset
@st.cache_data
def get_demo():
    return load_demo_data()

# Session State
if "model_bundle" not in st.session_state:
    st.session_state.model_bundle = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None

if "data" not in st.session_state:
    st.session_state.data = get_demo()


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.subheader("Customer Feedback Overview")

    df = st.session_state.data.copy()

    if "sentiment" not in df.columns:

        st.warning("No sentiment labels found.")

    else:

        counts = df["sentiment"].astype(str).str.lower().value_counts()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Reviews", f"{len(df):,}")
        col2.metric("Positive Reviews", f"{counts.get('positive', 0):,}")
        col3.metric("Negative Reviews", f"{counts.get('negative', 0):,}")

        left, right = st.columns(2)

        with left:

            fig = px.pie(
                names=counts.index,
                values=counts.values,
                title="Sentiment Distribution",
                hole=0.42
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="dashboard_pie_chart"
            )

        with right:

            fig = px.bar(
                x=counts.index,
                y=counts.values,
                labels={
                    "x": "Sentiment",
                    "y": "Reviews"
                },
                title="Review Counts"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="dashboard_bar_chart"
            )

        st.subheader("Recent Customer Reviews")

        st.dataframe(
            df.head(12),
            use_container_width=True
        )

    st.caption("ReviewSense Customer Sentiment Analytics")


# =====================================================
# ANALYZE REVIEWS
# =====================================================

elif page == "Analyze Reviews":

    st.subheader("Upload and Analyze Customer Reviews")

    uploaded = st.file_uploader(
        "Upload Customer Reviews CSV",
        type=["csv"]
    )

    if uploaded:

        try:

            uploaded_df = pd.read_csv(uploaded)

            if "review" not in uploaded_df.columns:

                st.error("CSV must contain a 'review' column.")

            else:

                st.session_state.data = uploaded_df

                st.success(
                    f"Successfully loaded {len(uploaded_df)} reviews."
                )

                st.dataframe(
                    uploaded_df.head(),
                    use_container_width=True
                )

        except Exception as e:

            st.error(f"CSV Error: {e}")

    df = st.session_state.data

    # Train Models
    if "sentiment" in df.columns and df["sentiment"].notna().any():

        if st.button("Train and Compare Models", type="primary"):

            with st.spinner("Training machine learning models..."):

                try:

                    bundle, metrics = train_and_evaluate(df)

                    st.session_state.model_bundle = bundle
                    st.session_state.metrics = metrics

                    st.success("Model training completed successfully!")

                    st.write(
                        "Best Model:",
                        bundle["best_model"]
                    )

                except Exception as e:

                    st.error(f"Training Error: {e}")

    # Single Review Prediction
    st.markdown("---")

    st.subheader("Single Review Sentiment Prediction")

    review_text = st.text_area(
        "Enter Customer Review",
        placeholder="The product quality is excellent and delivery was fast."
    )

    if st.button("Analyze Sentiment"):

        if not review_text.strip():

            st.warning("Please enter a review.")

        else:

            if st.session_state.model_bundle is None:

                if "sentiment" not in df.columns:

                    st.error("Upload labelled data before training.")

                else:

                    with st.spinner("Training models..."):

                        try:

                            bundle, metrics = train_and_evaluate(df)

                            st.session_state.model_bundle = bundle
                            st.session_state.metrics = metrics

                        except Exception as e:

                            st.error(str(e))

            if st.session_state.model_bundle is not None:

                result = predict_reviews(
                    [review_text],
                    st.session_state.model_bundle
                )

                row = result.iloc[0]

                col1, col2 = st.columns(2)

                col1.metric(
                    "Predicted Sentiment",
                    row["predicted_sentiment"].title()
                )

                col2.metric(
                    "Confidence",
                    f"{row['confidence']:.1%}"
                )

                st.write("Customer Review:", review_text)

    # Batch Prediction
    if st.session_state.model_bundle is not None:

        st.markdown("---")

        st.subheader("Bulk Review Prediction")

        batch = st.text_area(
            "Enter multiple reviews (one per line)",
            height=150
        )

        if st.button("Predict All Reviews"):

            items = [
                x.strip()
                for x in batch.splitlines()
                if x.strip()
            ]

            if items:

                output = predict_reviews(
                    items,
                    st.session_state.model_bundle
                )

                st.dataframe(
                    output,
                    use_container_width=True
                )

                st.download_button(
                    "Download Predictions CSV",
                    output.to_csv(index=False).encode("utf-8"),
                    "sentiment_predictions.csv",
                    "text/csv"
                )


# =====================================================
# MODEL EVALUATION
# =====================================================

elif page == "Model Evaluation":

    st.subheader("Machine Learning Model Evaluation")

    metrics = st.session_state.metrics

    if not metrics:

        st.info(
            "Go to Analyze Reviews and train the models first."
        )

    else:

        st.subheader("Performance Metrics")

        metric_df = pd.DataFrame(metrics).T

        st.dataframe(
            metric_df.style.format("{:.3f}"),
            use_container_width=True
        )

        # Accuracy Comparison
        accuracy_df = metric_df.reset_index()
        accuracy_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]

        fig = px.bar(
            accuracy_df,
            x="Model",
            y="Accuracy",
            title="Model Accuracy Comparison",
            range_y=[0, 1]
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="evaluation_accuracy_chart"
        )

        # Confusion Matrices
        st.subheader("Confusion Matrices")

        for model_name, result in st.session_state.model_bundle["evaluation"].items():

            st.markdown(f"### {model_name}")

            cm = result["confusion_matrix"]

            fig = px.imshow(
                cm,
                text_auto=True,
                x=["Predicted Negative", "Predicted Positive"],
                y=["Actual Negative", "Actual Positive"],
                title=f"{model_name} Confusion Matrix",
                color_continuous_scale="Blues"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"confusion_matrix_{model_name.replace(' ', '_')}"
            )

        st.caption(
            "Metrics are calculated using a held-out test dataset."
        )


# =====================================================
# PROJECT GUIDE
# =====================================================

else:

    st.subheader("Project Documentation")

    st.markdown("""
    ### Project Aim

    To develop an NLP-based system that automatically analyzes
    customer reviews and classifies opinions into positive
    and negative sentiments.

    ### NLP Techniques

    - Text Preprocessing
    - Tokenization
    - TF-IDF
    - Word2Vec

    ### Machine Learning Algorithms

    1. Logistic Regression
    2. Naive Bayes
    3. Support Vector Machine

    ### Evaluation Metrics

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix

    ### Dataset Format

    | review | sentiment |
    |---|---|
    | Excellent product | positive |
    | Poor quality | negative |

    ### Future Enhancements

    - Neutral Sentiment Classification
    - Emotion Detection
    - AI Review Summarization
    - Product Comparison
    - PDF Report Generation
    """)

    st.subheader("Customer Opinion Summary")

    st.write(
        summarize_reviews(st.session_state.data)
    )

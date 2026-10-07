# ============================================================
# FOREST COVER TYPE PREDICTION APP
# LightGBM + SHAP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Forest Cover Type Predictor",
    page_icon="🌲",
    layout="wide"
)


# ============================================================
# 2. LOAD SAVED ARTIFACTS
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "final_lightgbm_forest_cover_type.pkl"
    )

    return model


@st.cache_resource
def load_explainer():

    explainer = joblib.load(
        "lightgbm_shap_explainer.pkl"
    )

    return explainer


@st.cache_data
def load_feature_names():

    feature_names = joblib.load(
        "feature_names.pkl"
    )

    return feature_names


@st.cache_data
def load_class_labels():

    class_labels = joblib.load(
        "class_labels.pkl"
    )

    return class_labels


# ============================================================
# 3. LOAD ARTIFACTS
# ============================================================

try:

    model = load_model()
    explainer = load_explainer()
    feature_names = load_feature_names()
    class_labels = load_class_labels()

except Exception as e:

    st.error(
        "Could not load the required model files."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 4. VALIDATE FEATURE NAMES
# ============================================================

if not isinstance(feature_names, list):

    feature_names = list(feature_names)


if len(feature_names) != model.n_features_in_:

    st.error(
        f"Feature mismatch detected.\n\n"
        f"feature_names.pkl contains {len(feature_names)} "
        f"features, but the model expects "
        f"{model.n_features_in_} features."
    )

    st.stop()


# ============================================================
# 5. TITLE
# ============================================================

st.title("🌲 Forest Cover Type Prediction")

st.markdown(
    """
    This application uses a **Balanced + Tuned LightGBM**
    model to predict the forest cover type from environmental
    and geographic features.

    The model was selected using **Macro F1** and achieved a
    final test Macro F1 of **0.9261**.
    """
)


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.header("Model Information")

st.sidebar.write(
    "**Model:** Balanced + Tuned LightGBM"
)

st.sidebar.write(
    "**Number of features:**",
    len(feature_names)
)

st.sidebar.write(
    "**Number of classes:**",
    len(class_labels)
)

st.sidebar.write(
    "**Test Macro F1:** 0.9261"
)


# ============================================================
# 7. INPUT METHOD
# ============================================================

st.header("Input Data")

input_method = st.radio(
    "Choose input method:",
    [
        "Manual Input",
        "CSV Upload"
    ],
    horizontal=True
)


# ============================================================
# 8. CREATE INPUT DATAFRAME
# ============================================================

input_data = None


# ============================================================
# OPTION A: CSV INPUT
# ============================================================

if input_method == "CSV Upload":

    st.subheader("Upload CSV")

    uploaded_file = st.file_uploader(
        "Upload a CSV containing the model features",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            st.write("Uploaded data preview:")

            st.dataframe(
                uploaded_df.head()
            )

            missing_features = [
                feature
                for feature in feature_names
                if feature not in uploaded_df.columns
            ]

            extra_features = [
                column
                for column in uploaded_df.columns
                if column not in feature_names
            ]

            if missing_features:

                st.error(
                    "The uploaded CSV is missing required "
                    "features."
                )

                st.write(
                    "Missing features:",
                    missing_features
                )

                st.stop()

            # Keep ONLY the exact model feature order
            input_data = uploaded_df[
                feature_names
            ].copy()

            # Convert to numeric
            input_data = input_data.apply(
                pd.to_numeric,
                errors="coerce"
            )

            if input_data.isnull().any().any():

                st.error(
                    "The uploaded data contains missing or "
                    "non-numeric values in the model features."
                )

                st.dataframe(
                    input_data[
                        input_data.isnull().any(axis=1)
                    ]
                )

                st.stop()

            st.success(
                f"{len(input_data)} row(s) ready for prediction."
            )

        except Exception as e:

            st.error(
                "Could not process the uploaded CSV."
            )

            st.exception(e)


# ============================================================
# OPTION B: MANUAL INPUT
# ============================================================

else:

    st.subheader("Enter Feature Values")

    st.info(
        "The feature names and their order are loaded directly "
        "from feature_names.pkl. No feature names are hard-coded "
        "in this application."
    )

    user_values = {}

    # Divide features into columns for a cleaner UI
    columns = st.columns(3)

    for i, feature in enumerate(feature_names):

        column = columns[i % 3]

        with column:

            user_values[feature] = st.number_input(
                label=feature,
                value=0.0,
                step=1.0,
                key=f"feature_{i}"
            )

    input_data = pd.DataFrame(
        [user_values],
        columns=feature_names
    )


# ============================================================
# 9. PREDICTION BUTTON
# ============================================================

if input_data is not None:

    st.divider()

    predict_button = st.button(
        "🌲 Predict Forest Cover Type",
        type="primary",
        width="stretch"
    )


    # ========================================================
    # 10. PREDICTION
    # ========================================================

    if predict_button:

        try:

            # ------------------------------------------------
            # Ensure exact feature order
            # ------------------------------------------------

            input_data = input_data[
                feature_names
            ]


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = model.predict(
                input_data
            )

            predicted_class = prediction[0]


            # ------------------------------------------------
            # Probabilities
            # ------------------------------------------------

            probabilities = model.predict_proba(
                input_data
            )[0]


            # =================================================
            # 11. DISPLAY PREDICTION
            # =================================================

            st.success(
                f"Predicted Forest Cover Type: "
                f"**{predicted_class}**"
            )


            # =================================================
            # 12. PROBABILITY TABLE
            # =================================================

            st.subheader(
                "Prediction Probabilities"
            )

            probability_df = pd.DataFrame({

                "Class": class_labels,

                "Probability": probabilities

            })

            probability_df["Probability (%)"] = (
                probability_df["Probability"] * 100
            )

            probability_df = probability_df.sort_values(
                "Probability",
                ascending=False
            )

            probability_df = probability_df.reset_index(
                drop=True
            )

            display_probability_df = (
                probability_df[
                    [
                        "Class",
                        "Probability (%)"
                    ]
                ].copy()
            )

            display_probability_df[
                "Probability (%)"
            ] = display_probability_df[
                "Probability (%)"
            ].round(2)

            st.dataframe(
                display_probability_df,
                width="stretch",
                hide_index=True
            )


            # =================================================
            # 13. SHAP EXPLANATION
            # =================================================

            st.subheader(
                "Why did the model make this prediction?"
            )

            st.write(
                f"SHAP explanation for predicted "
                f"Class **{predicted_class}**"
            )


            # -------------------------------------------------
            # Calculate SHAP values
            # -------------------------------------------------

            shap_values = explainer.shap_values(
                input_data
            )


            # -------------------------------------------------
            # Determine class index
            # -------------------------------------------------

            class_index = list(
                class_labels
            ).index(predicted_class)


            # -------------------------------------------------
            # Handle different SHAP output formats
            # -------------------------------------------------

            if isinstance(shap_values, list):

                shap_class_values = (
                    shap_values[class_index][0]
                )

            else:

                shap_array = np.asarray(
                    shap_values
                )

                if shap_array.ndim == 3:

                    shap_class_values = (
                        shap_array[
                            0,
                            :,
                            class_index
                        ]
                    )

                elif shap_array.ndim == 2:

                    shap_class_values = (
                        shap_array[0]
                    )

                else:

                    shap_class_values = (
                        shap_array
                    )


            # =================================================
            # 14. LOCAL SHAP TABLE
            # =================================================

            shap_df = pd.DataFrame({

                "Feature": feature_names,

                "SHAP Value": shap_class_values

            })

            shap_df["Impact"] = np.abs(
                shap_df["SHAP Value"]
            )

            shap_df = shap_df.sort_values(
                "Impact",
                ascending=False
            )

            top_shap_df = shap_df.head(15)


            # =================================================
            # 15. SHAP BAR CHART
            # =================================================

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            plot_df = (
                top_shap_df
                .sort_values("SHAP Value")
            )

            ax.barh(
                plot_df["Feature"],
                plot_df["SHAP Value"]
            )

            ax.set_xlabel(
                "SHAP Value"
            )

            ax.set_ylabel(
                "Feature"
            )

            ax.set_title(
                f"Top Features Influencing "
                f"Prediction: Class {predicted_class}"
            )

            plt.tight_layout()

            st.pyplot(
                fig
            )

            plt.close(fig)


            # =================================================
            # 16. SHAP VALUES TABLE
            # =================================================

            st.write(
                "Top 15 features influencing this prediction:"
            )

            shap_display = top_shap_df[
                [
                    "Feature",
                    "SHAP Value",
                    "Impact"
                ]
            ].copy()

            shap_display[
                "SHAP Value"
            ] = shap_display[
                "SHAP Value"
            ].round(4)

            shap_display[
                "Impact"
            ] = shap_display[
                "Impact"
            ].round(4)

            st.dataframe(
                shap_display,
                width="stretch",
                hide_index=True
            )


        except Exception as e:

            st.error(
                "An error occurred while making the prediction."
            )

            st.exception(e)
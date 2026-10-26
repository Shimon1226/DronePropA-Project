from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# DRONEPROPA | AI DRONE HEALTH MONITORING
# ============================================================

st.set_page_config(
    page_title="DronePropA | Flight Intelligence",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path("drone_features.csv")
MODEL_FILE = Path("best_drone_fault_model.joblib")
COMPARISON_FILE = Path("model_comparison.csv")

# -------------------- THEME --------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #13283c 0%, transparent 36%),
            linear-gradient(145deg, #07111d 0%, #0a1421 55%, #0d1927 100%);
        color: #e8f0f8;
    }

    [data-testid="stSidebar"] {
        background: #091421;
        border-right: 1px solid #203347;
    }

    [data-testid="stSidebar"] * {
        color: #dce8f5;
    }

    .brand {
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #f2f8ff;
    }

    .brand span {
        color: #39d6ed;
    }

    .muted {
        color: #91a8bd;
        font-size: 13px;
    }

    .eyebrow {
        color: #39d6ed;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero {
        padding: 25px 28px;
        border: 1px solid #24435a;
        border-radius: 18px;
        background: linear-gradient(120deg, #10253a, #10202f);
        margin-bottom: 22px;
    }

    .hero h1 {
        font-size: 32px;
        margin: 8px 0;
        color: #f2f8ff;
    }

    .hero p {
        color: #9fb5c9;
        margin-bottom: 0;
    }

    .metric-card {
        padding: 19px;
        border: 1px solid #20394e;
        border-radius: 15px;
        background: linear-gradient(145deg, #112338, #0d1c2b);
        min-height: 125px;
    }

    .metric-label {
        font-size: 12px;
        color: #9ab0c5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 29px;
        font-weight: 800;
        color: #f1f7fd;
        margin-top: 8px;
    }

    .metric-note {
        font-size: 12px;
        color: #39d6ed;
        margin-top: 4px;
    }

    .panel {
        border: 1px solid #20394e;
        border-radius: 15px;
        background: #0d1b2a;
        padding: 20px;
    }

    .status-good {
        padding: 15px;
        border: 1px solid #1f785e;
        border-radius: 12px;
        background: #0c302b;
        color: #6df0b5;
        font-size: 18px;
        font-weight: 700;
    }

    .status-bad {
        padding: 15px;
        border: 1px solid #8c3c4c;
        border-radius: 12px;
        background: #381d2a;
        color: #ff8795;
        font-size: 18px;
        font-weight: 700;
    }

    div[data-testid="stButton"] > button {
        border-radius: 10px;
        border: 1px solid #2b8297;
        background: #123b50;
        color: white;
        font-weight: 700;
        min-height: 42px;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: #39d6ed;
        background: #18536b;
        color: white;
    }

    h1, h2, h3 {
        color: #eff7ff;
    }

    hr {
        border-color: #203347;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- DATA LOADING --------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def make_plot_dark(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=15, r=15, t=45, b=15),
        font=dict(color="#c8d7e6"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    return fig


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_feature_columns(df):
    excluded = ["filename", "fault_group", "label"]
    features = df.drop(columns=excluded, errors="ignore")
    features = features.select_dtypes(include=[np.number])
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")
    features = features.loc[:, features.nunique(dropna=True) > 1]
    return features.columns.tolist()


# -------------------- LOAD PROJECT FILES --------------------

if not DATA_FILE.exists():
    st.error("drone_features.csv was not found. Run main.py first.")
    st.stop()

if not MODEL_FILE.exists():
    st.error(
        "The trained model was not found. Run train_model.py first."
    )
    st.stop()

df = load_dataset()
model = load_model()

required_columns = {"filename", "label"}
if not required_columns.issubset(df.columns):
    st.error("The CSV does not contain the required filename and label columns.")
    st.stop()

feature_columns = get_feature_columns(df)

if not feature_columns:
    st.error("No usable numerical features were found.")
    st.stop()


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">DRONE<span>PROPA</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("FLIGHT INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "3D Drone Viewer",
            "AI Fault Detection",
            "Flight Analytics",
            "Model Performance",
            "Reports",
        ],
    )

    st.divider()

    st.markdown("**SYSTEM STATUS**")
    st.success("Dataset loaded")
    st.success("ML model loaded")

    st.caption("DronePropA Research Prototype · v1.0")


# -------------------- PAGE HEADER --------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AUTONOMOUS SYSTEMS / HEALTH INTELLIGENCE</div>
        <h1>Drone Health Command Center</h1>
        <p>Machine-learning-assisted propeller fault analysis using recorded flight telemetry.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

total_flights = len(df)
healthy_count = int((df["label"] == 0).sum())
faulty_count = int((df["label"] == 1).sum())

fault_rate = 100 * faulty_count / total_flights if total_flights else 0


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown("### Flight Operations Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Recorded flights", total_flights, "Dataset sequences")

    with c2:
        metric_card("Healthy flights", healthy_count, "Label: Healthy")

    with c3:
        metric_card("Faulty flights", faulty_count, "Label: Faulty")

    with c4:
        metric_card("Faulty share", f"{fault_rate:.1f}%", "Dataset composition")

    st.write("")

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### Dataset Health Distribution")

        distribution = pd.DataFrame(
            {
                "Condition": ["Healthy", "Faulty"],
                "Flights": [healthy_count, faulty_count],
            }
        )

        fig = px.pie(
            distribution,
            names="Condition",
            values="Flights",
            hole=0.72,
            color="Condition",
            color_discrete_map={
                "Healthy": "#36d6a0",
                "Faulty": "#ff6b7d",
            },
        )

        fig.update_traces(
            textposition="outside",
            textinfo="label+percent",
        )

        st.plotly_chart(
            make_plot_dark(fig, 350),
            use_container_width=True,
        )

    with right:
        st.markdown("### Flight Group Distribution")

        if "fault_group" in df.columns:
            group_counts = (
                df["fault_group"]
                .value_counts()
                .sort_index()
                .rename_axis("Fault group")
                .reset_index(name="Flights")
            )

            fig = px.bar(
                group_counts,
                x="Fault group",
                y="Flights",
                text="Flights",
                color="Flights",
                color_continuous_scale=["#164e63", "#39d6ed"],
            )

            st.plotly_chart(
                make_plot_dark(fig, 350),
                use_container_width=True,
            )
        else:
            st.info("Fault-group metadata is not available.")

    st.markdown("### Recent Dataset Records")

    st.dataframe(
        df[["filename", "fault_group", "label"]].head(10)
        if "fault_group" in df.columns
        else df[["filename", "label"]].head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Health counts represent the dataset labels, not live drone status."
    )


# ============================================================
# PAGE 2: 3D DRONE VIEWER
# ============================================================

# ============================================================
# PAGE 2: ADVANCED 3D DRONE VIEWER
# ============================================================

elif page == "3D Drone Viewer":

    st.markdown(
        """
        <div class="eyebrow">3D AIRFRAME INSPECTION</div>
        <h2>Quadcopter Digital Model</h2>
        <p style="color:#9ab0c5;">
            Explore the drone frame, propulsion layout, and motor positions.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Interactive conceptual model. The displayed geometry is illustrative "
        "and is not a live physical drone or a measured representation of "
        "the dataset's aircraft."
    )

    # Viewer controls
    control1, control2, control3 = st.columns(3)

    with control1:
        view_mode = st.selectbox(
            "Camera view",
            ["Perspective", "Top", "Front", "Side"],
        )

    with control2:
        frame_style = st.selectbox(
            "Frame style",
            ["Cyan carbon", "Silver", "Safety orange"],
        )

    with control3:
        show_rotors = st.toggle("Show propellers", value=True)

    frame_colors = {
        "Cyan carbon": "#35D6ED",
        "Silver": "#C8D3E0",
        "Safety orange": "#FF9B54",
    }

    frame_color = frame_colors[frame_style]

    fig = go.Figure()

    # --------------------------------------------------------
    # 1. Four diagonal arms
    # --------------------------------------------------------

    motor_positions = [
        (-2.0, -2.0, 0.0),
        (-2.0, 2.0, 0.0),
        (2.0, -2.0, 0.0),
        (2.0, 2.0, 0.0),
    ]

    for x, y, z in motor_positions:
        fig.add_trace(
            go.Scatter3d(
                x=[0, x],
                y=[0, y],
                z=[0.08, z],
                mode="lines",
                line=dict(
                    color=frame_color,
                    width=12,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # --------------------------------------------------------
    # 2. Central body: layered 3D markers
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0.12],
            mode="markers",
            marker=dict(
                size=29,
                color="#172D43",
                symbol="diamond",
                line=dict(color="#EAF7FF", width=3),
            ),
            name="Central airframe",
            hovertemplate="Central airframe<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0.18],
            mode="markers",
            marker=dict(
                size=11,
                color="#39D6ED",
                symbol="circle",
            ),
            name="Flight controller",
            hovertemplate="Illustrative flight controller<extra></extra>",
        )
    )

    # --------------------------------------------------------
    # 3. Motor housings and propellers
    # --------------------------------------------------------

    for i, (x, y, z) in enumerate(motor_positions, start=1):

        # Motor housing
        fig.add_trace(
            go.Scatter3d(
                x=[x],
                y=[y],
                z=[z + 0.08],
                mode="markers",
                marker=dict(
                    size=15,
                    color="#FFB454",
                    line=dict(color="#FFF0D7", width=2),
                ),
                name=f"Motor {i}",
                legendgroup="motors",
                hovertemplate=f"Motor {i}<extra></extra>",
            )
        )

        # Propeller blades
        if show_rotors:

            blade_length = 0.85

            # Two perpendicular blades
            fig.add_trace(
                go.Scatter3d(
                    x=[
                        x - blade_length,
                        x + blade_length,
                    ],
                    y=[y, y],
                    z=[z + 0.12, z + 0.12],
                    mode="lines",
                    line=dict(
                        color="#A8F3FF",
                        width=7,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[x, x],
                    y=[
                        y - blade_length,
                        y + blade_length,
                    ],
                    z=[z + 0.12, z + 0.12],
                    mode="lines",
                    line=dict(
                        color="#A8F3FF",
                        width=7,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # --------------------------------------------------------
    # 4. Camera presets
    # --------------------------------------------------------

    camera_settings = {
        "Perspective": dict(
            eye=dict(x=1.65, y=1.65, z=1.35)
        ),
        "Top": dict(
            eye=dict(x=0.0, y=0.0, z=2.8)
        ),
        "Front": dict(
            eye=dict(x=0.0, y=-3.0, z=0.8)
        ),
        "Side": dict(
            eye=dict(x=3.0, y=0.0, z=0.8)
        ),
    }

    fig.update_layout(
        scene=dict(
            bgcolor="#081522",
            xaxis=dict(
                title="X",
                range=[-3.3, 3.3],
                backgroundcolor="#081522",
                gridcolor="#1C3448",
                zerolinecolor="#244258",
            ),
            yaxis=dict(
                title="Y",
                range=[-3.3, 3.3],
                backgroundcolor="#081522",
                gridcolor="#1C3448",
                zerolinecolor="#244258",
            ),
            zaxis=dict(
                title="Z",
                range=[-1.5, 2.0],
                backgroundcolor="#081522",
                gridcolor="#1C3448",
                zerolinecolor="#244258",
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.65),
            camera=camera_settings[view_mode],
        ),
        height=680,
        margin=dict(l=0, r=0, t=35, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCEAF7"),
        legend=dict(
            bgcolor="rgba(10,25,40,0.8)",
            bordercolor="#244258",
            borderwidth=1,
        ),
        title=dict(
            text="DRONE AIRFRAME / 3D INSPECTION VIEW",
            x=0.03,
            font=dict(size=15, color="#39D6ED"),
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "responsive": True,
        },
    )

    # --------------------------------------------------------
    # 5. Component cards
    # --------------------------------------------------------

    st.markdown("### Airframe Components")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Airframe",
            "1",
            "Central body and arms",
        )

    with c2:
        metric_card(
            "Motors",
            "4",
            "Illustrative motor positions",
        )

    with c3:
        metric_card(
            "Propellers",
            "4" if show_rotors else "Hidden",
            "Visual representation",
        )

    with c4:
        metric_card(
            "Camera",
            view_mode,
            "Interactive viewing mode",
        )

    st.markdown("### Viewer Controls")

    st.markdown(
        """
        - **Rotate:** Click and drag the model.
        - **Zoom:** Use the mouse wheel or trackpad.
        - **Pan:** Drag using the appropriate mouse control.
        - **Reset:** Use the camera-reset icon in the chart toolbar.
        - **Change viewpoint:** Select Top, Front, Side, or Perspective above.
        """
    )
# ============================================================
# PAGE 3: AI FAULT DETECTION
# ============================================================

# ============================================================
# PAGE 3: AI FAULT DETECTION - ENHANCED
# ============================================================

elif page == "AI Fault Detection":

    st.markdown(
        """
        <div class="eyebrow">MACHINE LEARNING / FLIGHT DIAGNOSTICS</div>
        <h2>AI Propeller Fault Assessment</h2>
        <p style="color:#9ab0c5;">
            Analyse a recorded flight using the trained classification model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "This assessment uses previously extracted flight features. "
        "It is an offline research prototype, not a live drone diagnostic system."
    )

    # --------------------------------------------------------
    # Select flight
    # --------------------------------------------------------

    selected_file = st.selectbox(
        "Select a recorded flight",
        df["filename"].astype(str).tolist(),
        key="assessment_flight",
    )

    selected_row = df.loc[
        df["filename"].astype(str) == selected_file
    ].iloc[0]

    actual_label = int(selected_row["label"])

    st.markdown("### Flight Information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.markdown("**Flight file**")
        st.code(selected_file)

    with info2:
        st.markdown("**Dataset label**")
        if actual_label == 0:
            st.success("HEALTHY")
        else:
            st.error("FAULTY")

    with info3:
        if "fault_group" in df.columns:
            st.markdown("**Dataset fault group**")
            st.write(str(selected_row["fault_group"]))
        else:
            st.markdown("**Dataset fault group**")
            st.write("Not available")

    st.divider()

    # --------------------------------------------------------
    # Run assessment
    # --------------------------------------------------------

    if st.button(
        "RUN AI FAULT ASSESSMENT",
        type="primary",
        width="stretch",
    ):

        # Preserve the model's expected feature names and order
        input_row = pd.DataFrame(
            [
                {
                    feature: selected_row[feature]
                    for feature in feature_columns
                }
            ],
            columns=feature_columns,
        )

        # Use the same saved estimator already loaded by the app
        try:
            prediction = int(model.predict(input_row)[0])

            probabilities = None

            if hasattr(model, "predict_proba"):
                raw_probabilities = model.predict_proba(input_row)[0]

                probabilities = {
                    int(class_value): float(probability)
                    for class_value, probability
                    in zip(model.classes_, raw_probabilities)
                }

        except Exception as exc:
            st.error(
                "The model could not process this flight's features. "
                "Check the model's expected feature columns and preprocessing."
            )
            st.exception(exc)
            st.stop()

        # ----------------------------------------------------
        # Prediction result
        # ----------------------------------------------------

        st.markdown("### Assessment Result")

        result_col, detail_col = st.columns([1.2, 1])

        with result_col:

            if prediction == 0:
                st.success("PREDICTED CONDITION: HEALTHY")
                result_name = "Healthy"
            else:
                st.error("PREDICTED CONDITION: FAULTY")
                result_name = "Faulty"

            if probabilities is not None:
                predicted_probability = probabilities.get(prediction)

                if predicted_probability is not None:
                    st.metric(
                        "Model probability for predicted class",
                        f"{predicted_probability * 100:.2f}%",
                    )
            else:
                st.warning(
                    "This model does not provide probability estimates."
                )

        with detail_col:

            st.markdown("#### Dataset comparison")

            if prediction == actual_label:
                st.success(
                    "Prediction matches the dataset label."
                )
                comparison_result = "Match"
            else:
                st.warning(
                    "Prediction differs from the dataset label."
                )
                comparison_result = "Mismatch"

            st.caption(
                "A match on one selected record does not establish "
                "overall model accuracy or real-world reliability."
            )

        # ----------------------------------------------------
        # Class probabilities
        # ----------------------------------------------------

        if probabilities is not None:

            st.markdown("### Class Probability Estimates")

            probability_cols = st.columns(2)

            for column, class_value, class_name in [
                (probability_cols[0], 0, "Healthy"),
                (probability_cols[1], 1, "Faulty"),
            ]:

                probability = probabilities.get(class_value)

                with column:

                    if probability is None:
                        st.metric(class_name, "Unavailable")
                    else:
                        st.metric(
                            f"{class_name} probability",
                            f"{probability * 100:.2f}%",
                        )

                        st.progress(probability)

        # ----------------------------------------------------
        # Selected flight's feature summary
        # ----------------------------------------------------

        st.markdown("### Flight Feature Summary")

        available_display_features = [
            feature
            for feature in feature_columns
            if feature in selected_row.index
            and pd.notna(selected_row[feature])
        ]

        if available_display_features:

            feature_summary = pd.DataFrame(
                {
                    "Feature": available_display_features,
                    "Value": [
                        float(selected_row[feature])
                        for feature in available_display_features
                    ],
                }
            )

            feature_summary["Value"] = feature_summary["Value"].round(5)

            st.dataframe(
                feature_summary,
                width="stretch",
                hide_index=True,
                height=350,
            )

        else:
            st.warning("No numerical feature values are available.")

        # ----------------------------------------------------
        # Downloadable report
        # ----------------------------------------------------

        report = {
            "filename": selected_file,
            "predicted_label": prediction,
            "predicted_condition": result_name,
            "dataset_label": actual_label,
            "dataset_condition": (
                "Healthy" if actual_label == 0 else "Faulty"
            ),
            "dataset_comparison": comparison_result,
            "model_file": str(MODEL_FILE),
        }

        if probabilities is not None:
            report["healthy_probability"] = probabilities.get(0)
            report["faulty_probability"] = probabilities.get(1)

        report_df = pd.DataFrame([report])

        st.markdown("### Export Assessment")

        st.download_button(
            "Download Flight Assessment CSV",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="drone_flight_assessment.csv",
            mime="text/csv",
            width="stretch",
        )

        st.caption(
            "Probability values are model estimates, not necessarily "
            "calibrated probabilities of a real physical fault."
        )

# ============================================================
# PAGE 4: FLIGHT ANALYTICS
# ============================================================

elif page == "Flight Analytics":
    st.markdown("## Flight Analytics")
    st.write("Explore statistical features extracted from recorded flights.")

    selected_feature = st.selectbox(
        "Select a numerical feature",
        feature_columns
    )

    st.markdown("### Feature Statistics")

    st.dataframe(
        df.groupby("label")[selected_feature]
        .agg(["mean", "std", "min", "max"])
        .rename(index={0: "Healthy", 1: "Faulty"}),
        use_container_width=True
    )

# ============================================================
# PAGE 5: MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":

    st.markdown("### Machine Learning Model Comparison")

    if COMPARISON_FILE.exists():

        comparison = pd.read_csv(COMPARISON_FILE)

        required_metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
        ]

        available_metrics = [
            metric
            for metric in required_metrics
            if metric in comparison.columns
        ]

        if available_metrics and "Model" in comparison.columns:

            fig = px.bar(
                comparison,
                x="Model",
                y=available_metrics,
                barmode="group",
                title="Initial random-split evaluation",
            )

            fig.update_yaxes(range=[0, 1.05])

            st.plotly_chart(
                make_plot_dark(fig, 450),
                use_container_width=True,
            )

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.warning(
                "The comparison file has an unexpected column format."
            )

    else:
        st.warning(
            "model_comparison.csv not found. Run train_model.py first."
        )

    st.markdown("### Validation Summary")

    st.info(
        "Your separate validation experiment reported preliminary "
        "held-out-trajectory results of 80.8% accuracy for Logistic "
        "Regression and 84.6% for Random Forest. These values come from "
        "a different split and should not be mixed with the random-split "
        "chart above."
    )


# ============================================================
# PAGE 6: REPORTS
# ============================================================

elif page == "Reports":

    st.markdown("### Flight Data & Report Export")

    st.write(
        "Download the processed feature dataset or the model comparison table."
    )

    st.download_button(
        "Download Flight Feature Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="drone_features_export.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if COMPARISON_FILE.exists():

        comparison = pd.read_csv(COMPARISON_FILE)

        st.download_button(
            "Download Model Comparison",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("### Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Dataset Summary")

    st.json(
        {
            "total_flights": int(total_flights),
            "healthy_flights": int(healthy_count),
            "faulty_flights": int(faulty_count),
            "numerical_features": int(len(feature_columns)),
            "faulty_percentage": round(fault_rate, 2),
        }
    )


# -------------------- FOOTER --------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#7f96aa; padding:10px;">
        DRONEPROPA · FLIGHT INTELLIGENCE PLATFORM<br>
        <span style="font-size:11px;">
            Machine-learning research prototype · Recorded dataset analysis
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
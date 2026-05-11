import streamlit as st
import plotly.graph_objects as go
import numpy as np

IPL_TEAMS = {
    "Chennai Super Kings": "#FFFF00",
    "Mumbai Indians": "#004BA0",
    "Royal Challengers Bengaluru": "#EC1C24",
    "Kolkata Knight Riders": "#3A225D",
    "Rajasthan Royals": "#FF69B4",
    "Gujarat Titans": "#1C1C1C",
    "Delhi Capitals": "#0078BC",
    "Punjab Kings": "#D71920",
    "Sunrisers Hyderabad": "#FF822A",
    "Lucknow Super Giants": "#00AEEF"
}

INTERNATIONAL_TEAMS = {
    "India": "#FF9933",
    "Australia": "#FFD700",
    "England": "#012169",
    "Pakistan": "#006B3F",
    "South Africa": "#007A5E",
    "New Zealand": "#000000",
    "West Indies": "#CE1126",
    "Sri Lanka": "#0052CC",
    "Bangladesh": "#006C4C",
    "Afghanistan": "#CC0000"
}

st.title("CricVision AI")
st.subheader("AI-Powered Cricket Match Analyst")


match_type = st.sidebar.radio(
    "Match Type",
    ["IPL", "International"]
)


if match_type == "IPL":
    match_format = "T20"
    max_overs = 20
    default_overs = 12.0
    teams = IPL_TEAMS

else:
    match_format = st.sidebar.radio(
        "Match Format",
        ["T20", "ODI", "Test"]
    )

    if match_format == "T20":
        max_overs = 20
        default_overs = 12.0

    elif match_format == "ODI":
        max_overs = 50
        default_overs = 25.0

    else:
        max_overs = 90
        default_overs = 45.0

    teams = INTERNATIONAL_TEAMS


batting_team = st.sidebar.selectbox(
    "Batting Team",
    list(teams.keys())
)

bowling_team = st.sidebar.selectbox(
    "Bowling Team",
    list(teams.keys())
)


if match_format == "Test":
    target = st.sidebar.number_input(
        "Target Score",
        min_value=0,
        value=300
    )

    current_score = st.sidebar.number_input(
        "Current Score",
        min_value=0,
        value=200
    )

elif match_format == "ODI":
    target = st.sidebar.number_input(
        "Target Score",
        min_value=0,
        value=280
    )

    current_score = st.sidebar.number_input(
        "Current Score",
        min_value=0,
        value=200
    )

else:
    target = st.sidebar.number_input(
        "Target Score",
        min_value=0,
        value=180
    )

    current_score = st.sidebar.number_input(
        "Current Score",
        min_value=0,
        value=120
    )


overs_options = []

for over in range(max_overs + 1):
    for ball in range(6):

        if over == max_overs and ball > 0:
            break

        overs_options.append(
            float(f"{over}.{ball}")
        )


overs_input = st.sidebar.select_slider(
    "Overs Completed",
    options=overs_options,
    value=default_overs
)


wickets = st.sidebar.slider(
    "Wickets Lost",
    0,
    10,
    3
)


completed_overs = int(overs_input)

completed_balls = int(
    round(
        (overs_input - completed_overs) * 10
    )
)

balls_completed = (
    completed_overs * 6
) + completed_balls


overs = balls_completed / 6

total_balls = max_overs * 6


bat_color = teams[batting_team]
bowl_color = teams[bowling_team]


runs_left = target - current_score
balls_left = total_balls - balls_completed
wickets_remaining = 10 - wickets


current_rr = (
    current_score / overs
    if overs > 0 else 0
)

required_rr = (
    (runs_left * 6) / balls_left
    if balls_left > 0 else 0
)


if balls_left == 0:

    if runs_left <= 0:
        win_probability = 100
    else:
        win_probability = 0

else:

    rr_diff = current_rr - required_rr

    if rr_diff >= 0 and wickets_remaining >= 3:

        win_probability = min(
            99,
            70 + (rr_diff * 8) +
            (wickets_remaining * 2.5)
        )

    elif rr_diff >= 0:

        win_probability = min(
            95,
            60 + (rr_diff * 12)
        )

    else:

        win_probability = max(
            1,
            35 -
            (abs(rr_diff) * 8) -
            (wickets * 6)
        )


win_probability = round(
    win_probability,
    2
)

lose_probability = round(
    100 - win_probability,
    2
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Runs Left",
    runs_left
)

col2.metric(
    "Balls Left",
    balls_left
)

col3.metric(
    "Current RR",
    round(current_rr, 2)
)

col4.metric(
    "Required RR",
    round(required_rr, 2)
)


st.subheader("Win Prediction")


col5, col6 = st.columns(2)

col5.metric(
    batting_team,
    f"{win_probability}%"
)

col6.metric(
    bowling_team,
    f"{lose_probability}%"
)


fig = go.Figure(
    data=[
        go.Pie(
            labels=[
                batting_team,
                bowling_team
            ],

            values=[
                win_probability,
                lose_probability
            ],

            hole=0.5,

            marker=dict(
                colors=[
                    bat_color,
                    bowl_color
                ]
            )
        )
    ]
)

fig.update_layout(
    title=f"{match_format} - Winning Probability",
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)


if match_format == "T20":
    overs_list = [2,4,6,8,10,12,14,16,18,20]

elif match_format == "ODI":
    overs_list = [10,20,30,40,50]

else:
    overs_list = [20,40,60,80]


rr_data = [
    6,7,8,7.5,9,
    8.2,10,11,12,13
][:len(overs_list)]


graph = go.Figure()

graph.add_trace(
    go.Scatter(
        x=overs_list,
        y=rr_data,
        mode="lines+markers",
        name=batting_team,

        line=dict(
            color=bat_color,
            width=4
        )
    )
)

graph.update_layout(
    title="Run Rate Over Time",
    xaxis_title="Overs",
    yaxis_title="Run Rate"
)

st.plotly_chart(
    graph,
    use_container_width=True
)


st.success(
    f"{batting_team} need "
    f"{runs_left} runs from "
    f"{balls_left} balls. "
    f"AI gives them "
    f"{win_probability}% chance to win."
)


if win_probability > 50:

    st.info(
        f"{batting_team} are likely to win!"
    )

else:

    st.info(
        f"{bowling_team} are likely to win!"
    )



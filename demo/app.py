import streamlit as st

st.set_page_config(page_title="Smarter Shelves", page_icon="☕", layout="wide")

pages = [
    st.Page("home.py", title="Home", icon="☕", default=True),
    st.Page("pages/1_Out_of_stock.py", title="Out of stock? Ask the AI", icon="🤖"),
    st.Page("pages/2_Plan_the_shelf.py", title="Plan the shelf", icon="🛒"),
    st.Page("pages/3_Beat_the_optimiser.py", title="Beat the optimiser", icon="🏆"),
    st.Page("pages/4_Which_AI.py", title="Does the AI matter?", icon="🧪"),
    st.Page("pages/5_Check_by_hand.py", title="Check it by hand", icon="🧮"),
]
st.navigation(pages).run()

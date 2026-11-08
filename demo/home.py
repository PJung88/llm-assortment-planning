import streamlit as st

st.title("☕ Smarter Shelves")
st.subheader("Which coffees should a shop stock when it can't stock them all?")

st.markdown(
    """
**The problem.** A shop could sell 60 coffees but has room for only 30. When a customer's usual
coffee isn't on the shelf, some buy a similar one and the rest leave without buying. So the best
shelf is not simply the 30 bestsellers. It is the 30 coffees that keep the most customers,
**counting the ones who switch**.

**The catch.** To plan like that, you need to know which coffee people switch to. Shops normally
learn this from their sales records, but new products have no sales records yet.

**The idea tested here.** Ask AI models to judge, from the product descriptions alone, which coffees
are the closest substitutes. Then let an **optimiser** (a program that checks the possible
combinations and finds the best one) choose the shelf.
"""
)

st.markdown("#### What we found")
a, b, c = st.columns(3)
a.metric("More customers kept than with the bestsellers", "+6%")
a.caption("Typical case: room for 30 coffees, 6 in 10 customers willing to switch.")
b.metric("When shelf space is tight", "about +18%")
b.caption("Room for only 15 coffees, 7.5 in 10 customers willing to switch.")
c.metric("AI models compared", "4")
c.caption("Gemini, Claude, Qwen and DeepSeek. Any one of them gave almost the same shelf as all four together.")

st.markdown("#### How it works")
s1, s2, s3 = st.columns(3)
s1.markdown("**1. Ask the AI**  \nFor each coffee, four AI models see the 7 coffees with the most similar "
            "descriptions and rank which one a customer would most likely buy instead. They see only the "
            "descriptions, never sales figures.")
s2.markdown("**2. Apply common sense**  \nSubstitutes a customer can't actually use are ruled out: "
            "capsules for a different machine, ground coffee for someone who buys whole beans, "
            "regular coffee for someone who wants decaf.")
s3.markdown("**3. Plan the shelf**  \nThe optimiser picks the 30 coffees that keep the most customers, "
            "counting both those who find their usual coffee and those who switch.")

st.markdown("#### Try it (each page takes about a minute)")
st.page_link("pages/1_Out_of_stock.py", label="Out of stock? Ask the AI: see what the AIs suggest "
             "when a coffee runs out", icon="🤖")
st.page_link("pages/2_Plan_the_shelf.py", label="Plan the shelf: compare the bestseller shelf with "
             "the smart shelf", icon="🛒")
st.page_link("pages/3_Beat_the_optimiser.py", label="Beat the optimiser: pick 30 coffees yourself "
             "and see how you score", icon="🏆")
st.page_link("pages/4_Which_AI.py", label="Does the AI matter? See what changes if you use one "
             "AI, a simpler method, or no AI", icon="🧪")
st.page_link("pages/5_Check_by_hand.py", label="Check it by hand: work through the optimiser's "
             "maths on 3-4 coffees with a calculator", icon="🧮")

with st.expander("Words used in this demo"):
    st.markdown(
        """
- **Substitute**: a coffee a customer buys when their usual one is missing.
- **Customers willing to switch** ("6 in 10"): of the customers whose coffee is missing, how many buy
  another coffee instead of leaving. The study tests 4.5, 6 and 7.5 in 10.
- **Customers kept**: the share of all customers who leave with a coffee, either their usual one or a
  substitute.
- **Popularity**: how well each coffee sells. The 60 coffees are real Amazon listings, but their sales
  are **simulated**. The study deliberately sets real sales data aside to imitate new products that
  have no sales history.
- **Optimiser**: a program that searches the possible shelves and finds the one that keeps the most
  customers under the stated assumptions. The demo uses the same model as the thesis.
- **Common-sense rules** (compatibility rules): rule out substitutes a customer physically can't
  use, even if an AI ranked them highly.
"""
    )

st.caption("Built on 60 real Amazon coffee listings. Sales are simulated, and the substitutes are AI "
           "estimates, not observed purchases. The demo shows how the AI estimates change the shelf "
           "decision; it does not prove that real customers would behave this way.")

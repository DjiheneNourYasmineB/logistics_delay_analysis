import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import folium 
from streamlit_folium import st_folium
from scipy.stats import ks_2samp




st.set_page_config(page_title="Logistic Delays EDA & Classification", page_icon="📦", layout="wide")




@st.cache_data
def load_data():
    link = "logistics_delays.csv"
    df = pd.read_csv(link, sep=",", engine='python')
    df.drop_duplicates(inplace=True)
    df.drop(columns=["customer_id", "customer_zipcode", "product_card_id", "order_customer_id", "order_item_cardprod_id", "order_item_id", "product_card_id", "product_category_id"], inplace=True, errors='ignore')
    #df['delay_label'] = df['label'].map({'Yes': 1, 'No': 0, 'Early': -1})
    


    return df

df = load_data()

st.sidebar.title("Logistic Delays EDA & Classification ")
page = st.sidebar.radio("Select Page", ["🔍 Data Preprocessing", "📊 Visualizations", "⚙️ Classification"])

if page == "🔍 Data Preprocessing":
    st.title("🔍 Data Preprocessing")
    st.subheader("🚚 Introduction")
    st.write("Supply chain management is a very challenging yet a crucial element businesses nowadays. Managing delays effectively relies on good strategy and the use of technology. Here we have a dataset that groups a combination of different variables (numrerical and categorical) such as prices or location. The goal is to analyze it and implement a performant classification algorithm.")
    st.subheader("📌 Raw Data Preview")
    st.write(df.head())
    st.subheader("📊 Data Overview")
    st.write(f"Number of Rows: {df.shape[0]}")
    st.write(f"Number of Columns: {df.shape[1]}")
    st.subheader("🔢 Basic stats")
    st.write(df.describe())
    st.subheader("🗂️ About The Dataset")
    link_to_dataset = "https://www.kaggle.com/datasets/pushpitkamboj/logistics-data-containing-real-world-data"
    st.write("This dataset has can be found on Kaggle [here](%s). It has no missing values and around 41 variables intially. Some were removed promptly as they were not really relevant/used in our case."%link_to_dataset)

numerical_variables = df[['profit_per_order', 'sales_per_customer', 'category_id', 'department_id', 'order_item_discount', 
                          'order_item_discount_rate', 'order_item_product_price', 'order_item_profit_ratio', 'order_item_quantity',
                          'sales', 'order_item_total_amount', 'order_profit_per_order']]

categorical_variables = df[['payment_type', 'customer_city', 'customer_country', 'customer_segment', 'customer_state', 'market', 'order_city', 'order_country', 'order_region',
                            'order_state', 'order_status', 'shipping_mode']]


if page == "📊 Visualizations":

    st.subheader("Key Metrics:")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sales", f"${df['sales_per_customer'].sum():,.0f}")
    
    with col2:
        st.metric("Avg Profit Per Order", f"${df['profit_per_order'].mean():,.2f}")
    with col3:
        st.metric("On Time Delivery Rate", f"{(df['label'] == 0).mean()*100:,.2f}%")
    with col4:
        st.metric("Delay Rate", f"{(df['label'] == 1).mean()*100:,.2f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["Bar Plots", "Distributions", "Vizualiazations", " Key Relationships"])

    with tab1:

        categorical_variables1 = df[['label','payment_type', 'customer_city', 'customer_country', 'customer_segment', 'customer_state', 'market', 'order_city', 'order_country', 'order_region',
                            'order_state', 'order_status', 'shipping_mode']]



        for variable in categorical_variables1:
            counts = df[variable].value_counts().reset_index()
            counts.columns = [variable, 'Count']
            fig = px.bar(
                counts,
                x= variable,
                y='Count',
                title = f"Bar Plot of The Variable {variable}"

            )
            
            tab1.plotly_chart(fig)



    with tab2:

        for idx, variable in enumerate(numerical_variables):
            fig = px.histogram(
                df,
                x=variable,
                title= f"Histogram of The Variable {variable}"
            )
            
            tab2.plotly_chart(fig, key=f"{variable}_hist_{idx}")




    with tab3:
        st.subheader("Map of Shop Locations:")
        m = folium.Map(location=[20, 0], tiles="OpenStreetMap", zoom_start=2)
  
        for _, row in df.iterrows():
            folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,  
            color="pink",
            fill=True,
            fill_color="blue"
            ).add_to(m)
        
        st_map = st_folium(m)

    

    

    with tab4:

        st.subheader("Correlation Matrix:")
        fig, ax = plt.subplots(figsize=(16, 11))
        sns.heatmap(numerical_variables.corr(), cmap="coolwarm", annot=True, ax=ax)
        st.pyplot(fig)

        st.subheader("")

        """
        ks = {
    var: ks_2samp(df[var][df["flag_reachat"] == 1],
                  df[var][df["flag_reachat"] == 0]).statistic
    for var in numerical_variables}
        

    """

if page == "⚙️ Classification":

    st.subheader("💻 Using Random Forest for delay detection:")
    st.write("Now that we have a peak into our data, let's start classification. But before, let's have some notes on our dataframe. Here we are going for a very simplistic approach. Looking into our data we could take more steps to improve the quality of our variables.")
    st.write("• To start, this dataset has no missing values so we have less things to worry about. ")
    st.write("• I generally would not recommend removing variables from the get go, but we have some variables that I find to be redundant (variables that cover regions) with too many classes. I personnally try work with variables that contain a reasonable amount of classes (i.e. 5), unless they're very relevant. So, here I'm striking the variables related to customer city/state.")
    st.write("• As mentionned, we could do a lot more feature engineering. we could create new indicators. For instance, if we had a time reference we could count how much time did the order take to be delivered.")
    st.write("• Also, looking at our histograms, it might be interesting to consider discretizing some variables (i.e. item_quantity)")
    st.write("• In this case, I am going for a binary classification, but either way we won't have class imbalance which is a good thing because we won't have to worry about that (it might not always be the case for real world data). That said, in general, ensemble methods seem to handle this issue quite well without falling into overfitting.")
    
    
    st.subheader("Let's start classification!")

    
    model_button = st.selectbox("Choose Model", ["🕸️ Neural Networks", "🌲 Random Forest"])

    if model_button == ("🌲 Random Forest"):
        st.info("ℹ️ This Step Might Take some Time!")


   
        





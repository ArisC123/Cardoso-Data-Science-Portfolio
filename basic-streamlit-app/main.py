import streamlit as st
import pandas as pd

st.title("Welcome to the Penguin Disovery Dashboard!")

data = pd.read_csv("Data/penguins.csv")

#st.write("Here's a sneak peak of our dataset:")
#st.dataframe(data.head(5))

##Summary Statistics Portion
st.subheader("Penguin Trait Averages")
species = st.selectbox("Select the species", data['species'].unique())
island = st.selectbox("Select the island", data['island'].unique())
year = st.selectbox("Select the year", data['year'].unique())
sex = st.selectbox("Select the gender", data['sex'].unique())

filtered_data = data[(data['species'] == species) & (data['island'] == island) & (data['year'] == year) & 
                     (data['sex'] == sex)]

#st.dataframe(filtered_data.select_dtypes(include = ['number']).describe())  

#st.metric(label="Count", value= f'{filtered_data.shape[0]}')

#label="Count", value= f'{round(filtered_data['bill_length_mm'].mean(),2)}')

col1, col2,col3, col4, col5 = st.columns(5)
col1.metric('Count', f'{filtered_data.shape[0]}')
col2.metric('Bill Length', f'{round(filtered_data["bill_length_mm"].mean(),1)}')
col3.metric('Bill Depth', f'{round(filtered_data["bill_depth_mm"].mean(),1)}')
col4.metric('Flipper Length', f'{round(filtered_data["flipper_length_mm"].mean(),1)}')
col5.metric('Body Mass', f'{round(filtered_data["body_mass_g"].mean(),1)}')





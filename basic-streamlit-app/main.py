import streamlit as st
import pandas as pd

st.title("Welcome to the Penguin Disovery Dashboard!")
st.divider()
data = pd.read_csv("Data/penguins.csv")

st.write("Here's a sneak peak of our dataset:")
st.dataframe(data.head())

st.divider()
#Summary Statistics Portion
st.header("Penguin Trait Averages")
species = st.selectbox("Select the species", data['species'].unique())
island = st.selectbox("Select the island", data['island'].unique())
year = st.selectbox("Select the year", data['year'].unique())
sex = st.selectbox("Select the gender", data['sex'].unique())

filtered_data = data[(data['species'] == species) & (data['island'] == island) & (data['year'] == year) & 
                     (data['sex'] == sex)]

col1, col2,col3, col4, col5 = st.columns(5)
col1.metric('Count', f'{filtered_data.shape[0]}')
col2.metric('Bill Length', f'{round(filtered_data["bill_length_mm"].mean(),1)}')
col3.metric('Bill Depth', f'{round(filtered_data["bill_depth_mm"].mean(),1)}')
col4.metric('Flipper Length', f'{round(filtered_data["flipper_length_mm"].mean(),1)}')
col5.metric('Body Mass', f'{round(filtered_data["body_mass_g"].mean(),1)}')

st.divider()
#Data Visualization Portion
st.header("Data Visualizations")


tab1, tab2, tab3, tab4 = st.tabs(["Species Distribution", "Island Distribution", 'Flipper & Bodymass Scatterplot', 
                                 'Sex Makeup'])


with tab1:
    species_count = data['species'].value_counts()
    st.bar_chart(species_count, y_label='Species', x_label='Count', color = '#CFA8FF', width = 200, height = 300,
                 horizontal= True)

with tab2:
    island_porportion = data['island'].value_counts(normalize=True)
    st.bar_chart(island_porportion*100, y_label='Island', x_label='Count', color = '#FFC0CB', width = 200, height = 300,
                 stack = True) 

with  tab3:
    species_hist = st.selectbox('Select a species', data['species'].unique())
    sex_hist = st.selectbox('Select a gender', data['sex'].unique())

    hist_data = data[(data['species'] == species_hist) & (data['sex'] == sex_hist)] 
    st.scatter_chart(hist_data, x = 'flipper_length_mm', y = 'body_mass_g', x_label = 'Flipper Length',
                    y_label = 'Body Mass(g)', width = 10, height = 300)

with tab4:
    gender_make = data['sex'].value_counts()
    st.bar_chart(gender_make, x_label ='Sex', y_label = 'Count', color = '#9CAF88', width = 5, height = 300)









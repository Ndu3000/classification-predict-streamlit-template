"""

    Simple Streamlit webserver application for serving developed classification
    models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import pprint

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifer")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Information", "Exploratory Data Analysis", "Prediction", "Conclusion"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.")
        #st.subheader("Who should use this tool")
        st.subheader("Benefits of using this tool")
        st.markdown("If you are a startup or even an established business looking to launch a new product, are you aware of your potential customers sentiments regarding climate change? As a not for profit organisation looking for donors for environmental projects, do you know what your donors thoughts are regarding climate change? Knowing this information can help you better prepare to take your organisations strategy forward. Not knowing this information can make you seem irrelevant to your target market and cause you to miss out on an opporunity of a lifetime. The tweet classifier will help you be more prepared and relevant to your audience.")
        st.subheader("Instructions for using this tool")
        st.markdown("Let us help you turn insights from your potential customers to action.")
        st.markdown("Get started by:")
        st.markdown("1. Navigating to the sidebar at the top left of this page")
        st.markdown("2. Choose an option by clicking the 'Choose Option' dropdown")
        st.markdown("3. Select the option you wish to view")
        st.markdown("4. Get insights that will help you be better prepared")
        st.subheader("Example chart: What are the most frequent opinions regarding climate change?")

        import plotly.figure_factory as ff
        # Add histogram data
        x1 = np.random.randn(200) + 1
        x2 = np.random.randn(200)
        x3 = np.random.randn(200)
        x4 = np.random.randn(200) - 1


        # Group data together
        hist_data = [x1, x2, x3, x4]

        group_labels = ['Anti: -1', 'Neutral: 0', 'Pro: 1', 'News: 2']

         # Create distplot with custom bin_size
        fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5, .75])

        # Plot!
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("Your conclusion from the above example chart may be that those who are against ('Anti:-1') climate change are likely to have tweet more about  their opinions. The resulting insight from this tool may help you with your next step for formulating your brand messaging and positioning etc.")
                #chart_data1 = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
        #st.line_chart(chart_data1)


        #if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            #st.write(raw[['sentiment', 'message']]) # will write the df to the page

        st.subheader("Raw Twitter data and label")

        st.markdown("The collection of this data which we use as our data source was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were collected. Each tweet is labelled as one of the following classes in the table below:")
        chart_data2 = {'Class':['2', '1', '0', '-1'], 'Description':['News: the tweet links to factual news about climate change', 'Pro: the tweet supports the belief of man-made climate change', 'Neutral: the tweet neither supports nor refutes the belief of man-made climate change', 'Anti: the tweet does not believe in man-made climate change']}
        df = pd.DataFrame(chart_data2)
        blankIndex=[''] * len(df)
        df.index=blankIndex
        st.subheader("Table: Data dictionary")
        st.table(df)
        st.markdown("Select tickbox to view raw data where the 'sentiment'column denotes the 'class' column in the above data dictionary table which is associated to a description ranging from 'Anti' to'News' as descriptions for tweet messages that classify a range people that do not believe to those that believe in climate change respectively.")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page


    # Building out the Exploratory Data Analysis page
    #Importing Data
    my_dataset = 'train.csv'

    if selection == "Exploratory Data Analysis":
        st.info("Tweet Data Insights")
        st.markdown("describe what data insights are here")

    #Loading Dataset
    def explore_data(dataset):
        train_df = pd.read_csv(os.path.join(dataset))
        return train_df

    if st.checkbox('Preview Dataset'):
        data = explore_data(my_dataset)
        if st.button('Head'):
            st.write(data.head())

        st.checkbox('Preview DataFrame')
        st.checkbox('Show All DataFrame')
        st.checkbox('Show All Column Names')
        
    if st.subheader("EDA Visualisations"):
        st.checkbox('Show Visuals')



    # Building out the predication page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

    # Building out the Conclusion page
    if selection == "Conclusion":
        st.info("Conclusion")
        st.markdown("Write what we concluded on here")


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()

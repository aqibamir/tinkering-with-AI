import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAI
from langchain.schema import StrOutputParser
from dotenv import load_dotenv

# Load the enviroment variables
load_dotenv()

# Age Brackets for Kids
age_brackets = ["2-3 Years","3-4 Years", "4-6 Years", "6-8 Years", "8-12 Years", "12+ Years"]

# Story Genres
genres = ["Classic", "Fable", "Fairytale", "Adventure", "Educational", "Mystery", "Science Fiction"]

# Writing Styles
styles = ["Imaginative", "Funny", "Heartwarming", "sppoky"]


def get_story_title_prompt():
    
    template = """
    You are a bed time stories writing enthusiast. You write stories that are engaging and easy to read to children. 

    YOUR STORIES ARE UNIQUE AND YOU DO NOT COPY OTHER PEOPLES WORK

    YOU HAVE TO GENERATE THE TITLE OF THE STORY BASED ON THE FOLLOWING INSTRUCTIONS AND OPTIONAL CONTEXT.

    === INSTRUCTIONS ===
    
    1. Age Group for the story is : ```{age_group}```.

    2. Genre of the story is : ```{story_genre}```. 

    3. Writing style :  ```{writing_style}```. 
    
    ======================= 
    
    ==== OPTIONAL CONTEXT ====
    
    {helping_prompt}

    ==========================
    

    Title : 
    """

    prompt = PromptTemplate.from_template(template=template)

    return prompt


def get_story_prompt():

    template = """
    You are a bed time stories writing enthusiast. You write stories that are engaging and easy to read to children. 

    YOUR STORIES ARE UNIQUE AND YOU DO NOT COPY OTHER PEOPLES WORK. 

    YOU WILL NOW CREATE A STORY BASED ON THE FOLLOWING INSTRUCTIONS

    === INSTRUCTIONS ===
    1. Title of the story : {story_title}

    2. Age Group for the story is : ```{age_group}```.

    3. Genre of the story is : ```{story_genre}```. 

    4. Writing style :  ```{writing_style}```. 

    5. Length of the story is MODERATE with maximum 1000 words.
    
    6. You will only generate the story and will not add the title in the final output.
    ======================= 
    
    ==== OPTIONAL CONTEXT ====
    
    {helping_prompt}

    ==========================
    

    Story : 
    """

    prompt = PromptTemplate.from_template(template=template)
    return prompt

def main():
    st.header("Bed Time Stories")
    st.text("Tell me what kind of story you want")

    # Initialize session state
    if 'age_group' not in st.session_state:
        st.session_state.age_group = age_brackets[0]

    if 'story_genre' not in st.session_state:
        st.session_state.story_genre = genres[0]

    if 'writing_style' not in st.session_state:
        st.session_state.writing_style = styles[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.session_state.age_group = st.selectbox("Reader's Age", age_brackets, index=age_brackets.index(st.session_state.age_group))

    with col2:
        st.session_state.story_genre = st.selectbox("Story Genre", genres, index=genres.index(st.session_state.story_genre))

    with col3:
        st.session_state.writing_style = st.selectbox("Writing Style", styles, index=styles.index(st.session_state.writing_style))

    st.text_area("Story Prompt", placeholder="Give me a starter !!", height=40, key="helping_prompt")

    if st.button("Generate"):
        with st.spinner("I'm generating the story !!!"):
            title, story = generate()

        # Display the title
        st.subheader("Title")
        st.markdown(f"**{title}**")

        # Display the story
        st.write(story)


def generate():
    llm = OpenAI(max_tokens=3500)

    # Create a dictionary of all the 
    user_selections = {
        'age_group': st.session_state.age_group,
        'story_genre': st.session_state.story_genre,
        'writing_style':  st.session_state.writing_style,
        'helping_prompt': st.session_state.helping_prompt
    }

    # Get the prompts for the chains
    title_prompt = get_story_title_prompt()
    story_prompt = get_story_prompt()

    title_chain = title_prompt | llm | StrOutputParser()
    story_chain = story_prompt | llm | StrOutputParser()

    story_title = title_chain.invoke(user_selections)
    user_selections['story_title'] = story_title
    story = story_chain.invoke(user_selections)

    return [story_title,story]

main()
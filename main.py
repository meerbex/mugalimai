import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate
from openai import OpenAI
import os
from docx import Document

# Create a new Word document
# import openai

st.set_page_config(page_title="Mugalim AI")
st.title("Mugalim AI")
my_secret = os.environ['OPEN_AI']
client = OpenAI(api_key=my_secret)

# openai.api_key = my_secret

# Create lists of subjects and grades
subjects =["Математика",
          "Русский язык",
          "Литература",
          "География",
          "Физика",
          "Химия",
          "Биология",
          "История",
          "Обществознание",
          "Информатика",
          "Иностранный язык",
          "Музыка",
          "Изобразительное искусство",
          "Физическая культура"]
grades = ["1 класс",
         "2 класс",
         "3 класс",
         "4 класс",
         "5 класс",
         "6 класс",
         "7 класс",
         "8 класс",
         "9 класс",
         "10 класс",
         "11 класс"]
# Создаем списки предметов и классов


# Выпадающий список для выбора предмета
selected_subject = st.selectbox("Выберите предмет:", subjects)

# Выпадающий список для выбора класса
selected_grade = st.selectbox("Выберите класс:", grades)

def generate_response(topic, subject, grade):
  # llm = OpenAI(model_name='gpt-3.5-turbo', openai_api_key=my_secret)
  # Prompt
  template = f"""Как эксперт ты делаешь план урока на тему {topic} в предмете {subject} 
  для {grade} указывая время в течении 45 минут. План урока должен быть
  строго по стандарту BOPPPS. Сделай по этому методу: 1. Наведение мостов (5 минут):	2. 
  Цель урока (5 минут):	
  3. Оценивание знаний (10 минут):	
  4. Активное обучение:	
  5. Оценивание (5 минут):	
  6. Итоги (5 минут):
  Пример:
  1. Наведение мостов	Активизация предварительных знаний учащихся о писателе Иване Бунине.	5 минут
  2. Цель урока	Объявление цели урока: изучение рассказа Ивана Бунина "Темные аллеи".	5 минут
  3. Оценивание знаний	Метод З-Х-У (Знаю, Хочу узнать, Узнал) с использованием разделения доски на три части.	10 минут
  4. Активное обучение	Применение метода Кубика Блума для анализа и обсуждения рассказа "Темные аллеи".	15 минут
  5. Оценивание	Раздача остатка рассказа учащимся для домашнего чтения и подготовки краткого содержания.	5 минут
  6. Итоги	Обсуждение впечатлений от урока и ожиданий от дальнейшего изучения рассказа.	5 минут
  """
  completion = client.completions.create(
    model="text-davinci-003",
    prompt=template,
    max_tokens=2016,
    temperature=0
  )
  return completion.choices[0].text

def generate_response2(topic, subject, grade):
  # llm = OpenAI(model_name='gpt-3.5-turbo', openai_api_key=my_secret)
  # Prompt
  template = f"""Создай вопросы по {topic} по предмету {subject} для {grade} . Вопросы должны быть с 1. 2. 3. ... до 10 и ответы a) b) c) d) строго по стандарту BOPPPS. """
  completion = client.completions.create(
    model="text-davinci-003",
    prompt=template,
    max_tokens=3016,

    temperature=0
  )
  return completion.choices[0].text

# Check if both subject and grade are selected
if selected_subject and selected_grade:
  # Function to generate response
  

  if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-1106-preview"

  if "messages" not in st.session_state:
    st.session_state.messages = []

  # set initial message
  if "messages" not in st.session_state.keys():

    st.session_state.messages = [{
        "role":
        "assistant",
        "content":
        "Привет меня зовут Mugalim AI. Я могу помочь тебе с созданием плана"
    }]
  for message in st.session_state.messages:
    with st.chat_message(
        message["role"],
        avatar=
        "https://mugalim-edu.com/theme/image.php/moove/theme/1696251712/favicon"
    ):
      st.markdown(message["content"])

  if prompt := st.chat_input("Напишите тему на сегодня..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

  # Generate response when the user inputs a topic
  if prompt:
    response = ''
    with st.spinner('Загрузка данных...'):
      response = generate_response(prompt, selected_subject, selected_grade)
    with st.chat_message("assistant"):
      st.info(response)
    # Generate a test with 10 questions
    test_prompt = "Создать тест с 10 вопросами:"
    response2 = ''
    with st.spinner('Создаем тест на эту тему '+prompt):
      response2 = generate_response2(test_prompt, selected_subject, selected_grade)
    with st.chat_message("assistant"):
        st.info(response2)

        doc = Document()
    
        doc.add_paragraph(response2)
    
        # Save the Word document
        doc.save('Тест на тему '+prompt+'.docx')
        with open(f'Тест на тему {prompt}.docx', 'rb') as f:
           st.download_button('Скачать как Docx', f, file_name=f'Тест на тему {prompt}.docx')

else:
  st.warning("Please select a subject and a grade before entering the text.")

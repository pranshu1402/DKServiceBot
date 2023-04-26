from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo", max_tokens=num_outputs, api_key=os.environ.get("OPENAI_API_KEY")))
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk('index.json')
    return index

index = construct_index("./docs")

def chatbot(input_text):
    # index = GPTSimpleVectorIndex.load_from_disk('index.json')
    prompt = "I want you to act as a customer service agent for an organization named Deskera. " \
            "Deskera is an all-in-one cloud-based accounting/customer relationship management/hrms software that helps the small business run their business efficiently anytime, anywhere, on any device. " \
            "You role will be to answer queries/issues/concerns from clients regarding our product suite." \
            "Query: " + input_text
    print(prompt);
    response = index.query(prompt, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

iface.launch(share=True)
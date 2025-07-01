import streamlit as st
from transformers import pipeline

st.title("AI Text Model Interface")

model_selected = st.selectbox(
    "Select model type", 
    ["Text Classification", "Text Generation", "Sentiment Analysis", "Fill Mask", "Summarization"]
)

text = st.text_area("Enter your text here")

if model_selected == "Text Classification":
    model = pipeline("text-classification", model="gpt2")
elif model_selected == "Text Generation":
    model = pipeline("text-generation", model="gpt2")
elif model_selected == "Sentiment Analysis":
    model = pipeline("sentiment-analysis", model="gpt2")
elif model_selected == "Fill Mask":
    model = pipeline("fill-mask", model="bert-base-uncased")
elif model_selected == "Summarization":
    model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


if st.button("Run Model"):
    if len(text.strip()) == 0:
        st.error("Please enter some text first")
    elif model_selected == "Fill Mask" and "[MASK]" not in text:
        st.error("For Fill Mask, you must include [MASK] in the sentence.")
    else:
        with st.spinner("Processing..."):
            result = model(text)

            if model_selected == "Text Classification":
                st.success(result[0]['score'])
            elif model_selected == "Text Generation":
                st.success(result[0]['generated_text'])
            elif model_selected == "Sentiment Analysis":
                st.success(result[0]['score'])
            elif model_selected == "Fill Mask":
                st.success("Top predictions:")
                for res in result:
                    st.write(res['token_str'])
                    st.write(res['sequence'])
                    st.write(res['score'])
                    st.markdown("---")
            elif model_selected == "Summarization":
                st.success("Summary:")
                st.write(result[0]['summary_text'])


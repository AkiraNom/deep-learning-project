import streamlit as st

from model import LSTM
from utils import Processing

st.set_page_config(page_title="Sentiment Analysis", page_icon=":bar_chart:")

col1,col2,col3=st.columns([1,20,1])
with col2:
    st.markdown("""## <center><strong>:grey[Sentiment Analyzer]</strong></center>""", unsafe_allow_html=True)

intro3,intro1,intro2=st.columns([1,5,1])
intro="""The :blue[**sentiment analyzer**] is a machine learning model developed to predict the sentiment of customer comments.
         The LSTM model has been trained for sentiment classification suing the pretrained word and phrase vectors from Google.
     """

with intro1:
    st.write("")
    st.info(intro)
    st.write("")
    with st.expander("**Model Characteristics**"):
        st.write("1. **Model Architecture**: LSTM")
        st.write("2. **Framework**: Pytorch")
        st.write("3. **Training data**: Amazon Review Polarity Dataset")
        st.write("4. **Number of Classes**: 2 (Positive, Negative)")
        st.write("5. **Optimizer**: Adam")
        st.write("6. **Learning Rate**: 0.0001 to 0.00001, adjusted dynamically using CosineAnnealingLR")
        st.write("7. **Loss Function**: Binary Crossentropy")
        st.markdown("""8. <b>Model Author</b>:
                    <ul>
                        <li> <b>Name</b>: Akira N.<br>
                        <li> <b>Email</b>: XXXXXXXX@gmail.com <br>
                    </ul>
                """, unsafe_allow_html=True)
        st.write("""**Additional Note**: codes for model training can be found in the [notebook](https://github.com/AkiraNom/deep-learning-project/blob/main/CNN-Transfer-Learning/.ipynb).""")

    with st.expander("**Model Stats**"):
        st.write("1. **Training Epochs**: 20")
        st.write("2. **Validation Loss**: ")
        st.write("3. **Validation Accuracy**: %")
        st.write("4. **Validation Precision**: ")
        st.write("5. **Validation Recall (Sensitivity)**: ")
        st.write("6. **Validation F1 score**: ")

st.write('')
st.write('')

st.subheader(":gray[Sentiment predictions]", divider=True)

st.write("")

vocab = Processing.load_vocab_library('../model/vocab.pth')
full_model_path = '../model/model.pth'
model_state_dict_path = '../model/model_state_dict.pth'
model = LSTM.load_model(model_state_dict_path,vocab, full_model=False)

tokenizer = Processing.set_tokenizer()

cols = st.columns([1,1])

## User Input
default_txt = "I love this product. It is amazing!"

# Initialize session state for the text input
if "txt" not in st.session_state:
    st.session_state.txt = default_txt

# Initialize session state for the analyze button
if "analyze_triggered" not in st.session_state:
    st.session_state.analyze_triggered = False

# Callback function to handle the "Analyze" action
def analyze_text():
    st.session_state.analyze_triggered = True

# Callback function to handle the "Clear Text" action
def reset_text():
    st.session_state.txt = default_txt

with st.container():
    with cols[0]:
        st.write('**Text Input**')

        placeholder_txt = 'Enter a comment'
        txt = st.text_area('Text to analyze',
                        value=st.session_state.txt,
                        placeholder=placeholder_txt,
                        on_change=analyze_text,
                        height=100,
                        max_chars=400,
                        key='txt',
                        )

        button_cols = st.columns([0.1,1,1,0.1])
        # Analysis button
        button_cols[1].button('Analyze', on_click=analyze_text)
        # Clear text button
        button_cols[2].button('Clear Text', on_click=reset_text)

        if txt == None:
            st.warning('Please type a comment')
            st.stop()

with cols[1]:
    st.write('**Result**')

    if st.session_state.analyze_triggered:
        input_tensor = Processing.preprocess_texts(txt, tokenizer, vocab)
        predicted_class, prob = LSTM.predict_class(model, input_tensor, device='cpu')

        if predicted_class == 1:
            value = 'Positive'
        else:
            value = 'Negative'
        cols = st.columns([0.2,1])
        cols[1].metric(label='Sentiment', value=value)
        cols[1].image('../rsc/img/happy.png' if value == 'Positive' else '../rsc/img/sad.png', width=100)

        pass


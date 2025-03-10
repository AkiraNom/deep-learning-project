from glob import glob
import os
import streamlit as st
# import matplotlib.pyplot as plt

from model import load_pretrained_model, predict_class
from utils import ImageProcessing, load_category_names, ResultAnalysis


st.set_page_config(page_title="Flower Image Classifier", page_icon=":bar_chart:")

col1,col2,col3=st.columns([1,20,1])
with col2:
    st.markdown("""## <center><strong>:cherry_blossom: :grey[Flower Image Classifier] :cherry_blossom:</strong></center>""", unsafe_allow_html=True)

intro3,intro1,intro2=st.columns([1,5,1])
intro=""":maple_leaf: The :blue[**Flower Image Classifier**] is a machine learning model designed to predict the class of flower images.
         The pretrained model, along with its weights has been fine-tuned with data augmentation specifically for flower image classification.
         impbalance data handling techniques have been implemented to improve the model's performance. The model has been trained on the Oxford 102 Flower Dataset.
     """

with intro1:
    st.write("")
    st.info(intro)
    st.write("")
    with st.expander("**Model Characteristics**"):
        st.write("1. **Model Architecture**: MobileNetV3")
        st.write("2. **Framework**: Pytorch")
        st.write("3. **Training data**: Oxford 102 Flower Dataset")
        st.write("4. **Number of Classes**: 102")
        st.write("5. **Optimizer**: Adam")
        st.write("6. **Learning Rate**: 0.001 to 0.0001, adjusted dynamically using CosineAnnealingLR")
        st.write("7. **Loss Function**: Crossentropy")
        st.write("8. **Input Size**: 128x128pixels")
        st.write("9. **Early stop**, **Early stop patience**: True, 10")
        st.write("10. **Data Augmentation**: Horizontal Flip, RandomClip, Rotation, ColorJitter")
        st.write("11. **Imbalance Data Handling**: class weighting")
        st.markdown("""12. <b>Model Author</b>:
                    <ul>
                        <li> <b>Name</b>: Akira N.<br>
                        <li> <b>Email</b>: XXXXXXXX@gmail.com <br>
                    </ul>
                """, unsafe_allow_html=True)
        st.write("""**Additional Note**: codes for fine-tuning and validating the model can be found in the [notebook](https://github.com/AkiraNom/deep-learning-project/blob/main/CNN-Transfer-Learning/Flower_image_classification/flower_image_classification_using_transfer_learning.ipynb).""")

    with st.expander("**Model Stats**"):
        st.write("1. **Training Epochs**: 200")
        st.write("2. **Validation Loss**: 0.2694")
        st.write("3. **Validation Accuracy**: 95.07%")

    with st.expander("**Recognized Flower Classes**"):
        classes = load_category_names()
        st.write('')
        st.write("&nbsp; The model recognizes the following 102 flower classes:", unsafe_allow_html=True)
        st.write(classes.values())

st.write('')
st.write('')

st.subheader(":gray[Test the Classifer Model]", divider=True)

st.write("")


model = load_pretrained_model(params_path='./model/flower_classifier_model.pth')

## Select a flower image
image_source = None

st.write("**Upload your own image**")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    image_source = uploaded_file

st.warning("The latest version of streamlit may have an issue with uploading an image file.")

st.write("")
on = st.toggle("Select from the test images")
if on:
    path = glob('./img/test/*.jpg')
    # Extract file names from paths
    file_names = [os.path.basename(p) for p in path]
    selected_file = st.selectbox("Choose an image", file_names)
    image_source = path[file_names.index(selected_file)]

st.write("")
st.markdown(f"""###### <center>Class Prediction</center>""", unsafe_allow_html=True)

if image_source is None:
    st.warning("Please upload an image or select from the test images to proceed.")
else:
    ## Display the selected image
    st.write("")
    cols = st.columns([1,1])
    with cols[0]:
        st.write("<center><strong>Selected Flower Image</strong></center>", unsafe_allow_html=True)
        st.image(image_source, use_container_width=True)


    ## Class prediction
    image = ImageProcessing.load_image(image_source)
    input_batch = ImageProcessing.preprocess_iamge(image)
    probs = predict_class(model, input_batch)

    top_class_probs, top_class_name, plot_title = ResultAnalysis.find_top_classes(probs, classes)

    with cols[1]:
        with st.container():
            st.markdown(f"""<center><strong>Prediction Result</strong></center>""", unsafe_allow_html=True)
            st.write("")

            subcols = st.columns([0.5,2,2,0.5])
            subcols[1].metric(label="**Class**",value=top_class_name[-1], label_visibility="visible", border=False)
            subcols[2].metric(label="**Probability**",value=str(f'{top_class_probs[-1]*100:.1f}')+"%", label_visibility="visible", border=False)

            with st.expander("**Sanity Check: Display Most Probable Flower Classes**"):

                st.plotly_chart(ResultAnalysis.sanity_check(top_class_probs, top_class_name, plot_title))



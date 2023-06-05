import streamlit as st
from model_class import CNN
import main

st.set_page_config(page_title= "page2circuit",layout='centered')
st.title("page2circuit")
st.write('transform hand-drawn circuit into beautiful diagram')
st.markdown("""---""")

img = st.file_uploader(label='Upload your hand-drawn circuit image', type=["png", "jpg", "jpeg"])

if img is None:
    st.write('download circuit images below for testing')
    st.image('test_images/test2.jpeg', caption='Sample Image', use_column_width=True)

st.markdown("""---""")
col1, col2 = st.columns(2)
if img is not None:
    st.success("Image uploaded successfully")
    col1.write("Image Preview", )
    col1.image(img, caption='Uploaded Image', use_column_width=True)
    col2.write("Result")

    is_preprocess = False
    components_ext = []
    
    result_img, transition_imgs, components_ext, rects = main.preprocess_img(img) 
    if result_img is None:
        st.stop()
    st.success("Image processed successfully")
    is_preprocess = True
    
    if len(components_ext) != 0:
        classified, annoted_img = main.classify_components(components_ext, img, rects)  
        if len(classified) == 0:
            st.stop()

        col2.image(annoted_img, caption='Processed Image', use_column_width=True)
    st.markdown("""---""")

if img is not None and is_preprocess:
    st.subheader("transitional steps")
    col1, col2, col3 = st.columns(3)
    col1.image(img, caption='Uploaded Image', use_column_width=True)
    col2.image(transition_imgs[0], caption='Step 1: Adaptive Thresholding', use_column_width=True)
    col3.image(transition_imgs[1], caption='Step 2: Noise Removal & Dilation', use_column_width=True)
    col1, col2, col3 = st.columns(3)
    col1.image(transition_imgs[2], caption='Step 3: Clustering Anchor Points for Component Detection', use_column_width=True)
    col2.image(transition_imgs[3], caption='Step 4: Applying FAST for Corner Detection', use_column_width=True)
    col3.image(transition_imgs[4], caption='Step 5: Combining Clustering Outputs', use_column_width=True)
    col1, col2, col3 = st.columns(3)
    col1.image(transition_imgs[5], caption='Step 6: Removing Components', use_column_width=True)
    col2.image(annoted_img, caption='Step 7: Classifying Components', use_column_width=True)
    col3.image(result_img, caption='Step 8: Mapping Connections & making Contours', use_column_width=True)
    st.markdown("""---""")
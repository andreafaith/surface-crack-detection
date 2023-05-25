# import streamlit as st
# import tensorflow as tf

# @st.cache(allow_output_mutation=True)
# def load_model():
#   model=tf.keras.models.load_model('SurfaceCrackDetection.h5')
#   return model
# model=load_model()
# st.write("""
# # Surface Crack Detection System"""
# )
# file=st.file_uploader("Choose a photo from computer",type=["jpg","png"])

# import cv2
# from PIL import Image,ImageOps
# import numpy as np
# def import_and_predict(image_data,model):
#     size=(120,120)
#     image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
#     img=np.asarray(image)
#     img_reshape=img[np.newaxis,...]
#     prediction=model.predict(img_reshape)
#     return prediction
# if file is None:
#     st.text("Please upload an image file")
# else:
#     image=Image.open(file)
#     st.image(image,use_column_width=True)
#     prediction=import_and_predict(image,model)
#     class_names=['Looks like there is a crack on that image you just provided','Looks like there is no crack on that image you just provided']
#     string="PREDICTION : "+class_names[np.argmax(prediction)]
#     st.success(string)


import streamlit as st
import tensorflow as tf

model = model=tf.keras.models.load_model('SurfaceCrackDetection.h5')

def detection(image):
    """ Run detection on uploaded image """
    results = model([image])
    return results

def draw_bounding_boxes(image, boxes, labels, confidences):
    """ Drawing bounding boxes on detected objects on image 
        whether it has facemask or none """
    draw = ImageDraw.Draw(image)
    for box, label, confidence in zip(boxes, labels, confidences):
        color = 'red' if label == 'NO-Mask' else 'green'  # Use red for no_mask, green for mask
        draw.rectangle(box, outline=color, width=5)
        draw.text((box[0], box[1] - 20), f"{label}: {confidence:.2f}", fill=color)
    return image

def main():
    """ Displaying window using Streamlit API """
    st.header("Surface Crack Detection")
    st.write("""
    #####
    """)

    img = st.file_uploader("Choose an image from your computer", type=['jpg', 'png'])

    col1, col2 = st.columns(2)
    if img is not None:
        with col1:
            image = Image.open(img)
            st.image(image, caption='Uploaded image', use_column_width=True)
        with col2:
            image = Image.open(img)
            results = detection(image)
            boxes = results.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
            labels = results.pandas().xyxy[0]['name'].tolist()
            confidences = results.pandas().xyxy[0]['confidence'].tolist()

            # Display the image with bounding boxes and labels
            image_with_boxes = draw_bounding_boxes(image.copy(), boxes, labels, confidences)
            st.image(image_with_boxes, caption='With bounding boxes', use_column_width=True)

        st.success("Image successfully detected!")
    else:
        default_image = 'assets/download.jpg'
        with col1:
            st.image(default_image, caption='Sample Image', use_column_width=True)
        with col2:
            image = Image.open(default_image)
            results = detection(image)
            boxes = results.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
            labels = results.pandas().xyxy[0]['name'].tolist()
            confidences = results.pandas().xyxy[0]['confidence'].tolist()

            # Display the image with bounding boxes and labels
            image_with_boxes = draw_bounding_boxes(image.copy(), boxes, labels, confidences)
            st.image(image_with_boxes, caption='With bounding boxes', use_column_width=True)

        st.write("You can upload an image that have a crack or not")

    st.write(
        """
        The following link will redirect you to the Colab where the training of the model occurs:
        """
    )

if __name__ == '__main__':
    main()
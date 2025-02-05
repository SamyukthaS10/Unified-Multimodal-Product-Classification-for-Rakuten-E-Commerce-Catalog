import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Custom CSS for Soft Purple Background with Black Text
st.markdown("""
    <style>
        body {
            background-color: #EDE7F6; /* Soft purple background */
            color: #000000; /* Default black text */
        }
        .streamlit-expanderHeader {
            color: #000000;
        }
        .stSidebar {
            background-color: #D1C4E9; /* Lighter purple sidebar */
            color: #000000; /* Black text for sidebar */
        }
        h1, h2, h3, h4 {
            color: #000000; /* Black color for all heading levels */
        }
        .stTitle, .stSubheader, .stText, .stMarkdown {
            color: #000000; /* Black color for text */
        }
        a {
            color: #6200EA; /* Dark purple links */
        }
        a:hover {
            color: #BB86FC; /* Lighter purple for hover effect */
        }
    </style>
""", unsafe_allow_html=True)


# Sidebar - Navigation
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Select a section:",
    [
        "Introduction & Objective",
        "Datasets",
        "Exploratory Analysis",
        'Preprocessing & Feature Engineering',
        "Model Strategy",
        "Models",
        "Multimodal"
    ]
)

# 1. Load Pre-trained Models and Supporting Files
text_model = TFDistilBertForSequenceClassification.from_pretrained(
    '/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/DistilBERT_Final_model/DistilBERT_with_LRDecay_model_5e-5')
tokenizer = DistilBertTokenizer.from_pretrained(
    '/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/DistilBERT_Final_model/DistilBERT_with_LRDecay_tokenizer_5e-5')
image_model = load_model(
    '/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/my_model_VGG16_reducelr_1e-5.keras')
text_label_encoder = joblib.load(
    '/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/text_label_encoder.joblib')

with open('/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/class_indices_my_model_VGG16.json', 'r') as f:
    image_class_indices = json.load(f)
    image_label_decoder = {v: k for k, v in image_class_indices.items()}

# 2. Preprocessing Functions
def preprocess_text(text):
    """Preprocess text for the DistilBERT model."""
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
    return inputs['input_ids'], inputs['attention_mask']

def preprocess_image(img_path):
    """Preprocess image for the VGG16 model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)
    return img_array

def late_fusion_weighted_soft_voting(text_input, img_path, text_label_encoder, image_label_decoder, text_weight=0.5, image_weight=0.5):
    """Perform late fusion using weighted soft voting with probability outputs from both models."""
    # Initialize probabilities
    text_probs, image_probs = None, None
    
    if text_input:
        text_ids, text_mask = preprocess_text(text_input)
        text_logits = text_model(text_ids, attention_mask=text_mask).logits.numpy()
        text_probs = np.exp(text_logits) / np.sum(np.exp(text_logits), axis=1, keepdims=True)  # Softmax conversion

    if img_path:
        img_array = preprocess_image(img_path)
        image_probs = image_model.predict(img_array)

    if text_probs is not None and image_probs is not None:
        # Weighted fusion of probabilities
        combined_probs = (text_probs * text_weight) + (image_probs * image_weight)
    elif text_probs is not None:
        combined_probs = text_probs
    elif image_probs is not None:
        combined_probs = image_probs
    else:
        return None, None  # No inputs provided

    # Get final class index with maximum probability
    final_class = np.argmax(combined_probs, axis=1)[0]

    # Decode the final class index to original labels
    final_label = (
        text_label_encoder.inverse_transform([final_class])[0]
        if text_probs is not None
        else image_label_decoder.get(final_class, "Unknown Category")
    )

    return final_label, combined_probs[0]



# Main Section - Dynamically load content
if menu == "Introduction & Objective":
    st.title("Unified Multimodal Product Classification for Rakuten’s E-Commerce Catalog")

    st.markdown("""
    In Rakuten France, cataloging millions of product listings efficiently is a complex task. Products sourced 
    from both professional and non-professional merchants come with titles, images, and additional descriptions 
    that must be classified into predefined product type codes. This classification underpins the platform’s 
    e-commerce functionalities but is complicated by noisy data, unbalanced class distributions, and the scale of modern catalogs.
    
    This project addresses the challenge of large-scale multimodal product type code classification, 
    where the goal is to accurately predict product type codes using both text and image data, along 
    with additional descriptions when available. For instance, a product titled Klarstein Présentoir 2 Montres 
    Optique Fibre with a corresponding image is classified under a specific product type code. By developing a scalable classifier, 
    the project aims to automate the categorization process, improving efficiency and reducing duplication across the platform.
    """)

    st.subheader("Objective:")

    st.markdown("""
    The main objective of this project is to develop a Multimodal Product Data Classification system 
    that combines Natural Language Processing (NLP) and Convolutional Neural Networks (CNN) to predict 
    the product type based on either the product's description title or its image. The steps carried out 
    in the project are listed below:

    1. Understanding the context of the problem
    2. Data preprocessing (cleaning, translation of text, categorizing)
    3. Data visualization
    4. Feature Engineering (data transformation, lemmatization/stemming, tokenization, vectorization)
    5. Modeling (simple models, transfer models)
    6. Hyperparameter optimization
    7. Feature fusion of text and image model
    8. Documentation on GitHub
    """)

elif menu == "Datasets":
    st.title("Datasets")
    st.write("""The dataset used in this project was provided by the Rakuten Institute of Technology as 
             part of their challenge data. It is freely accessible and open to the public.
             """)
    
    st.subheader("Textual dataset")
    st.markdown("""
    The textual data used for training the model consists of the X_train.csv and Y_train.csv files, which 
    together contain a total of 84,916 records.
    """)
    
    # Bold caption using Markdown
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>X_train dataset</h4>", unsafe_allow_html=True)
    st.image(
        "/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/x_train_text_dataset.png", 
        caption="Figure 1. X_train dataset", 
        use_container_width=True
    )
     # Add space between subsections
    st.write("")  # Empty space
    st.write("") 
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Y_train dataset</h4>", unsafe_allow_html=True)
    st.image(
        "/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/y_train_text_dataset.png", 
        caption="Figure 2. Y_train dataset", 
        use_container_width=True
    )     
    # Add space between subsections
    st.write("")  # Empty space
    st.write("") 
    st.subheader("Image dataset")
    st.markdown("""
    The image dataset used for training the model was created from the image_train file provided by the Rakuten project on the ChallengeData platform 
    (Link: https://challengedata.ens.fr/challenges/35). The image_train file contain a total of 84,916 product images. Each image is labelled with both 
    a product ID and an image ID, which correspond to the entries in the X_train dataset.
    """)
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Image_train folder</h4>", unsafe_allow_html=True)
    st.image(
        "/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/image_train_data.png", 
        caption="Figure 3. Image_train data", 
        use_container_width=True
    )
                
elif menu == "Exploratory Analysis":
    st.title("Exploratory Analysis")
    st.write("""
             The relationships between the text variables and the target classes were thoroughly analyzed, 
             revealing significant correlations between specific terms and product categories. Advanced text 
             embedding techniques were employed to effectively capture these relationships. Additionally, 
             image data was incorporated, providing valuable contextual features that contributed to improving 
             classification accuracy.
             """)
    
     # Tabs for organized data
    tab1, tab2, tab3 = st.tabs(["Textual Data Exploration", "Image Data Exploration", "Key Observations"])
    
    with tab1:
        st.subheader("Textual Data Exploration")
        
        # Subsection 1
        st.markdown("### 1. Imbalance in product category or prdtypecode")
        st.image("/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/categorical_distribution.png", caption="Figure 4. Number of Products in Each Category")
        st.markdown("""
        - **Pool and Accessories**: Highest number of products (10,209 items).  
        - **Trading Cards**: Fewest products (764 items).  
        """)
                
        # Add space between subsections
        st.write("")  # Empty space
        st.write("") 
        
        # Subsection 2
        st.markdown("### 2. Distribution of Product Designation (Title) Text")
        st.image("/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/product_designation_length.png", caption="Figure 5. Distribution of Product Designation Lengths")
        st.markdown("""
        - **Games Consoles**: Longest designation length with fewest outliers.  
        - **Used Newspaper and Magazines**: Exhibits a wide range of designation lengths.  
        - **New Books**: Shortest designation length on average.  
        """)
        
        # Add space between subsections
        st.write("")  # Empty space
        st.write("") 
        
        # Subsection 3
        st.markdown("### 3. Distribution of Product Description Text")
        st.image("/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/product_description_length.png", caption="Figure 6. Distribution of Product Description Lengths")
        st.markdown("""
        - **PC Video Games**: Longest description length (median ~2,500 characters).  
        - Categories such as **Used Books** and **Newspapers** often lack descriptions.  
        """)
        
        # Add space between subsections
        st.write("")  # Empty space
        st.write("") 
        
        # Subsection 4
        st.markdown("### 4. Products Without Descriptions")
        st.image("/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/products_with_and_without_description.png", caption="Figure 7. Proportion of Products with and without Descriptions")
        st.markdown("""
        - **Board Games and Role-Playing Games**: Highest proportion without descriptions.  
        - **PC Video Games**: All products include descriptions.  
        """)
        
        # Add space between subsections
        st.write("")  # Empty space
        st.write("") 

        # Subsection 5
        st.markdown("### 5. Frequent Words in Titles and Descriptions")
        st.image("/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/wordcloud_general.png", caption="Figure 8. Frequent Words in Product Titles/Descriptions")
        st.markdown("""
        - Common terms include 'stainless steel,' 'package,' and 'include.'  
        - These frequent words are likely due to their prevalence in product specifications.  
        """)

    with tab2:
        st.subheader("Image Data Exploration")
        
        # Subsection 1
        st.markdown("### 1. Blank Images")
        st.image("/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/blank_images.png", caption="Figure 9. Number of Blank Images Across Categories")
        st.markdown("""
        - Total of 7 blank images detected.  
        - **Decoration** category has the most blank images (3).  
        """)
        
        # Add space between subsections
        st.write("")  # Empty space
        st.write("") 

        # Subsection 2
        st.markdown("### 2. Image Quality Assessment")
        st.image("/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/image_sharpness.png", caption="Figure 10. Proportion of Images by Sharpness Category")
        st.markdown("""
        - **91%** of images are sharp and of high quality.  
        - A small proportion (1.51%) is blurry.  
        """)
        
        st.image("/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/image_sharpness_distribution.png", caption="Figure 11. Distribution of Image Sharpness Across Categories")
        st.markdown("""
        - **Garden Tools** category has the highest sharpness scores.  
        - **PC Video Games**: Highest median sharpness value.  
        """)
        
        
             
        # Subsection 1
        st.markdown("### 3. Distribution of Text presence in Product Images")
        st.image("/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/images_with_text.png", caption="Figure 12. Images with Text Across Categories")
        st.markdown("""
        - **New Books**: This category has the highest number of images with text, as expected, since books typically have textual information on their covers.
        - **Pet Store / Baby Socks / Small Photos**: These categories have the least number of images with text, as many product images in these categories may focus on the visual appeal of the product with minimal or no text.
        - Other categories show a moderate amount of text in their images, suggesting that labels or model names are commonly included in these products.
        """)

    with tab3:
        st.subheader("Key Observations")
        st.markdown("### Dataset Challenges and Impact on Model Performance")
        st.markdown("""
        - **Imbalanced Data:** Some product categories have more samples than others, leading to potential model bias.
        - **Image Quality Issues:** Low-quality or inconsistent product images may limit the model's ability to extract features.
        - **Missing/Inconsistent Data:** Certain products lack images or complete descriptions, reducing dataset size for training.
        - **Multilingual Text Data:** Product descriptions in multiple languages (primarily French) require language-specific preprocessing.
        - **Noisy Text Data:** Spelling errors, missing information, and irrelevant terms (special characters, HTML tags) affect text quality.
        """)
        
    
        
elif menu == "Preprocessing & Feature Engineering":
    st.title("Preprocessing & Feature Engineering")
    st.write("""Based on the insights from the exploratory analysis, we implemented the following preprocessing steps to address data quality issues and enhance the 
             dataset for model training.""")
    
    # Tabs for organized data
    tab1, tab2, tab3 = st.tabs(["Textual Data Pre-processing", "Image Data Pre-processing", "Target Variable"])
    
    with tab1:
        st.subheader("Textual Data Pre-processing")

        # 1. Handling Missing Data
        st.write("**1. Handling Missing Data**")
        st.markdown("- **Observation:** 35.09% of product descriptions were missing.\n"
        "- **Action:** Created a new column, title_description, by concatenating designation and description columns.\n"
        "- **Reasoning:** Combining columns mitigates data imbalance and ensures that essential product information is preserved for accurate classification."
    )
        
        # 2. Data Cleaning
        st.write("**2. Data Cleaning**")
        st.markdown("""
        - Checked for duplicates and missing (NaN) values in the dataset.
        - **Reasoning:** Ensures the dataset is free from redundant or incomplete entries, reducing noise and errors during modeling.
        """)

        # 3. Text Pre-processing
        st.write("**3. Text Pre-processing**")
        st.markdown("- **Translation:** Used GoogleTrans API to translate text present in column `title_description` into English and stored in a column `description_english` .\n"
        "- **Reasoning:** Since the dataset contains multilingual text (French/German/English), translating it to a single language ensures consistency for downstream processing and model compatibility.\n"
        "- **Data Cleaning before Translation:**\n"
        "    - Restored special characters using Latin encoding.\n"
        "    - Removed HTML tags, special characters, and punctuation.\n"
        "    - Converted text to lowercase for consistency.\n"
        "- **Reasoning:** Cleaning text improves model interpretability by removing irrelevant noise and standardizing the input format."
    )

        # 4. Pre-processing Steps for Exploratory Data Analysis
        st.write("**4. Pre-processing Steps for Exploratory Data Analysis**")
        st.markdown("- **Tokenization:** Split text into individual tokens.\n"
        "- **Reasoning:** Tokenization is a foundational step to analyze word frequencies and distributions.\n"
        "- **Stop Words Removal:** Removed common stop words (e.g., 'and', 'the').\n"
        "- **Reasoning:** Eliminating stop words highlights meaningful content for analysis and visualization.\n"
        "- **WordCloud:** Visualized top 200 frequent words.\n"
        "- **Reasoning:** WordCloud provides a quick overview of prominent terms, revealing patterns and dominant themes in the dataset."
    )

        # 5. Pre-processing Steps for Text Classification Model
        st.write("**5. Pre-processing Steps for Text Classification Model**")

        # For Machine Learning Models
        st.markdown("- **Machine Learning Models:**\n"
        "    - Tokenized text, applied stemming and lemmatization, and used vectorization methods like CountVectorizer and TF-IDF.\n"
        "    - **Reasoning:** These steps reduce words to their base forms, ensuring features are more generalizable while creating a numerical representation of text for traditional ML models.\n"
        "- **Transfer Learning Models:**\n"
        "    - Used DistilBERT tokenizer and preprocessed text as per model requirements.\n"
        "    - **Reasoning:** DistilBERT tokenizer is optimized for pre-trained models and ensures input text adheres to expected tokenization patterns, leveraging contextual embeddings."
    )

            
    with tab2:            
        # 6.2 Image Data Pre-processing Section
        st.header("Image Data Pre-processing")
        
        st.write("**1. Image Size and Resolution**")
        st.markdown("""
        - **Observation:** The images in the dataset are of same size (500 x 500), but the object in the images vary in size and resolution.
        - **Action:** Extracted the image size (width and height) and pixel resolution for each image.
        - **Reasoning:** Understanding the image size and resolution helps in determining the quality of the images. It also ensures that all 
        images are standardized and appropriately resized for model training. Large images may need to be resized to improve computational efficiency 
        and avoid memory overload during training.
        """)

        # 2. Sharpness Calculation
        st.write("**2. Sharpness Calculation**")
        st.markdown("""
        - **Observation:** Sharpness is an important feature that indicates the clarity of an image.
        - **Action:** Calculated sharpness using the Laplacian variance method, which measures the variance in pixel intensity to determine the image sharpness.
        - **Reasoning:** Sharpness is crucial for distinguishing high-quality images from blurry ones. Images with low sharpness may not provide useful information for classification tasks. 
        By quantifying sharpness, we can filter out blurry images and ensure that the dataset is of high quality for training machine learning models.
        """)

        # 3. Text Detection on Images
        st.write("**3. Text Detection on Images**")
        st.markdown("""
        - **Observation:** Some images contain text that may provide valuable information for product classification.
        - **Action:** Used Optical Character Recognition (OCR) to detect and extract text from the images where applicable.
        - **Reasoning:** Extracting text from images can provide additional features, especially for e-commerce product images. For instance, brand names, model numbers, or other product-related text could assist in categorizing products more accurately. 
        Including this feature enriches the dataset and can improve model performance, particularly in text-based tasks. unfornately, the extracted text could not by captured precisely due to 
        the low resolution of product in the images.  
        """)

        # 4. Blank Image Detection
        st.write("**4. Blank Image Detection**")
        st.markdown("""
        - **Observation:** Some images may be blank or contain no relevant information.
        - **Action:** Identified and removed blank/solid images by comparing pixel values (e.g., detecting images with no variation in pixel intensity).
        - **Reasoning:** Blank or solid-color images add no value to the dataset and could negatively impact model training. Removing these images ensures that the model is trained on relevant 
        data, which improves its ability to generalize.
        """)

        # 5. Pre-processing for Image Classification Model
        st.write("**5. Pre-processing for Image Classification Model**")
        st.markdown("""
        - **Action:** After feature extraction, the images were preprocessed for use in classification models. This involved:
            - Resizing images to a consistent size (e.g., 224x224 pixels).
            - Augmenting the dataset through transformations like rotations, flipping, and cropping to improve model robustness.
        - **Reasoning:** Image preprocessing ensures that the data is in a format suitable for machine learning models. Resizing standardize the input for neural networks, 
        preventing issues with varying input sizes and scale. Augmentation improves generalization by exposing the model to a broader range of variations in the images.
        """)
    
    with tab3:
        # 6.3 Target Variable
        st.header("Target Variable")

        st.markdown("""
        - For exploratory data analysis, a new column `product category` was created.
        - This column represents the product categorization used on Rakuten for each `prdtypecode` (product type code).
        """)
        
elif menu == "Model Strategy":
    st.title("Rakuten E-commerce Project: Model Strategy and Testing")
    st.write("""
        In this project, we employed a comprehensive strategy for testing and optimizing machine learning models and transfer learning models.
        Initially, traditional machine learning models were tested and evaluated. We then transitioned to transfer learning models (DistilBERT for text classification and VGG16 for image classification) to enhance classification accuracy.
    """)

    # Strategy for Model Testing and Selection
    st.header("1. Strategy for Model Testing and Selection")
    st.write("""
        The Rakuten E-commerce Project employed a structured strategy for model testing. Initially, we tested traditional machine learning models and evaluated their performance.
        Subsequently, we applied advanced transfer learning models to improve classification accuracy. Below is an overview of how different models were tested, optimized, and evaluated.
    """)
    
    st.header("2. Feature and Target Variables")
    st.write("""
        - For **text classification model**, the column `description_english` was used as feature and column `prdtypecode` was used as target variable.
        - For **image classification model**, the `image_path` which directs to the image file was used as feature and `prdtypecode` as target variable.
        """)
    
    # Assessment Methods
    st.header("3. Handling Class Imbalance")
    st.subheader("Textual Dataset")
    st.write("""
        - **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-Sampling Technique) was applied to machine learning models.
        - **Class Weight Method**: This method was used for transfer learning models.
    """)

    st.subheader("Image Dataset")
    st.write("""
        - **Class Imbalance Handling**: Data augmentation was used by generating three augmented images for each image in the minority classes.
        - Augmentation was applied exclusively to the **training set** to improve model generalization.
    """)

    # Classification Problem Overview
    st.header("4. Classification of the Problem")
    st.write("""
        The project involves **multimodal classification**, combining both **text classification** and **image classification** tasks.
    """)
    st.subheader("Example for a Multimodal Classification")
    st.write("**Text Classification**")
    st.write("""
        - **DistilBERT** can be used for classifying textual data into predefined categories.
    """)

    st.write("**Image Classification**")
    st.write("""
        - **VGG16** can be employed to categorize images into different classes.
    """)

    # Metrics Used for Model Comparison
    st.header("5. Metrics Used to Compare the Models")
    st.write("""
        We used the following metrics to evaluate and compare model performance:
        - **Validation Accuracy**: Measures the overall correctness of the model on the validation set.
        - **Validation Loss**: Assesses how well the model minimizes errors during training.
        - **F1-Score**: A balanced measure of precision and recall, especially useful for imbalanced datasets.
    """)
    
    
    # Table 1: Machine Learning Model Results for Text Classification
    st.header("6. Machine Learning Model Results for Text Classification")
    st.write("""
        Below are the results for various machine learning models trained on 80% of the dataset and with different vectorizers i.e., **CountVectorizer** and **TF-IDF Vectorizer**.
        The results show validation accuracy and F1-Score for different models tested with the remaining 20% of the dataset.
    """)

    # Display the first table: Machine Learning Models
    data = {
        'Tokenization/Vectorization': ['Simple Tokenizer, CountVectorizer', 'Simple Tokenizer, CountVectorizer', 'Simple Tokenizer, CountVectorizer', 
                                    'Simple Tokenizer, CountVectorizer', 'Simple Tokenizer, CountVectorizer', 'Simple Tokenizer, CountVectorizer',
                                    'Simple Tokenizer, TF-IDF Vectorizer (features=5000)', 'Simple Tokenizer, TF-IDF Vectorizer (features=5000)', 
                                    'Simple Tokenizer, TF-IDF Vectorizer (features=5000)', 'Simple Tokenizer, TF-IDF Vectorizer (features=5000)'],
        'Model': ['SVC', 'Logistic Regression', 'RFC', 'Multinomial NB', 'XGBoost', 'KNN', 'SVC', 'Logistic Regression', 'RFC', 'KNN'],
        'Validation Accuracy': [0.6477, 0.6864, 0.6741, 0.6652, 0.6100, 0.5313, 0.7555, 0.7333, 0.7148, 0.5457],
        'F1-Score': [0.6579, 0.6923, 0.6744, 0.6643, 0.6333, 0.5471, 0.7586, 0.7350, 0.7176, 0.5682]
    }
    df = pd.DataFrame(data)
    st.table(df)

    st.write("""
        From the table, we can see that **SVC**, **Logistic Regression**, and **Random Forest Classifier (RFC)** performed the best when using **TF-IDF Vectorizer**. These models achieved better accuracy and F1-scores compared to others like **KNN** and **GaussianNB**.
    """)

    # Advanced NLP Models
    st.header("7. Advanced NLP Models")
    st.write("""
        We also tested various **BERT** variants for text classification. **DistilBERT**, **ALBERT**, and **RoBERTa** were the primary models tested for text classification. 
        The models were trained with 80% of dataset for training and 20% for validation. 
    """)

    # Table 2: NLP BERT Model Results
    st.write("""
        Below are the results for the BERT variants tested for text classification.
    """)

    # Display the second table: BERT Models
    data_bert = {
        'Tokenizer': ['DistilBERT', 'ALBERT', 'RoBERTa'],
        'Model': ['DistilBERT-base-uncased (learning rate= 3e-5) (Epoch 4)', 'Albert(learning rate= 1e-5) (Epoch 3)', 'Roberta(learning rate=1e-5) (Epoch 3)'],
        'Validation Accuracy': [0.8300, 0.7800, 0.8087]
    }
    df_bert = pd.DataFrame(data_bert)
    st.table(df_bert)

    st.write("""
        **DistilBERT** yielded the best performance with a validation accuracy of **83%**, as shown in the table above.
    """)

    # Optimization Results for DistilBERT
    st.header("8. Optimization Results for DistilBERT")
    st.write("""
        Various hyperparameters were experimented with for DistilBERT, including learning rates, batch sizes, and train-validation-test splits.Here we used 70% or 80% of dataset for training,
        and the remaining was split into half, where the first half is used for validation and second half is used for testing the model. \n
        The following table shows the configurations and results obtained for **DistilBERT** model:
    """)

    # Table 3: DistilBERT Optimization Results
    data_distilbert = {
        'Tokenizer': ['DistilBERT', 'DistilBERT', 'DistilBERT', 'DistilBERT', 'DistilBERT', 'DistilBERT', 'DistilBERT'],
        'Model': [
            'DistilBERT-base-uncased (learning rate= 4e-5), Epoch =6, batch =32 (70, 15, 15 split)', 
            'DistilBERT-base-uncased (learning rate= 1e-6), Epoch =10, batch =32 (80, 10, 10 split)', 
            'DistilBERT-base-uncased (learning rate= 4e-5), Epoch =6, batch =16 (70, 15, 15 split)',
            'DistilBERT-base-uncased (learning rate= 3e-5), Epoch =6, batch =16 (80, 10, 10 split)', 
            'DistilBERT-base-uncased (learning rate= 3e-5), Epoch =6, batch =32 (80, 10, 10 split)', 
            'DistilBERT-base-uncased (learning rate= 4e-5), Epoch =10, batch =32 (80, 10, 10 split)', 
            'DistilBERT-base-uncased (learning rate= 5e-5), Epoch =10, batch =32 (80, 10, 10 split)'
        ],
        'Test Accuracy': [0.8321, 0.744, 0.835, 0.84, 0.8387, 0.8434, 0.8482],
        'F1-Score': [0.8174, 0.6817, 0.8209, 0.8236, 0.8236, 0.8294, 0.8382]
    }
    df_distilbert = pd.DataFrame(data_distilbert)
    st.table(df_distilbert)

    st.write("""
        The best performance for DistilBERT was achieved with a **learning rate of 5e-5**,**Epoch 10**, **batch size 32**, and an **80/10/10 train-validation-test split**, achieving an **F1-score of 83.82%**.
    """)

    # Image Classification Results
    st.header("9. Image Classification Results")
    st.write("""
        For image classification, we transitioned from custom CNN models to pre-trained models like **VGG16** and **ResNet50**
        We used 80% of dataset for training and the remaining was split into half, where the first half is used for validation and second half is used for testing the model.Data augmentation was used to address class imbalance, 
        and **VGG16** outperformed **ResNet50** in terms of accuracy and F1-score.
    """)

    # Table 4: Image Classification Results
    st.write("""
        Below are the results for the **VGG16** and **ResNet50** models, showing their performance in terms of validation accuracy and loss.
    """)

    # Display the image classification table
    data_image = {
        'Model': ['ResNet50 (Epoch 10)', 'ResNet50 (base_layer 10) (Epoch 10)', 'VGG16 (Epoch 25)'],
        'Learning Rate': ['1e-4', '1.95e-07', '1e-04'],
        'Validation Accuracy': [0.2668, 0.4345, 0.5500],
        'Validation Loss': [2.5473, 1.9663, 1.5377]
    }
    df_image = pd.DataFrame(data_image)
    st.table(df_image)

    st.write("""
        The **VGG16 (Epoch 25) ** model achieved the best performance with a **validation accuracy of 55%** and **validation loss of 1.5377**. This shows that VGG16 outperformed ResNet50 in terms of classification accuracy and loss.
    """)

 
elif menu == "Models":
    st.title("Models")
    st.write("**Text Model:** DistilBERT-base-uncased (learning rate = 5e-5, Epoch = 10, batch_size = 32, split = 80/10/10)\n"
         "**Image Model:** VGG16 (learning rate = 1e-4, Epoch = 25, batch_size = 16, split = 80/10/10)")

    
     # Tabs for organized data
    tab1, tab2 = st.tabs(["Textual Model", "Image Model"])
    
    with tab1:      
        st.subheader("DistilBERT-base Model for Text Classification")
        
        # Description of DistilBERT
        st.markdown("""
        - **Architecture**: DistilBERT is a smaller, faster version of BERT (Bidirectional Encoder Representations from Transformers), designed to reduce the size and computational cost while maintaining performance.
        - **Use Case**: Primarily used for Natural Language Processing (NLP) tasks such as text classification, sentiment analysis, and named entity recognition.
        - **Advantages**:
            - **Faster** and more **memory-efficient** than BERT.
            - Retains **over 95% of BERT's performance** despite being smaller.
            - Ideal for NLP tasks where speed and efficiency are critical.
        """)
        
        # Images related to DistilBERT
        st.markdown("<h3 style='font-weight: bold;'>Confusion Matrix</h3>", unsafe_allow_html=True)
        st.image("/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/distilbert_cm_v03.PNG", caption="Figure 13. Text classification Confusion Matrix", use_container_width=True)

    
        st.subheader("Classification Statistics")
        st.markdown("""
        The highest number of correctly classified products was observed in the **"Pool and Accessories"** category (prdtypecode: 2583). 
        This category represents the majority class, with a total of **10,209 records** in the dataset.
        """)
        
        st.subheader("Misclassification Statistics")
        st.markdown("""
        Analyzing misclassified records provides valuable insights into the model's performance and limitations. Among the 
        misclassified records, the **Children's Toys** category (prdtypecode: 1280) has the highest number of errors, 
        with **162 instances**. Most of these are misclassified into the **Childcare and Baby Accessories** category 
        (prdtypecode: 1320) and **Figurines** (prdtypecode: 1140). This misclassification is understandable since both categories deal with children's products. 

        Interestingly, the **Children's Toys** category is not a minor class—it accounts for a significant **4,870 records** 
        in the dataset.
        """)

        st.markdown("""
        From the confusion matrix, another notable observation is that the highest number of products misclassified into 
        a single category belongs to the **General Furniture** category (prdtypecode: 1560). Specifically, **39 products** 
        were incorrectly classified as part of the **Decoration** category (prdtypecode: 2060).
        """)

        
    with tab2:
        st.subheader("VGG16 Model for Image Classification")    
                    
        # Description of VGG16
        st.markdown("""
        - **Architecture**: VGG16 is a deep convolutional neural network (CNN) with 16 layers, consisting of convolutional layers followed by fully connected layers.
        - **Use Case**: Commonly used for image classification tasks, especially with large datasets like ImageNet.
        - **Advantages**:
            - Simple, **elegant architecture** that performs well in many image classification tasks.
            - **Transfer learning**: Pretrained models are widely available and can be fine-tuned for specific tasks.
            - Known for its ability to **capture spatial hierarchies** in images due to deep layers and small receptive fields.
        """)

        # Images related to VGG16
        st.markdown("<h3 style='font-weight: bold;'>Confusion Matrix</h3>", unsafe_allow_html=True)
        st.image("/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/image_cm_3.PNG", caption="Figure 14. Image classification Confusion Matrix", use_container_width=True)

        st.subheader("Classification Statistics")
        st.markdown("""
        The highest number of correctly classified products was observed in the **"Pool and Accessories"** category (prdtypecode: 2583). 
        This category represents the majority class, with a total of **10,209 records** in the dataset.
        """)

        
        st.subheader("Misclassification Statistics")
        st.markdown("""
        ### Mismatch Analysis and Image Comparison

        **Category Children's Toys (prdtypecode 1280)**:
        - This category has the highest number of mismatches, with **606 instances**.
        - Most misclassified instances are incorrectly labeled as **prdtypecode 1300 (Modeling)**.
        - This indicates that the model struggles to differentiate between children's toys and modeling-related products.
        - Possible reasons include similarities in **shape, color, or contextual features** between these two categories.

        **Category Game Consoles (prdtypecode 60)**:
        - Despite being a minority class with only **832 records**, this category has the **lowest number of mismatches**—just **58 instances**.
        - The model demonstrates high accuracy in recognizing game consoles.
        - Likely contributing factors include the **distinctive and easily recognizable features** of game consoles.
        """)

    
    
elif menu == "Multimodal":
    st.title("Multimodal Classification")
    
    tab1, tab2 = st.tabs(['Multimodal', 'Demo'])
    with tab1:
        st.subheader("Late fusion with Voting classifier") 
        st.markdown("""Multimodal fusion was implemented using late fusion, employing soft voting with class weights to combine predictions from different modalities 
                    and make the final decision. \n The structure and outline of the fusion model is given below:""")
        
        st.image("/Users/samyuktha/Downloads/DSProject/Rakuten streamlit final/Late_Fusion_flowchart.png", caption="Figure 15. Multimodal classification flow chart", use_container_width=True)
        
    
    with tab2:
    

        st.subheader("Enter Text and/or Upload an Image for Classification")

        # Text input
        text_input = st.text_area("Enter text for classification:", "")

        # Image input
        img_file = st.file_uploader("Upload an image for classification:", type=["jpg", "jpeg", "png"])

        # Category Mapping (Moved to Streamlit)
        category_mapping = {
            10: "Used books",
            2280: "Used Newspapers and magazines",
            2403: "Books, comics, and magazines",
            2522: "Office supplies and stationery accessories",
            2705: "New books",
            40: "Video games, CDs, equipment, cables, new",
            50: "Gaming accessories",
            60: "Game consoles",
            2462: "Used video games",
            2905: "PC video games",
            1140: "Figurines, pop culture items",
            1180: "Trading cards",
            1160: "Board games and role-playing games",
            1280: "Children's toys",
            1260: "Children's board games",
            1281: "Model kits",
            1300: "Modeling",
            1325: "Outdoor games, air games, clothing",
            1560: "General furniture: beds, mattresses, sofas, chairs",
            2582: "Garden furniture: furniture and tools for the garden",
            1320: "Childcare, baby accessories",
            2567: "Pet supplies",
            2583: "Pool and accessories",
            2588: "Garden tools, outdoor technical equipment, house and pool accessories",
            1325: "Home linens, pillows, cushions",
            2560: "Decoration",
            1301: "Baby socks, small photos",
            1940: "Confectionery",
            2060: "Decoration",
            1920: "Household linen, pillows, cushions",
            2585: "Garden tools, Outdoor technical equipment for homes and swimming pools",
            1302: "Outdoor games, clothes",
            2220: "Pet store"
        }

        # Predict button
        predict_button = st.button("Predict")

        if predict_button:
            if text_input or img_file:
                img_path = None
                if img_file:
                    # Save uploaded image to a temporary file
                    img_path = f"./temp_image.{img_file.name.split('.')[-1]}"
                    with open(img_path, "wb") as f:
                        f.write(img_file.getbuffer())

                # Perform classification
                final_label, combined_probs = late_fusion_weighted_soft_voting(
                    text_input, img_path, text_label_encoder, image_label_decoder, text_weight=0.61, image_weight=0.39
                )

                # DEBUG: Check what the final_label is
                #st.write(f"Final predicted label (class index): {final_label}")

                # Ensure final_label is an integer
                final_label = int(final_label)  # Cast to int if it's not already an integer

                # Map the predicted label to the category name using category_mapping
                category_name = category_mapping.get(final_label, "Unknown Category")

                # Display results
                if final_label:
                    st.subheader("Prediction Result:")
                    st.write(f"Predicted Category ID: **{final_label}**")
                    st.write(f"Category Name: **{category_name}**")
                else:
                    st.error("Could not make a prediction. Please check your inputs.")
            else:
                st.error("Please provide either text or an image input for classification.")
    
    
# Sidebar footer with LinkedIn profiles
st.sidebar.info("**Streamlit Application - Sep 2024 BootCamp Datascience (INT)**")
#st.sidebar.info("**Participants:**")
st.sidebar.markdown("""
- **Samyuktha Soundararaj**  
  <a href="https://www.linkedin.com/in/samyukthasoundararaj" target="_blank">
      <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="16" style="vertical-align:middle;margin-right:5px;">LinkedIn
  </a>  
""", unsafe_allow_html=True)

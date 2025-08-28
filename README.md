# SousChef-Model (ByteBites) 🍴

A Recipe Generation and Ingredient Recognition System Powered by AI

## Collaborators 🤝

This project was created by the [Caffinated Coders](https://github.coecis.cornell.edu/cs4701-24fa-projects/PC_Caffeinated-Coders_apk67_gff29_rs929_rsh256) team as part of Cornell CS 4701:

[Aditya Kakade](https://github.com/adityakakade432), [Gabby Fite](https://github.com/gabbif), Ramisha Hossain, Richie Sun

## Overview

Byte Bites is an AI-powered application that generates custom recipes based on user-provided ingredients and images. By leveraging advanced transformer models and machine learning techniques, Byte Bites aims to revolutionize the cooking experience, enabling users to explore creative and coherent recipes with minimal effort.

## Features

- **Recipe Generation:** Produces detailed recipes with appropriate actions and ingredients using fine-tuned T5 and GPT-2 transformer models.
- **Semantic Validation:** Ensures contextual accuracy in recipe instructions by incorporating BERT-based semantic validation techniques.
- **Evaluation and Metrics:** Recipes are evaluated for coherence, complexity, and length using BLEU and ROUGE scores, alongside subjective human assessments.

## Technologies Used

**Language:** Python
**Frameworks and Libraries:**
TensorFlow, PyTorch, scikit-learn
Transformers (T5, GPT-2, BERT)
NumPy, Pandas, Matplotlib

## Results

Successfully generated recipes of sufficient length and complexity (~100 words on average), aligning with common short recipe benchmarks.
Semantic validation ensured the elimination of nonsensical outputs (e.g., "fry water").
Ingredient labeling achieved high accuracy for professional-quality images; performance on user-uploaded images remains an area for improvement.
Future Work

Enhance image recognition accuracy for user-uploaded photos by improving dataset diversity and preprocessing techniques.
Explore additional transformer architectures to further optimize recipe quality.

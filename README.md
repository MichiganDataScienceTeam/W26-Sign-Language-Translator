# Sign Language Translation

This project explores the use of Convolutional Recurrent Neural Networks to translate a curated subset of American Sign Language (ASL) gestures directly into text through a live webcam feed.

Using the WLASL video dataset, alongside MediaPipe for precise hand-landmark and pose detection, members will learn how to transform videos of gestures into meaningful features suitable for machine learning. Early milestones include classifying single frames of hand gestures, before transitioning into recognition of gestures that consist of spatial changes over multiple frames using Long Short-Term Memory (LSTM) layers. The project culminates in a Streamlit application that visualizes hand movements and skeletal structures on screen in real-time, and displays translations using the trained gesture recognition model.

**Final deliverable:** A streamlit dashboard with an interface that provides real-time sign language translation, with a visualization of the skeletal structure and display of both the ASL glosses and the english translation of multiple gestures in sequence

## Timeline

Subject to changes.
| Date | Activity |
|-----------|------------------------------|
| Jan 25 | 1: Introduction + MediaPipe 🤟 |
| Feb 1 | 2: neural networks 🕸️ |
| Feb 8 | ️3: recurrent neural networks, long short-term memory 🕸️ |
| Feb 15 | 4: RNN review, model training 🚂 |
| Feb 22 | 5: mid-semester expo, training cont., datasets 💽 |
| - | Spring Break 🛶 |
| - | Spring Break 🛶 |
| Mar 15 | 6: streamlit interface, project groups 🌐 |
| Mar 22 | 7: sequential gestures, custom datasets 👋 |
| Mar 29 | 8: project group work - sentence translation, APIs 🐧 |
| Apr 5 | 9: project group work 🐧 |
| Apr 12 | 10: prepare for final presentations 🎉 |
| Apr 17 | Data Science Night 🔮 |


# Week 1 - Intro
- [Week 1 Slides](https://docs.google.com/presentation/d/1mJ6QjMuurNHb7etnTNe96_rq9khBtgGW8bxEydZ8n68/edit?usp=sharing)
- [Week 1 Notebook - Intro to MediaPipe](https://colab.research.google.com/drive/1bkYvOFaU2PBsfJxAZBg9k0pSEUVr8SAu?usp=drive_link)
- Download image: `!wget -q -O image.jpg https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg`

# Week 2 - Neural Networks
- [Week 2 Slides](https://docs.google.com/presentation/d/1SBArJkI8JzcSzYju07fL3S4_uXxj5jDPHdy_tK8KAic/edit?slide=id.g3bc92866f7b_2_300#slide=id.g3bc92866f7b_2_300)

# Week 3 - Recurrent Neural Networks
- [Week 3 Slides](https://docs.google.com/presentation/d/1AVtqcNsD8Uw9kVlnnCw_Dvl9rETAWDoTYzptZpswKXc/edit?usp=drive_link)

# Week 4 - Model Training
- [Week 4 Reading](https://www.handspeak.com/learn/98/)
- [Week 4 Slides](https://docs.google.com/presentation/d/1z2I8vZRb1QiWqPLZIYbeTonWW0ZZw4xAj_oPC-LuPV8/edit?usp=sharing)
- [Week 4 Notebook](https://colab.research.google.com/github/MichiganDataScienceTeam/W26-Sign-Language-Translator/blob/main/Week%204/week4_wlasl.ipynb)
- Dataset and starter code: [wlasl_week4.zip](https://drive.google.com/file/d/15l1LFlF9gTqms8ufQYqA1ry_qoWlOXVL/view?usp=sharing)
- [Form for sharing results](https://docs.google.com/forms/d/e/1FAIpQLSell28hoohp7f9UpNQgXAbUgE-dG5ll6G1uaZ5dlusEZlQRJg/viewform?usp=sharing&ouid=113884584305464810417)

# Week 5 - Datasets
- [Week 5 Slides](https://docs.google.com/presentation/d/1iVnfVsaVBV4j-yOcUrzDBM0M6zAqR2_JYonSDxFm45s/edit?usp=sharing)
- [Week 5 Notebook](https://colab.research.google.com/drive/19S9zdzCImVxbAyJaSkoUhV7Gk8ymqTQV?usp=sharing)
- [Week 5 Python Files (also in repo)](https://drive.google.com/drive/folders/1OgBGL46I9Nf79j0pBYqnN71Mfgetc3EJ?usp=sharing)
- [ASL Citizen Processed](https://www.kaggle.com/datasets/dennisfj/asl-citizen-processed)

# Week 6 - Streamlit
- [Week 6 Slides](https://docs.google.com/presentation/d/1BwaehXjX-Sjx6QM_3Uv5FGRv8K9EQYm26rmQNJ3vlDE/edit?usp=sharing)
- [Week 6 Notebook (training code from before break)](https://colab.research.google.com/drive/1FAIX7Gid5Hpr_hFu5KdkMQCNVjbCVTIL#scrollTo=6s5JMqZKDj9n)

# Week 7 - Continuous Gestures
- [Week 7 Slides](https://docs.google.com/presentation/d/1pyzL0gUe7jOKgQ4u3czzLBziFHfc3q-9YRlW1Fn5o50/edit?usp=sharing)
    

# Google Colab Setup Guide

Start by uploading your notebook to Google [Colab](https://colab.research.google.com/). You can also access the notebook through Colab via the following link:

<a href="https://colab.research.google.com/github/MichiganDataScienceTeam/W26-Sign-Language-Translator/blob/main/Week%204/week4_wlasl.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

On the top left of your screen, click "File > Save as copy to Drive" or click "Copy to Drive" so that you can save your changes.


To enable the GPU on Google Colab, select the following settings from the menu bar at the top:
```python
# colab: enable GPU:
# Runtime > Change Runtime Type > T4 GPU > Save

# note: may need to restart your Notebook session (Runtime > Restart Session)
```

For running on Google Colab, note that you won't be able to use a web camera for real-time classification, however you can use Colab to train your model using the provided GPU option, then download the saved model weights to your local machine and run inference using your computer's CPU. See the [PyTorch documentation](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) or feel free to ask us if you need assistance with saving and loading model weights.

```python
model = LSTM_Model()
model.to(device)
model.train()
# ... train your model ...
torch.save(model.state_dict(), "model_weights.pth")

# <download from Colab to your local setup>

# load model weights using same class definition used for training (LSTM_Model)
model = LSTM_Model()
model.load_state_dict(torch.load("model_weights.pth"))
model.to(device)
model.eval()
```

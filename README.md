# VR Eye Gaze Tracking
Eye tracking and gaze tracking in virtual space.

With the help of an ESP32 microcontroller, the resulting images are accessed via a web server by the python image processing algorithm. The algorithm uses a tflite model to process the data and detect the pupil position. From the pupil positions, the gaze coordinate is calculated and displayed in a godot virtual environment.

# Eye tracking technology pros:
- Visual attention analysis
- Non-invasive
- Uses:
  - Psychological studies
  - Marketing purposes
  - Human-computer interaction

# Solution
- ESP32-CAM
- Camera image on a web server
- Python image processing
- Godot 
- VR gaze point display

| ![Picture1](https://github.com/BenceBiricz/VR_Eye_Gaze_Tracking/assets/71565433/fa25f687-29a1-4fa3-9fdf-46c24ff0fea3) | ![Picture2](https://github.com/BenceBiricz/VR_Eye_Gaze_Tracking/assets/71565433/9faace27-08e0-46cb-be3c-948aafd05c1c) |
| --- | --- |

- Pupil detection
- Determination of maximum values
- Convert pupil position to gaze

| ![Picture3](https://github.com/BenceBiricz/VR_Eye_Gaze_Tracking/assets/71565433/4f30ccfd-530f-4639-a05c-4e80f754d4f4) | ![Picture4](https://github.com/BenceBiricz/VR_Eye_Gaze_Tracking/assets/71565433/7e19a866-efda-4904-906b-454e074e97d8) |
| --- | --- |

- pX - pupil X coordinate
- pWmax - pupil max width
- gWmax - gaze max width
- pY - pupil Y coordinate
- pHmax - pupil max height
- gHmax - gaze max height

# Godot visualization

| ![Picture5](https://github.com/BenceBiricz/VR_Eye_Gaze_Tracking/assets/71565433/765e8ef9-0f99-4791-a54f-e71822048c81) | ![Picture6](https://github.com/BenceBiricz/VR_Eye_Gaze_Tracking/assets/71565433/620510fb-4693-4932-8c46-68abb702684d) |
| --- | --- |

# Camera positioned in the VR headset

| ![Picture7](https://github.com/BenceBiricz/VR_Eye_Gaze_Tracking/assets/71565433/81536f39-4157-4cab-8120-10d32bbb30fb) | ![Picture8](https://github.com/BenceBiricz/VR_Eye_Gaze_Tracking/assets/71565433/258b56e4-5723-4e47-9ac3-7811ee5a42c7) |
| --- | --- |

# Visualizing the Gaze points

![ezgif com-video-to-gif](https://github.com/BenceBiricz/VR_Eye_Gaze_Tracking/assets/71565433/7f4e4645-c1c2-4b1a-bcba-138e9d13d787)





  

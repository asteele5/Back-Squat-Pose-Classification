import cv2
import csv
import mediapipe as mp
import numpy as np
import json
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

numVideo = input('Please enter the number of the video: ')

while(True):

  videoType = input('Please enter whether the video is a good or bad squat: ')

  if(videoType == 'g'):
    filepath = r'C:\Users\mrswa\Senior Research\Back Squat Pose Classification\Squat Videos\Front Good\squat_video'+numVideo+videoType+'.mp4'
    break
  elif(videoType == 'b'):
    filepath = r'C:\Users\mrswa\Senior Research\Back Squat Pose Classification\Squat Videos\Front Bad\squat_video'+numVideo+videoType+'.mp4'
    break
  else:
    print('Please enter a b or g')


def standardize(list):
  mean = np.mean(list)
  std = np.std(list)
  list = (list-mean)/std
  return list

j = 0

with open(r'Front Training Data\squat_video'+numVideo+'.json', 'w') as json_out_file:
  landmarks_list = []
  cap = cv2.VideoCapture(filepath)
  with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        break


      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = pose.process(image)

      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

      landmarks = str(results.pose_landmarks)
      
      pose_landmarks = results.pose_landmarks
      if pose_landmarks is not None:
        # Check the number of landmarks and take pose landmarks.
        assert len(pose_landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(len(pose_landmarks.landmark))
        pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]

        # Map pose landmarks from [0, 1] range to absolute coordinates to get
        # correct aspect ratio.
        frame_height, frame_width = image.shape[:2]
        pose_landmarks *= np.array([frame_width, frame_height, frame_width])

        pose_landmarks = np.around(pose_landmarks, 5).transpose().tolist()
        for i, coordinate in enumerate(pose_landmarks):
          pose_landmarks[i] = standardize(coordinate)
          pose_landmarks[i] = pose_landmarks[i].tolist()
        
        landmarks_list.append(pose_landmarks)
        j += 1

    print(np.shape(landmarks_list))
    pose_dictionary = {'x': [], 'y': [], 'z': []}
    for frame in landmarks_list:
      i = 0
      for coordinates in frame:
        if(i == 0):
          pose_dictionary['x'].append(coordinates)
        elif(i == 1):
          pose_dictionary['y'].append(coordinates)
        elif(i == 2):
          pose_dictionary['z'].append(coordinates)
        i += 1
    if(videoType=='g'):
      pose_dictionary['value'] = 1 
    else:
      pose_dictionary['value'] = 0

    json.dump(pose_dictionary, json_out_file)
      
  cap.release()

import cv2
import mediapipe as mp

from scipy.spatial import distance as dist

import keyboard

from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

morse = {
  ".-": 'a',
  "-...": 'b',
  "-.-.": 'c',
  "-..": 'd',
  ".": 'e',
  "..-.": 'f',
  "--.": 'g',
  "....": 'h',
  "..": 'i',
  ".---": 'j',
  "-.-": 'k',
  ".-..": 'l',
  "--": 'm',
  "-.": 'n',
  "---": 'o',
  ".--.": 'p',
  "--.-": 'q',
  ".-.": 'r',
  "...": 's',
  "-": 't',
  "..-": 'u',
  "...-": 'v',
  ".--": 'w',
  "-..-": 'x',
  "-.--": 'y',
  "--..": 'z',
  ".----": '1',
  "..---": '2',
  "...--": '3',
  "....-": '4',
  ".....": '5',
  "-....": '6',
  "--...": '7',
  "---..": '8',
  "----.": '9',
  "-----": '0',
  ".-.-.-": '.',
  "--..--": ',',
  "---...": ';',
  ".----.": "'",
  ".----.-": '`',
  "-....-": '-',
  "-...-": '=',
  "-..-.": '/',
  "-..-.-": '\\',
  "----.-": '[',
  "------": ']',
  ".-.-": 'tab',
  "space": "space",
  "enter": "enter"
}


CAMERA = 1 # Usually 0, depends on input device(s)

# Optionally record the video feed to a timestamped AVI in the current directory
RECORDING = False
FPS = 10
RECORDING_FILENAME = str(datetime.now()).replace('.','').replace(':','') + '.avi'

FACE_TILT = .5

EYE_BLINK_HEIGHT = .15
EYE_SQUINT_HEIGHT = .18
EYE_OPEN_HEIGHT = .25
EYE_BUGGED_HEIGHT = .7

MOUTH_OPEN_HEIGHT = .2
MOUTH_OPEN_SHORT_FRAMES = 1
MOUTH_OPEN_LONG_FRAMES = 4
MOUTH_CLOSED_FRAMES = 1

MOUTH_FROWN = .006
MOUTH_NOSE_SCRUNCH = .09
MOUTH_SNARL = .1
MOUTH_DUCKFACE = 1.6

BROW_RAISE_LEFT = .0028
BROW_RAISE_RIGHT = .025
BROWS_RAISE = .19

WAIT_FRAMES = 6


blinking = False
blink_count = 0
blinking_frames = 0

squinting = False
squinting_frames = 0

bugeyed = False
bugeyed_frames = 0

winkedR = False
winkedR_frames = 0

winkedL = False
winkedL_frames = 0

mouth_open = False
mouth_open_frames = 0
mouth_closed_frames = 0

mouth_scrunched = False
mouth_scrunched_count = 0
mouth_scrunched_frames = 0

duckfacing = False

brows_raised = False
brows_raised_count = 0
brows_raised_frames = 0

command_on = False
control_on = False
shift_on = False

current_morse = ''
last_typed = ''


def type_and_remember():
  global current_morse, last_typed
  keys = []

  if command_on:
    keys.append('command')
  if control_on:
    keys.append('control')
  if shift_on:
    keys.append('shift')

  letter = morse.get(current_morse, '')
  if len(letter):
    keys.append(letter)
  current_morse = ''

  keystring = '+'.join(keys)
  if len(keystring):
    print("keys:", keystring)
    keyboard.press_and_release(keystring)
    last_typed = keystring


def get_aspect_ratio(top, bottom, right, left):
  height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
  width = dist.euclidean([right.x, right.y], [left.x, left.y])
  return height / width


def timeout_double(state, frames):
  if state:
    frames += 1
  if frames > WAIT_FRAMES:
    frames = 0
    state = False
  return state, frames


def draw_frame(image, face_landmarks):
  mp_drawing.draw_landmarks(
      image=image,
      landmark_list=face_landmarks,
      connections=mp_face_mesh.FACEMESH_TESSELATION,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp_drawing_styles
      .get_default_face_mesh_tesselation_style())
  mp_drawing.draw_landmarks(
      image=image,
      landmark_list=face_landmarks,
      connections=mp_face_mesh.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp_drawing_styles
      .get_default_face_mesh_contours_style())
  mp_drawing.draw_landmarks(
      image=image,
      landmark_list=face_landmarks,
      connections=mp_face_mesh.FACEMESH_IRISES,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp_drawing_styles
      .get_default_face_mesh_iris_connections_style())
  frame = cv2.flip(image, 1) # Flip image horizontally
  # Add current Morse code as supertitle
  cv2.putText(frame, current_morse, (620, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  cv2.imshow('face', frame)


cap = cv2.VideoCapture(CAMERA)

if RECORDING:
  frame_size = (int(cap.get(3)), int(cap.get(4)))
  recording = cv2.VideoWriter(
    RECORDING_FILENAME, cv2.VideoWriter_fourcc(*'MJPG'), FPS, frame_size)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
      face_landmarks = results.multi_face_landmarks[0]
      face = face_landmarks.landmark

      face_mid_right = face[234]
      face_mid_left = face[454]
      face_mid_top = face[10]
      face_mid_bottom = face[152]
      cheek_mid_right = face[50]
      cheek_mid_left = face[280]

      if cheek_mid_right.x < face_mid_right.x:
        print("head turn R")
        continue
      elif cheek_mid_left.x > face_mid_left.x:
        print("head turn L")
        continue

      face_angle = (face_mid_top.x - face_mid_bottom.x) / (
        face_mid_right.x - face_mid_left.x)
      if face_angle > FACE_TILT:
        print("head tilt R", face_angle)
      elif face_angle < -FACE_TILT:
        print("head tilt L", face_angle)

      eyeR_top = face[159]
      eyeR_bottom = face[145]
      eyeR_inner = face[133]
      eyeR_outer = face[33]
      eyeR_ar = get_aspect_ratio(eyeR_top, eyeR_bottom, eyeR_outer, eyeR_inner)

      eyeL_top = face[386]
      eyeL_bottom = face[374]
      eyeL_inner = face[362]
      eyeL_outer = face[263]
      eyeL_ar = get_aspect_ratio(eyeL_top, eyeL_bottom, eyeL_outer, eyeL_inner)
      eyeA_ar = (eyeR_ar + eyeL_ar) / 2

      command_on = False
      shift_on = False
      squinting = False
      bugeyed = False
      if eyeR_ar < EYE_BLINK_HEIGHT:
        if eyeL_ar > EYE_OPEN_HEIGHT:
          print("R wink", eyeR_ar)
          shift_on = True
          winkedR = True
          if winkedL and (winkedL_frames < WAIT_FRAMES):
            print("ESCAPE")
            keyboard.press_and_release('escape')
            winkedL_frames = 0
            winkedL = False
        elif eyeR_ar < EYE_BLINK_HEIGHT:
          if not blinking:
            blink_count += 1
            print("blink", blink_count)
            if duckfacing and blink_count == 2:
              print("BACKSPACE")
              keyboard.press_and_release("backspace")
          blinking = True
      elif eyeL_ar < EYE_BLINK_HEIGHT and eyeR_ar > EYE_OPEN_HEIGHT:
        print("L wink", eyeL_ar)
        command_on = True
        winkedL = True
        if winkedR and (winkedR_frames < WAIT_FRAMES):
          print("clear Morse queue")
          current_morse = ''
          winkedR_frames = 0
          winkedR = False
      elif eyeA_ar < EYE_SQUINT_HEIGHT:
        squinting = True
        squinting_frames += 1
        if squinting_frames > WAIT_FRAMES:
          print("squint", eyeA_ar)
          keyboard.press_and_release("command+-") # zoom out
          squinting_frames = 0
      elif eyeA_ar > EYE_BUGGED_HEIGHT:
        bugeyed = True
        bugeyed_frames += 1
        if bugeyed_frames > WAIT_FRAMES:
          bugeyed_frames = 0
          print("big eyes", eyeA_ar)
          keyboard.press_and_release("command+shift+=") # zoom in
      else:
        blinking = False

      winkedL, winkedL_frames = timeout_double(winkedL, winkedL_frames)
      winkedR, winkedR_frames = timeout_double(winkedR, winkedR_frames)
      blink_count, blinking_frames = timeout_double(blink_count, blinking_frames)

      mouth_outer_top = face[0]
      mouth_outer_bottom = face[17]
      mouth_outer_right = face[61]
      mouth_outer_left = face[291]

      mouth_inner_top = face[13]
      mouth_inner_bottom = face[14]
      mouth_inner_right = face[78]
      mouth_inner_left = face[308]
      mouth_inner_ar = get_aspect_ratio(
        mouth_inner_top, mouth_inner_bottom, mouth_inner_right, mouth_inner_left)

      nose_bottom = face[2]

      mouth_open = mouth_inner_ar > MOUTH_OPEN_HEIGHT
      if mouth_open:
        print("mouth open", mouth_inner_ar)
        mouth_open_frames += 1

      if (not mouth_open) and (mouth_open_frames >= MOUTH_OPEN_SHORT_FRAMES):
        if mouth_closed_frames >= MOUTH_CLOSED_FRAMES:
          if mouth_open_frames >= MOUTH_OPEN_LONG_FRAMES:
            current_morse += '-'
          elif mouth_closed_frames >= MOUTH_CLOSED_FRAMES:
            current_morse += '.'
          mouth_open_frames = 0
          mouth_closed_frames = 0
        else:
          mouth_closed_frames += 1

      mouth_frowny_right = (mouth_inner_right.y - mouth_inner_bottom.y) > MOUTH_FROWN
      mouth_frowny_left = (mouth_inner_left.y - mouth_inner_bottom.y) > MOUTH_FROWN
      mouth_frowny = mouth_frowny_right and mouth_frowny_left

      nose_to_mouth = (mouth_outer_top.y - nose_bottom.y) / (
        face_mid_bottom.y - face_mid_top.y)

      # TODO: adjust better for head pitch up/down
      # TODO: wait/ignore single frames during scrunch
      if (mouth_scrunched_count > 0) and (not mouth_scrunched):
        mouth_scrunched_frames += 1

      if (nose_to_mouth < MOUTH_NOSE_SCRUNCH) and mouth_frowny:
        type_and_remember()
        current_morse = ''
        if not mouth_scrunched:
          mouth_scrunched_count += 1
          print("mouth scrunch", nose_to_mouth)
        mouth_scrunched = True
      else:
        mouth_scrunched = False

      if mouth_scrunched_count >= 3:
        print("triple scrunch: ENTER")
        current_morse = 'enter'
        type_and_remember()
        mouth_scrunched_frames = 0
        mouth_scrunched_count = 0
      elif mouth_scrunched_count == 2:
        if mouth_scrunched_frames > WAIT_FRAMES:
          print("double scrunch: SPACE")
          current_morse = 'space'
          type_and_remember()
          mouth_scrunched_frames = 0
          mouth_scrunched_count = 0
      elif (mouth_scrunched_count == 1) and (mouth_scrunched_frames > WAIT_FRAMES):
        mouth_scrunched_frames = 0
        mouth_scrunched_count = 0

      mouth_outer_right_mid_top = face[39]
      mouth_outer_right_mid_bottom = face[181]
      mouth_outer_left_mid_top = face[269]
      mouth_outer_left_mid_bottom = face[405]

      # check for snarl; unused
      mouth_snarl_right = (mouth_outer_left_mid_top.y - mouth_outer_right_mid_top.y) / (
        face_mid_right.y - face_mid_left.y)
      if mouth_snarl_right > MOUTH_SNARL:
        print("snarl R", mouth_snarl_right)

      # check for duckface
      duckfacing = False
      if not mouth_open:
        mouth_width = (mouth_outer_right.x - mouth_outer_left.x) / (
          face_mid_right.x - face_mid_left.x)
        mouth_height_right = mouth_outer_right_mid_top.y - mouth_outer_right_mid_bottom.y
        mouth_height_left = mouth_outer_left_mid_top.y - mouth_outer_left_mid_bottom.y
        mouth_height = (mouth_height_right + mouth_height_left) / (
          face_mid_top.y - face_mid_bottom.y)
        mouth_outer_ar = mouth_width / mouth_height
        if mouth_outer_ar < MOUTH_DUCKFACE:
          print("duckface", mouth_outer_ar)
          duckfacing = True

      browR_top = face[52]
      browR_bottom = face[223]
      browR_eyeR_lower_dist = dist.euclidean([browR_bottom.x, browR_bottom.y],
        [eyeR_top.x, eyeR_top.y])
      browR_eyeR_upper_dist = dist.euclidean([browR_top.x, browR_top.y],
        [eyeR_top.x, eyeR_top.y])
      browR_eyeR_dist = (browR_eyeR_lower_dist + browR_eyeR_upper_dist) / 2

      browL_top = face[443]
      browL_bottom = face[257]
      browL_eyeL_lower_dist = dist.euclidean([browL_bottom.x, browL_bottom.y],
        [eyeL_top.x, eyeL_top.y])
      browL_eyeL_upper_dist = dist.euclidean([browL_top.x, browL_top.y],
        [eyeL_top.x, eyeL_top.y])
      browL_eyeL_dist = (browL_eyeL_lower_dist + browL_eyeL_upper_dist) / 2

      brows_avg_raise = (browR_eyeR_dist + browL_eyeL_dist) / (
        face_mid_bottom.y - face_mid_top.y)
      brows_relative_raise = browR_eyeR_dist - browL_eyeL_dist

      if brows_relative_raise < BROW_RAISE_LEFT:
        brows_raised = False
        if duckfacing:
          print("L brow duckfacing: ARROW LEFT", brows_relative_raise)
          keyboard.press_and_release("left arrow")
        else:
          print("L brow raise: SCROLL UP", brows_relative_raise)
          keyboard.press_and_release("up arrow")
      elif brows_relative_raise > BROW_RAISE_RIGHT:
        brows_raised = False
        if duckfacing:
          print("R brow duckfacing: ARROW RIGHT", brows_relative_raise)
          keyboard.press_and_release("right arrow")
        else:
          print("R brow raise: SCROLL DOWN", brows_relative_raise)
          keyboard.press_and_release("down arrow")
      elif brows_avg_raise > BROWS_RAISE and eyeA_ar > EYE_OPEN_HEIGHT:
        if not brows_raised:
          brows_raised_count += 1
        brows_raised = True
        print("brows raised", brows_avg_raise)
      else:
        brows_raised = False

      control_on = brows_raised

      if brows_raised_count >= 2:
        print("double brow raise - repeat:", last_typed)
        if len(last_typed):
          keyboard.press_and_release(last_typed)
        brows_raised_frames = 0
        brows_raised_count = 0
      elif brows_raised_count == 1:
        brows_raised_frames += 1
      if brows_raised_frames > WAIT_FRAMES:
        brows_raised_frames = 0
        brows_raised_count = 0

      draw_frame(image, face_landmarks)
      if RECORDING:
        recording.write(image)

    # Type 'q' on the video frame to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break

if RECORDING:
  recording.release()

cap.release()
cv2.destroyAllWindows()

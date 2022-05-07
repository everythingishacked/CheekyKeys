## CheekyKeys
### A Face-Computer Interface

CheekyKeys lets you control your keyboard using your face.

![demo](demo.gif)

View a fuller demo and more background on the project at https://youtu.be/rZ0DBi1avMM

CheekyKeys uses [OpenCV](https://github.com/opencv/opencv-python) and MediaPipe's [Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html#python-solution-api) to perform real-time detection of facial landmarks from video input. From there, relative differences are calculated to determine specific facial gestures and translate those into commands sent via [keyboard](https://github.com/boppreh/keyboard).

This version 0.1 is hardcoded to my facial features, but thresholds can easily be modified. It's also built for a Mac keyboard, but you can also swap i.e. Windows key for Command simply enough.

The primary input is to "type" letters, digits, and symbols via Morse code by opening and closing your mouth quickly for `.` and slightly longer for `-`. Rather than waiting a set time after every letter, you scrunch your mouth upward once to finish a letter, or twice to add a space (end a word). Three mouth scrunches types enter/return.

The cheatsheet includes the full alphabet as well as special characters and hotkeys.

Most of the rest of the keyboard and other helpful actions are included as modifier gestures, such as:

- `shift`: close right eye
- `command`: close left eye
- `arrow up/down`: raise left/right eyebrow
- `arrow left/right`: raise left/right eyebrow + duckface (pursed lips)
- `backspace`: duckface + double blink
- `zoom in`: eyes bulge
- `zoom out`: eyes squint
- repeat previous letter/command: double raise of both eyebrows
- clear current Morse queue: wink right eye, then wink left eye
- `escape`: wink left eye, then wink right eye

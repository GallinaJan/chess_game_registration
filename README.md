1. We created environment 'chess_register' in order to install all the needed libraries (Python version: 3.10.18).
2. There was the first dataset with real 3D chess pieces, but due to bad conditions (light, tile image size, etc.) it was changed. The second dataset consists of the pictures of 2D chess figures.
3. While doing chessboard recognition, you have to remove all the chess pieces from the board.
4. After the chessboard recognition (before chess pieces detection), you have to cast a shadow on the chessboard in order to properly identify figures.
5. CNN was teached with a GPU.

SOURCE:
- TheAILerner (https://theailearner.com/tag/cv2-getperspectivetransform/, https://theailearner.com/tag/cv2-warpperspective/) - perspective change - 2D homography (method and functions)
- CLAHE - https://www.youtube.com/watch?v=tn2kmbUVK50&t=204s&ab_channel=KevinWood%7CRobotics%26AI
- Kaggle - "Detecting Chess Pieces with a CNN" - https://www.kaggle.com/code/thomassvisser/detecting-chess-pieces-with-a-cnn (Apache 2.0 open source license)
- chess pieces ("Designed by brgfx / Freepik")
- ChatGPT
- OpenCV and Tensorflow documentation
- lectures
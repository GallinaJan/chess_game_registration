1. We created environment 'chess_register' in order to install all the needed libraries (Python version: 3.10.18).
2. There was the first dataset with real 3D chess pieces, but due to bad conditions (light, tile image size, etc.) it was changed. The second dataset consists of the pictures of 2D chess figures.
3. While doing chessboard recognition, you have to remove all the chess pieces from the board.
4. After the chessboard recognition (before chess pieces detection), you have to cast a shadow on the chessboard in order to properly identify figures.
5. CNN was teached with a GPU.
6. There were different possibilities to solve chess piece detection problem. The first one was to look at the chessboard from above and take pictures of real pieces, but due to problems stated in (2.) the solution was abandoned. There was also the way to change the viewpoint for the camera, but I think that it would cause other difficulties, such as one piece covering up the other figure. We remained with 'view from above' approach, but we changed to 2D chess pieces.
7. Light conditions are critical in this project.

SOURCE:
- TheAILerner (https://theailearner.com/tag/cv2-getperspectivetransform/, https://theailearner.com/tag/cv2-warpperspective/) - perspective change - 2D homography (method and functions)
- CLAHE - https://www.youtube.com/watch?v=tn2kmbUVK50&t=204s&ab_channel=KevinWood%7CRobotics%26AI
- Kaggle - "Detecting Chess Pieces with a CNN" - https://www.kaggle.com/code/thomassvisser/detecting-chess-pieces-with-a-cnn (Apache 2.0 open source license)
- chess pieces ("Designed by brgfx / Freepik")
- ChatGPT
- OpenCV and Tensorflow documentation
- lectures
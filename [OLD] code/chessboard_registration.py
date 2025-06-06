import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    _, frame = cap.read()
    #frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(frame_g, 100, 400, apertureSize=3)
    #lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    """
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        # Stores the value of cos(theta) in a
        a = np.cos(theta)
        # Stores the value of sin(theta) in b
        b = np.sin(theta)
        # x0 stores the value rcos(theta)
        x0 = a*r

        # y0 stores the value rsin(theta)
        y0 = b*r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000*(-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000*(a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000*(-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000*(a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 2)
    """
    cv2.imshow("okno", frame)
    key = cv2.waitKey(1)  # Czekaj na dowolny klawisz
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()  # Zamknij okno

chessboard_top = cv2.imread("chessboard_top.jpg")
chessboard_top = cv2.resize(chessboard_top, (600, 600))  
cv2.imshow("szachownica", chessboard_top)
cv2.waitKey(0)  # Czekaj na dowolny klawisz
cv2.destroyAllWindows()  # Zamknij okno

chessboard_top_g = cv2.cvtColor(chessboard_top, cv2.COLOR_BGR2GRAY)

cv2.imshow("szachownica", chessboard_top_g)
cv2.waitKey(0)  # Czekaj na dowolny klawisz
cv2.destroyAllWindows()  # Zamknij okno

edges = cv2.Canny(chessboard_top_g, 256, 257, apertureSize=3)

cv2.imshow("krawedzie", edges)
cv2.waitKey(0)  # Czekaj na dowolny klawisz
cv2.destroyAllWindows()  # Zamknij okno

lines = cv2.HoughLines(edges,1, np.pi/180, 170)

for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    # Stores the value of cos(theta) in a
    a = np.cos(theta)

    # Stores the value of sin(theta) in b
    b = np.sin(theta)

    # x0 stores the value rcos(theta)
    x0 = a*r

    # y0 stores the value rsin(theta)
    y0 = b*r

    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000*(-b))

    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000*(a))

    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000*(-b))

    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000*(a))

    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.
    cv2.line(chessboard_top, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("szachownica", chessboard_top)
cv2.waitKey(0)  # Czekaj na dowolny klawisz
cv2.destroyAllWindows()  # Zamknij okno
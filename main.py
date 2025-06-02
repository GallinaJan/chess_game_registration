import cv2
import numpy as np
import matplotlib.pyplot as plt

# coords for perspective change
coord = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        coord.append((x, y)) # the first = left down corner, the second - right down corner, the third - right up corner, the fourth - left up corner

if __name__ == "__main__":
    # camera options
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    cv2.namedWindow("Coords for perspective change")
    cv2.setMouseCallback("Coords for perspective change", click_event)

    # getting coord for the perspetive change
    while len(coord) < 4:
        _, frame = cap.read() # real-time
        #frame = cv2.imread("photos/chessboard.jpg") # simulation

        # viewing
        cv2.imshow("Coords for perspective change", frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    tile_side_long_computed_flag = False

    columns = 'abcdefgh'
    rows = '12345678'

    tiles_names = [f"{col}{row}" for row in rows for col in columns]

    intersection_coord = np.zeros(shape=(9, 9), dtype=tuple)

    while True:
        # frame
        _, frame = cap.read() # real-time
        #frame = cv2.imread("photos/chessboard.jpg") # simulation

        width_top_view_1 = np.sqrt(((coord[1][0] - coord[0][0])**2) + ((coord[1][1] - coord[0][1])**2))
        width_top_view_2 = np.sqrt(((coord[2][0] - coord[3][0])**2) + ((coord[2][1] - coord[3][1])**2))
        width_top_view = max(int(width_top_view_1), int(width_top_view_2))
        
        height_top_view_1 = np.sqrt(((coord[3][0] - coord[0][0])**2) + ((coord[3][1] - coord[0][1])**2))
        height_top_view_2 = np.sqrt(((coord[2][0] - coord[1][0])**2) + ((coord[2][1] - coord[1][1])**2))
        height_top_view = max(int(height_top_view_1), int(height_top_view_2))
        
        coord_from_image = np.float32(coord)
        coord_top_view = np.float32([[0, 0], [width_top_view-1, 0], [width_top_view-1, height_top_view-1], [0, height_top_view-1]])

        M = cv2.getPerspectiveTransform(coord_from_image, coord_top_view)

        top_view = cv2.warpPerspective(frame, M,(width_top_view, height_top_view), flags=cv2.INTER_LINEAR)

        height, width, _ = np.shape(top_view)

        # light equalization - CLAHE algorithm
        lab = cv2.cvtColor(top_view, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        clahe_l = clahe.apply(l)
        clahe_lab = cv2.merge((clahe_l, a, b))
        clahe_top_view = cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2BGR)
        
        # smoothing
        gaussian = cv2.GaussianBlur(clahe_top_view, (7, 7), 3)

        # finding the chessboard
        hsv = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
        r_table = []
        theta_table = []
        lines_view = np.zeros(np.shape(top_view), dtype=np.uint8)
        if lines is not None:
            for r_and_theta in lines:
                r, theta = r_and_theta[0]
                r_table.append(r)
                theta_table.append(theta)
                if np.sin(theta) == 0:
                    x1, x2 = int(r), int(r)
                    y1, y2 = 0, height-1
                else:
                    x1, x2 = 0, width-1
                    a = -np.cos(theta)/np.sin(theta)
                    b = r/np.sin(theta)
                    y1 = int(a*x1+b)
                    y2 = int(a*x2+b)
                # limiting lines
                y1 = max(0, min(height-1, y1))
                y2 = max(0, min(height-1, y2))
                if np.isclose(theta, np.pi/2, atol=0.01) or np.isclose(theta, 0, atol=0.01) or np.isclose(theta, np.pi, atol=0.01):
                    cv2.line(lines_view, (x1, y1), (x2, y2), (0, 0, 255), 1)
        lines_intersection = np.zeros((3, 3, 3), dtype=np.uint8)
        lines_intersection[:, :, 2] = np.array([[0, 255, 0], [255, 255, 255], [0, 255, 0]], dtype=np.uint8)
        intersection_view = np.zeros(np.shape(top_view), dtype=np.uint8)
        if not tile_side_long_computed_flag:
            row = 0
            col = 0
            for i in range(1, height-1):
                for j in range(1, width-1):
                    window = lines_view[max(i-1, 0):min(i+2, height), max(j-1, 0):min(j+2, width), :]
                    if window.shape == (3, 3, 3) and np.all(window == lines_intersection):
                        if np.any(intersection_view[max(i-10, 0):min(i+11, height), max(j-10, 0):min(j+11, width), 2] == 255):
                            continue
                        else:
                            intersection_view[i, j, 2] = 255
                            intersection_coord[row, col] = (i, j)
                            col += 1
                            if col > 8:
                                row += 1
                                col = 0
                            if row > 8:
                                break
                if row > 8:
                    break
            if np.sum(intersection_view[:, :, 2]) != 9*9*255:
                continue
            #tile_diameter = np.sqrt(((intersection_coord[8, 8])[0]-(intersection_coord[0, 0])[0])**2 + ((intersection_coord[8, 8])[1]-(intersection_coord[0, 0])[1])**2)/8
            #tile_side_long = int(tile_diameter/np.sqrt(2))
            tile_side_long_computed_flag = True     

        tiles_view = []

        for i in range(8):
            for j in range(8):
                y_first, x_first = intersection_coord[i, j]
                _, x_last= intersection_coord[i, j+1]
                y_last, _ = intersection_coord[i+1, j]
                tile = np.array(top_view[y_first:y_last, x_first:x_last, :], dtype=np.uint8)
                #tile = np.array(top_view[coord[0]:coord[0]+tile_side_long, coord[1]:coord[1]+tile_side_long, :], dtype=np.uint8)
                tile = cv2.resize(tile, None, fx=10.0, fy=10.0, interpolation=cv2.INTER_LINEAR)
                tiles_view.append(tile)

        tiles = {names:views for (names, views) in zip(tiles_names, tiles_view)} 

        # viewing
        cv2.imshow("View", tiles["h3"])
        key = cv2.waitKey(1)  # wait for the key ('Esc')
        if key == 27:
            break

    plt.scatter(theta_table, r_table)
    plt.show()

    # cleaning
    cap.release()
    cv2.destroyAllWindows()
import cv2
import numpy as np


# # Wczytaj obraz
# img = cv2.imread("chessboard_top.jpg")
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # Zaznacz fragment do analizy
# roi = cv2.selectROI("Select Region", img, False)
# cv2.destroyAllWindows()

# # Pobierz wartości HSV z wybranego obszaru
# roi_hsv = hsv[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
# mean_hsv = np.mean(roi_hsv, axis=(0,1))

# print("Średnia wartość HSV:", mean_hsv)


#manual finder for mask parametrer
def find_mask_parameters(img):
    print("test")
    def nothing(x):
        pass

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color-segmentation to get binary mask

    # Utwórz okno i suwaki
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L-H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

    while True:
        # Pobierz wartości z suwaków
        l_h = cv2.getTrackbarPos("L-H", "Trackbars")
        l_s = cv2.getTrackbarPos("L-S", "Trackbars")
        l_v = cv2.getTrackbarPos("L-V", "Trackbars")
        u_h = cv2.getTrackbarPos("U-H", "Trackbars")
        u_s = cv2.getTrackbarPos("U-S", "Trackbars")
        u_v = cv2.getTrackbarPos("U-V", "Trackbars")


        #chessboard_top_result
        # lwr = np.array([0, 0, 0]) #best result for 0 0 0
        # upr = np.array([179, 255, 76]) #best result for 179 255 76
        #chessboard_empty_top resized to (600,600)
        # lwr = np.array([0, 43, 0]) #best result for 0, 43, 0
        # upr = np.array([29, 116, 123]) #best result for 29, 116, 123

        # Utwórz maskę
        lwr = np.array([l_h, l_s, l_v]) #best result for 0 0 0
        upr = np.array([u_h, u_s, u_v]) #best result for 179 255 76
        mask = cv2.inRange(hsv, lwr, upr)

        # Pokaż maskę
        cv2.imshow("Maska", mask)
        
        # Przerwij pętlę po naciśnięciu klawisza "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    #oputput from manual searching of lwr and upr
    # lwr = np.array([0, 0, 0]) #best result for 0 0 0
    # upr = np.array([179, 255, 76]) #best result for 179 255 76
    msk = cv2.inRange(hsv, lwr, upr)
    return msk

# Wczytaj obraz
img = cv2.imread("chessboard_empty_top.jfif")
img = cv2.resize(img, (600, 600))  

msk = find_mask_parameters(img)

# Extract chess-board
krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
dlt = cv2.dilate(msk, krn, iterations=5)
res = 255 - cv2.bitwise_and(dlt, msk)

# Displaying chess-board features
res = np.uint8(res)
ret, corners = cv2.findChessboardCorners(res, (7, 7),
                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
if ret:
    print(corners)
    fnl = cv2.drawChessboardCorners(img, (7, 7), corners, ret)
    cv2.imshow("fnl", fnl)
    cv2.waitKey(0)
else:
    print("No Checkerboard Found")
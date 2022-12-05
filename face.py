import dlib

img = dlib.load_rgb_image("C:\Users\15083\Desktop\Fall 2022\ECE H315 - Honors Project\ece315h-pt2\0da9ef7c8f852bac98b57d03926465c7312bf0d9accf1a4d56667112.jpg")
win = dlib.image_window(img)
detector = dlib.get_frontal_face_detector()
face = detector(img)
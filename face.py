import dlib

img = dlib.load_rgb_image("0da9ef7c8f852bac98b57d03926465c7312bf0d9accf1a4d56667112.jpg")
win = dlib.image_window(img)
detector = dlib.get_frontal_face_detector()
face = detector(img)
win.add_overlay(face)
win.wait_until_closed()
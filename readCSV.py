import numpy as np
import sys


class Subject:

    def __init__(self, face_x, face_y, face_width, face_height, facial_landmarks, expression, valence, arousal):
        self.face_x = float(face_x)
        self.face_y = float(face_y)
        self.face_width = float(face_width)
        self.face_height = float(face_height)
        self.__facial_landmarks = self.identify_facial_landmarks(facial_landmarks)
        self.__standardized_landmarks = self.standardize_landmarks()
        self.__expression = int(expression)
        self.__valence = float(valence)
        self.__arousal = float(arousal)
        self.array = self.get_dimensions()

    def identify_facial_landmarks(self, string):
        coordinates = string.rstrip().split(';')
        list = []
        for i in range(0,len(coordinates),2):
            list += [(float(coordinates[i]), float(coordinates[i+1]))]
        return list

    def get_facial_landmarks(self):
        return self.__facial_landmarks

    def standardize_landmarks(self):
        coordinates = []
        for coordinate in self.get_facial_landmarks():
            x = (coordinate[0] - self.face_x) / self.face_width
            y = (coordinate[1] - self.face_y) / self.face_height
            coordinates.append((x,y))
        return coordinates

    def get_standardized_landmarks(self):
        return self.__standardized_landmarks

    def get_dimensions(self):
        list = []
        for subject in self.get_standardized_landmarks():
            #list.append(subject[0])
            list.append(subject[1])
        return list

    def get_valence(self):
        return self.__valence

    def get_arousal(self):
        return self.__arousal

    def get_expression(self):
        return self.__expression


class Database:

    def __init__(self, filename):
        self.__filename = filename + '.csv'
        self.__listsubjects = []
        self.readfile()

    def readfile(self):
        try: file = open(self.__filename, 'r')
        except:
            print("File does not exist")
            sys.exit(0)
        firstline = True
        for line in file:
            if firstline:
                firstline = False
            else:
                line.rstrip()
                subDirectory_filePath, face_x, face_y, face_width, face_height, facial_landmarks, expression, valence, arousal = line.split(',')
                if float(valence) > -2 and float(arousal) > -2:
                    self.__listsubjects.append(Subject(face_x, face_y, face_width, face_height, facial_landmarks, expression, valence, arousal))
        file.close()

    def get_listsubjects(self):
        return self.__listsubjects

    def return_array(self):
        list = []
        for subject in self.get_listsubjects():
            list.append(subject.array)
        return np.array(list)

    def return_target(self):
        list = []
        for subject in self.get_listsubjects():
            list.append(subject.get_expression())
        return np.array(list)

    def return_valence(self):
        list = []
        for subject in self.get_listsubjects():
            list.append(subject.get_valence())
        return np.array(list)
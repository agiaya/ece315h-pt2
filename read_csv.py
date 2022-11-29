import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Subject:

    def __init__(self, name, subDirectory_filePath, face_x, face_y, face_width, face_height, facial_landmarks, expression, valence, arousal):
        self.name = name
        self.__subDirectory_filePath = subDirectory_filePath
        self.face_x = float(face_x)
        self.face_y = float(face_y)
        self.face_width = float(face_width)
        self.face_height = float(face_height)
        self.__facial_landmarks = self.identify_facial_landmarks(facial_landmarks)
        self.__standardized_landmarks = self.standardize_landmarks()
        self.__expression = int(expression)
        self.__valence = float(valence)
        self.__arousal = float(arousal)
        self.important = self.get_dimensions()

    def __str__(self):
        return self.name

    def get_subDirectory_filePath(self):
        return self.__subDirectory_filePath

    def identify_facial_landmarks(self, string):
        coordinates = string.rstrip().split(';')
        list = []
        for i in range(0,len(coordinates),2):
            list += [(float(coordinates[i]), float(coordinates[i+1]))]
        return list

    def get_facial_landmarks(self):
        return self.__facial_landmarks

    def get_standardized_landmarks(self):
        return self.__standardized_landmarks

    def standardize_landmarks(self):
        coordinates = []
        for coordinate in self.get_facial_landmarks():
            x = (coordinate[0] - self.face_x) / self.face_width
            y = (coordinate[1] - self.face_y) / self.face_height
            coordinates.append((x,y))
        return coordinates

    def get_points(self):
        x = []
        y = []
        for i in self.__facial_landmarks:
            x.append(i[0])
            y.append(-i[1])
        return x,y

    def add_dimension(self,index1,index2):
        d = self.distance(self.get_standardized_landmarks()[index1],self.get_standardized_landmarks()[index2])
        self.important.append(d)

    def get_expression(self):
        return self.__expression

    def get_dimensions(self):
        list = []
        n = 68
        for i in range(n):
            for j in range(i,n):
                list.append(self.distance(self.get_standardized_landmarks()[i],self.get_standardized_landmarks()[j]))
        for i in range(n):
            for j in range(i,n):
                list.append(self.angle(self.get_standardized_landmarks()[i],self.get_standardized_landmarks()[j]))
        return list

    def get_valence(self):
        return self.__valence

    def get_arousal(self):
        return self.__arousal


    @staticmethod
    def distance(p1,p2):
        mag_squared = (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2
        return mag_squared**(1/2)

    @staticmethod
    def angle(p1,p2):
        y = p2[-1] - p1[-1]
        x = p2[0] - p1[0]
        if x == 0:
            if y > 0: return np.pi/2
            elif y == 0: return 0
            else: return -1*np.pi/2
        return np.arctan(y/x)


class Database:
    def __init__(self, filename):
        self.__filename = filename + '.csv'
        self.__listsubjects = []
        self.readfile()

    def __str__(self):
        string = ""
        for subject in self.get_listsubjects():
            string += str(subject) + '\n'
        return string

    def readfile(self):
        file = open(self.__filename, 'r')
        firstline = True
        i = 0
        for line in file:
            if firstline:
                firstline = False
                continue
            else:
                line.rstrip()
                subDirectory_filePath, face_x, face_y, face_width, face_height, facial_landmarks, expression, valence, arousal = line.split(',')
                self.__listsubjects.append(Subject("subject" + str(i), subDirectory_filePath, face_x, face_y, face_width, face_height, facial_landmarks, expression, valence, arousal))
            i += 1
        file.close()
    def get_listsubjects(self):
        return self.__listsubjects
    def return_array(self):
        list = []
        for subject in self.get_listsubjects():
            list.append(subject.important) # + [subject.get_expression()]
        array = np.array(list)
        return array
    def return_emotion(self):
        list = []
        list1 = self.return_array()
        for subject in self.get_listsubjects():
            list.append(subject.get_expression())
        emotion_list = list1 + list
        emotion_array = np.array(emotion_list)
        return emotion_array

    def return_result(self):
        list = []
        for subject in self.get_listsubjects():
            list.append(subject.get_expression())
        return np.array(list)

    def return_results(self):
        list = []
        for subject in self.get_listsubjects():
            list.append([subject.get_valence(),subject.get_arousal(),subject.get_expression()])
        return np.array(list)

#training = Database("training_short")
#array1 = np.array([[2,1,2,2],[4,5,5,4]])
#array = training.return_array()
#print(array)
#print(array.shape)


#x,y = training.get_listsubjects()[10].get_points()

#plt.plot(x,y,marker=".")
#plt.show()
# test again


#for subject in training.get_listsubjects():
    #print(subject.important)

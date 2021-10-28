import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import sys
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import tflearn
import random
import json
import string
import unicodedata
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize, QRect, QCoreApplication
from PyQt5 import QtGui

import pymysql.cursors


app = QCoreApplication.instance()
if app is None:
    app = QApplication(sys.argv)
    
model.load('model.tflearn')

def get_tf_record(sentence):
    global words
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))


db = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='saglik',
                             #charset='utf8_turkish_ci',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

baglanti = db.cursor()

        
        
class p_sinifi(QWidget):
    def __init__(self):
        super().__init__()        
        self.ekran_tasarla()
       
    def ekran_tasarla(self):
        
        self.txtGiris = QTextEdit(self) 
        self.txtGiris.setGeometry(10,40,200,200)
                        
        btnSorgula = QPushButton("Hastalık Bul",self)
        btnSorgula.clicked.connect(self.bul)
        btnSorgula.setGeometry(220,40,290,50)        
        
        layout = QGridLayout()
        self.setLayout(layout)
        layout.setColumnStretch(1, 4)
        layout.setRowStretch(2, 4)
        baglanti.execute("SELECT semptom FROM semptomlar")
        bg1 = baglanti.fetchall()
        names1=[]
        for i in bg1:
                names1.append(str(i['semptom']))                                            
        completer = QCompleter(names1)                             
        self.lineedit = QLineEdit()
        self.lineedit.setCompleter(completer)
        layout.addWidget(self.lineedit, 0, 0)
        self.lineedit.returnPressed.connect(self.onPressed)
        self.lineedit.setFixedWidth(500)
        
        self.label = QLabel(self)
        self.label.setFont(QtGui.QFont("Sanserif", 14))
        self.label.setGeometry(220,90,500,50)
        
        self.label2 = QLabel(self)
        self.label2.setFont(QtGui.QFont("Sanserif", 11))
        self.label2.setGeometry(220,140,500,50)        
        self.label3 = QLabel(self)
        self.label3.setFont(QtGui.QFont("Sanserif", 11))
        self.label3.setGeometry(220,170,500,50)
        self.label4 = QLabel(self)
        self.label4.setFont(QtGui.QFont("Sanserif", 11))
        self.label4.setGeometry(220,190,500,50)
        self.label5 = QLabel(self)
        self.label5.setFont(QtGui.QFont("Sanserif", 11))
        self.label5.setGeometry(220,210,500,50)
        
        self.setGeometry(500,30,520,300)
        self.setMinimumSize ( QSize ( 520 , 300 ))
        self.setMaximumSize ( QSize ( 520 , 300 ))
        self.setWindowTitle("Hastalık Teşhisi")
        self.show()
        
    def onPressed(self):
        self.txtGiris.insertPlainText(self.lineedit.text()+'\n')
        
    def selectionchange(self,i):
        self.txtGiris.insertPlainText(self.comboBox.currentText()+'\n')
    
    def bul(self):
        self.text=self.txtGiris.toPlainText()
        if self.text!="":
            self.label.setText("Teşhis:\n"+categories[np.argmax(model.predict([get_tf_record(self.text)]))])
            snc=categories[np.argmax(model.predict([get_tf_record(self.text)]))]
            baglanti.execute("SELECT poliklinik0,poliklinik1,poliklinik2,poliklinik3 FROM poliklinikler WHERE teshis='"+snc+"'")
            bg = baglanti.fetchall()
            for i in bg:
                if str(i['poliklinik0'])!='None':
                    self.label2.setText("Poliklinik Önerisi:\n"+str(i['poliklinik0']))
                if str(i['poliklinik1'])!='None':
                    self.label3.setText(str(i['poliklinik1']))
                if str(i['poliklinik2'])!='None':
                    self.label4.setText(str(i['poliklinik2']))
                if str(i['poliklinik3'])!='None':
                    self.label5.setText(str(i['poliklinik3']))
        else:
            self.label.setText("Hata!:\n Lütfen semptom giriniz...")
        
if __name__=='__main__':
    uygulama=QApplication(sys.argv)
    pencere = p_sinifi()
    sys.exit(uygulama.exec())


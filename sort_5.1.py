# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:53:37 2017

@author: Octave Chenavas
"""
import time
import numpy as np
import math
import MNIST_dataset as MNIST

start_time = time.time()

class MLP(object):
    
    def __init__(self, forme):
        """Size doit être une liste.
        Comportement: si sizes = [5,6,8], alors 3 layers de la taille indiquée;
        respect 5, 6 et 8 neurones. La 1ere couche recoit les entrées."""
        self.forme = forme
        self.b = [np.random.randn(x,1) for x in forme[1:]]
        self.W = [np.random.randn(x, y) for x,y in zip(forme[1:], forme[:-1])]
        self.z = None #vecteur des z de chaque couche
        self.a = None #vecteur de sig(a)
        self.V_APP = 0.09
        
    def manger_devant_rapide(self, a): #PAS RAPIDE EN FAIT !
        for W, b in zip(self.W, self.b):
            a = sigmoide(np.dot(W, a) + b)
        return a

    def manger_devant(self, a): #a vecteur d'entrée (couche 1)
        self.a = []
        self.z = []
        """entree: entrée de l=1 = a"""
        self.a.append(a)
        for W,b in zip(self.W, self.b):
            self.z.append(np.dot(W,a)+b)
            a = (sigmoide(self.z[-1]))
            self.a.append(a)

    def retropropagation(self, entrees, attendu):
        """Utilese machin pour faire un truc et actualiser les bidules #d4rkJules"""

        #Calcul et enregistrement des W et z de tt les couches, sortie
        entrees = np.array(entrees)
        self.manger_devant(entrees)
        obtenu = self.a[-1]
        
        #Calcul du premier delta. Récursif après
        delta = ((attendu-obtenu)/obtenu)*sigP(self.z[-1])

        #Création des listes n_W et n_b
        n_W = [np.zeros(W.shape) for W in self.W]
        n_b = [np.zeros(b.shape) for b in self.b]

        #Calcul de n_W[-1] et de n_b[-1]
        n_b[-1] = delta
        n_W[-1] = np.dot(delta, self.a[-2].transpose())

        #Calcul des deltas => des nablas pr tt les couches
        for l in range(2, len(self.forme)):
            delta = np.dot(self.W[-l+1].transpose(), delta)*sigP(self.z[-l])
            n_b[-l] = delta
            n_W[-l] = np.dot(delta, (self.a[-l-1]).transpose())

        #Actualisation des W et des b
        for l in range(len(self.W)): #Un peu bizzare comme facon de dire
            self.b[l] += self.V_APP*n_b[l]
            self.W[l] += self.V_APP*n_W[l]

    def entrainement(self, tests, resultats, cycles = 1):     
        for i in range(cycles):
            for t,r in zip(tests, resultats):
                self.retropropagation(t, r)            


def sigmoide(x):
    return 1/(1+np.exp(-x))

def sigP(x):
    return sigmoide(x)*(1-sigmoide(x))

def toutEnUn(forme, tests, resultats, cycles = 1000):
    reseau = MLP(forme)
    reseau.entrainement(tests, resultats)


if __name__ == '__main__':

    print("[DEBUT]")

    reseau = MLP([784, 100, 50, 10])

    db = MNIST.read()
    for i in range(1800):
        j = 1
        for i in db:
            j += 1
            #print([i[1]/256])
            reseau.entrainement([i[1]/256], [i[0]], 1)
            if j == 30000: break     
        
    db = list(MNIST.read())
    
    res = 0
    for i in range(50000, 59999):
        if np.argmax(db[i][0]) == np.argmax(reseau.manger_devant_rapide(db[i][1])) : res += 1
        #print("i: ", i, "A: ", np.argmax(db[i][0]), "O: ", np.argmax(reseau.manger_devant_rapide(db[i][1])))
    print(res/9999*100, "% de réussite")

    #l, p = db_MNIST[0]
    #print(p.shape)
    #MNIST.show(p)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("[FIN]")
  
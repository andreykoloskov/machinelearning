import numpy as np
import sys

__author__ = 'andreykoloskov'

class Em(object):
    def __init__(self, psi1, psi2, z, l):
        self.psi1 = float(psi1)
        self.psi2 = float(psi2)
        self.z = int(z)
        self.l = int(l)

    def generate(self):
        arr0 = np.random.binomial(1, self.psi1, self.z)
        arr1 = np.random.binomial(1, self.psi2, self.l - self.z)
        self.arr = np.concatenate((arr0, arr1))

    def generate_first_data(self):
        self.z0 = 0.4 * self.l
        self.pi0_1 = self.z0 * 1.0 / self.arr.size
        self.pi0_2 = 1 - self.pi0_1
        #self.psi0_1 = sum(self.arr[:z0]) * 1.0 / self.z0
        #self.psi0_2 = sum(self.arr[z0:]) * 1.0 / (self.arr.size - self.z0)
        self.psi0_1 = 0.6
        self.psi0_2 = 0.4
        print self.arr
        print self.pi0_1, self.pi0_2, self.psi0_1, self.psi0_2
        print self.z0

    def e_step(self):
        self.p_g_z_1 = np.zeros(self.arr.size)
        self.p_g_z_2 = np.zeros(self.arr.size)
        for i, y in enumerate(self.arr):
            p1 = (self.psi0_1 ** y) * ((1.0 - self.psi0_1) ** (1 - y))
            p2 = (self.psi0_2 ** y) * ((1.0 - self.psi0_2) ** (1 - y))
            self.p_g_z_1[i] = self.pi0_1 * p1 / (self.pi0_1 * p1 + self.pi0_2 * p2)
            self.p_g_z_2[i] = self.pi0_2 * p2 / (self.pi0_1 * p1 + self.pi0_2 * p2)
            #print p1, p2, self.p_g_z_1[i], self.p_g_z_2[i]

        #print self.p_g_z_1
        #print self.p_g_z_2

    def m_step(self):
        self.pi0_1 = sum(self.p_g_z_1) * 1.0 / self.l
        self.pi0_2 = sum(self.p_g_z_2) * 1.0 / self.l
        #self.psi0_1 = sum(self.p_g_z_1 * self.arr) / (sum(self.p_g_z_1))
        #self.psi0_2 = sum(self.p_g_z_2 * self.arr) / (sum(self.p_g_z_2))
        self.psi0_1 = sum(self.p_g_z_1 * self.arr) * 1.0 / (self.pi0_1 * self.l)
        self.psi0_2 = sum(self.p_g_z_2 * self.arr) * 1.0 / (self.pi0_2 * self.l)
        print self.pi0_1, self.pi0_2, self.psi0_1, self.psi0_2

def main():
    em = Em(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print em.psi1, em.psi2, em.z, em.l
    em.generate()
    em.generate_first_data()
    for i in range(10):
        em.e_step()
        em.m_step()

if __name__ == '__main__':
    main()

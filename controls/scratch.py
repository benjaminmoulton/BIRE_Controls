from time import time as tm
class test:
    def __init__(self,case=0):
        self.case = case
        if case == 0:
            self.runref = self.run0
            self.bunref = self.bun0
        elif case == 1:
            self.runref = self.run1
            self.bunref = self.bun1
        elif case == 2:
            self.runref = self.run2
            self.bunref = self.bun1
    
    def run0(self):
        a = 0
        return a
    def run1(self):
        a = 1
        return a
    def run2(self):
        a = 2
        return a
    
    def bun0(self):
        b = 0
        return b
    def bun1(self):
        b = 1
        return b
    
    def allrun(self):
        if self.case == 0:
            a = 0
        elif self.case == 1:
            a = 1
        elif self.case == 2:
            a = 2
        
        if self.case == 0:
            b = 0
        elif self.case == 1:
            b = 1
        return a,b

if __name__ == "__main__":

    # initialize
    cl = test(1)
    numrun = 10000000
    # test ref
    start = tm()
    for i in range(numrun):
        cl.runref()
        cl.bunref()
    ref_time = tm() - start

    # test iff
    start = tm()
    for i in range(numrun):
        cl.allrun()
    iff_time = tm() - start

    print("ref method time :",ref_time,"seconds")
    print("iff method time :",iff_time,"seconds")
    



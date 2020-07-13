class A:
    def __init__(self):
        self.a = 1
    def __getitem__(self, key):
        print(key)

#a = A()
#a[:, ::, ::-1, 1, 1:2, 1:3:1, :1, :1:-1]
#print()
def fun(b, *a):
    print(b)
    print(a)

fun(1, [2,3,4])
print(not True)
class B(A):
    pass
b = B()
if isinstance(b, B):
    print("ok")
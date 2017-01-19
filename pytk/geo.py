Q = namedtuple('Q', ['w', 'x', 'y', 'z'])

# class Q(namedtuple('Q', ['x', 'y', 'z', 'w']):
#     def normalize(self):
#         self.q /= la.norm(self.q)

#     def rotate(self, p):

class Q(object):
    def __init__(self, x=0., y=0., z=0., w=1.):
        self.x
        self.y
        self.z
        self.w

    @property
    def a(self):
        return = np.array([self.x, self.y, self.z, self.w])

    def __mul__(self, other):



if __name__ == '__main__':
    q = Quaternion()

    Q(1, 2, 3, 4).normalize()

# from: https://inareous.github.io/posts/opening-obj-using-py
# also checkout: https://pypi.org/project/PyWavefront/

class ObjLoader(object):
    def __init__(self, fileName=None):
        self.vertices = []
        self.faces = []
        ##
        if fileName:
            try:
                f = open(fileName)
                for line in f:
                    if line[:2] == "v ":
                        index1 = line.find(" ") + 1
                        index2 = line.find(" ", index1 + 1)
                        index3 = line.find(" ", index2 + 1)

                        vertex = (float(line[index1:index2]), float(
                            line[index2:index3]), float(line[index3:-1]))
                        self.vertices.append(vertex)

                    elif line[0] == "f":
                        string = line.replace("//", "/")
                        ##
                        i = string.find(" ") + 1
                        face = []
                        for item in range(string.count(" ")):
                            if string.find(" ", i) == -1:
                                face.append(string[i:-1])
                                break
                            face.append(string[i:string.find(" ", i)])
                            i = string.find(" ", i) + 1
                        ##
                        self.faces.append(tuple(face))

                f.close()
            except IOError:
                print(".obj file not found.")

    def export(self, filename):
        f = open(filename, "w")
        f.write("g ")
        f.write("\n")
        for vertex in self.vertices:
            line = "v " + " " + \
                str(vertex[0]) + " " + \
                str(vertex[1]) + " " + str(vertex[2])
            f.write(line)
            f.write("\n")
        f.write("g ")
        f.write("\n")
        for face in self.faces:
            line = "f " + " " + \
                str(face[0]) + " " + \
                str(face[1]) + " " + str(face[2])
            f.write(line)
            f.write("\n")
        f.close()

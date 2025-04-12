import ctypes
import numpy as np
import copy
import cv2
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

class VBO:
    def __init__(self, vertices):
        self.ID = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.ID)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    def Use(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.ID)
    def Delete(self):
        glDeleteBuffers(1, (self.ID,))

class IBO:
    def __init__(self, indices):
        self.ID = glGenBuffers(1)
        self.count = len(indices)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ID)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    def Use(self):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ID)
    def Delete(self):
        glDeleteBuffers(1, (self.ID,))

class FBO:
    def __init__(self, width, height):
        self.w = width
        self.h = height

        # Create framebuffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # Create colour buffer
        self.colourBuffer = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.colourBuffer)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.w, self.h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.colourBuffer, 0)

        # Create Depth Stencil buffer
        self.depthStencilBuffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depthStencilBuffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.w, self.h)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.depthStencilBuffer)

        # Unbind frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def Use(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

    def Unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def ReadColourBuffer(self):
        self.Use()

        data = glReadPixels(0, 0, self.w, self.h, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape((self.h, self.w, 3))
        image = np.flipud(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.Unbind()
        return image

    def ReadDepthBuffer(self):
        self.Use()

        # Read depth buffer
        depth_data = glReadPixels(0, 0, self.w, self.h, GL_DEPTH_COMPONENT, GL_FLOAT)
        depth_image = np.frombuffer(depth_data, dtype=np.float32).reshape((self.h, self.w))
        depth_image = np.flipud(depth_image)

        self.Unbind()
        return depth_image

    def Delete(self):
        glDeleteTextures(1, [self.colourBuffer])
        glDeleteRenderbuffers(1, [self.depthStencilBuffer])
        glDeleteFramebuffers(1, [self.fbo])

class LayoutElement:
    def __init__(self, type, count, normalized, typeSize):
        self.type = type
        self.count = count
        self.normalized = normalized
        self.typeSize = typeSize

class VBL:
    def __init__(self):
        self.elements = []
        self.layoutSize = 0
    def Push(self, elementType, count):
        if elementType == "float":
            self.elements.append(LayoutElement(GL_FLOAT, count, GL_FALSE, ctypes.sizeof(ctypes.c_float)))
            self.layoutSize += count * ctypes.sizeof(ctypes.c_float)
        elif elementType == "int":
            self.elements.append(LayoutElement(GL_INT, count, GL_FALSE, ctypes.sizeof(ctypes.c_int)))
            self.layoutSize += count * ctypes.sizeof(ctypes.c_int)
        elif elementType == "u_int":
            self.elements.append(LayoutElement(GL_UNSIGNED_INT, count, GL_FALSE, ctypes.sizeof(ctypes.c_uint)))
            self.layoutSize += count * ctypes.sizeof(ctypes.c_uint)
    def GetElements(self):
        return self.elements
    def GetStride(self):
        return self.layoutSize

class VAO:
    def __init__(self, vbo : VBO, vbl : VBL):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        vbo.Use()
        elements = vbl.GetElements()
        offset = 0
        for i in range(len(elements)):
            element = elements[i]
            glEnableVertexAttribArray(i)
            glVertexAttribPointer(i, element.count, element.type, element.normalized, ctypes.c_uint(vbl.layoutSize), ctypes.c_void_p(offset))
            offset += element.count * element.typeSize
    def Use(self):
        glBindVertexArray(self.vao)
    def Delete(self):
        glDeleteVertexArrays(1, (self.vao,))

class Shader:
    def __init__(self, vertex_shader, fragment_shader):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.ID = compileProgram(compileShader(vertex_shader, GL_VERTEX_SHADER), compileShader(fragment_shader, GL_FRAGMENT_SHADER))
        self.Use()
    def Use(self):
        glUseProgram(self.ID)
    def Delete(self):
        glDeleteProgram((self.ID,))

class Camera:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.near = 0.1
        self.far = 100.0

    def Update(self, shader, intrinsicMatrix, extrinsicMatrix):
        shader.Use()
        
        intrinsicMatrix1 = np.eye(4)
        intrinsicMatrix1[:3,:3] = intrinsicMatrix
        # print(intrinsicMatrix)

        example_point = np.array([3, 3, -10, 1], dtype=np.float32)
        tmp = intrinsicMatrix1 @ example_point
        # print("\nExample: ", tmp)

        tmp1 = (tmp[0]/(tmp[2]), tmp[1]/(tmp[2]))
        # print(tmp1)
        u = (tmp1[0] * (2.0/self.width)) - 1.0
        v = (tmp1[1] * (2.0/self.height)) - 1.0

        # print("u: ", u, "v: ", v,"\n")

        intrinsicMatrix = intrinsicMatrix1
        # intrinsicMatrix = intrinsicMatrix1

        intrinsicMatrixLocation = glGetUniformLocation(shader.ID, "intrinsicMatrix".encode('utf-8'))
        glUniformMatrix4fv(intrinsicMatrixLocation, 1, GL_TRUE, intrinsicMatrix)

        extrinsicMatrixLocation = glGetUniformLocation(shader.ID, "extrinsicMatrix".encode('utf-8'))
        glUniformMatrix4fv(extrinsicMatrixLocation, 1, GL_TRUE, extrinsicMatrix)


        widthLocation = glGetUniformLocation(shader.ID, "width".encode('utf-8'))
        glUniform1f(widthLocation, self.width)

        heightLocation = glGetUniformLocation(shader.ID, "height".encode('utf-8'))
        glUniform1f(heightLocation, self.height)

        nearLocation = glGetUniformLocation(shader.ID, "near".encode('utf-8'))
        glUniform1f(nearLocation, self.near)

        farLocation = glGetUniformLocation(shader.ID, "far".encode('utf-8'))
        glUniform1f(farLocation, self.far)
    
class Object:
    def __init__(self, shader, properties):
        self.properties = copy.deepcopy(properties)

        self.vbo = VBO(self.properties['vertices'])
        self.ibo = IBO(self.properties['indices'])
        self.vbl = VBL()

        self.vbl.Push("float", 3)
        self.vbl.Push("float", 3)

        self.vao = VAO(self.vbo, self.vbl)

        self.properties.pop('vertices')
        self.properties.pop('indices')

        # Create shaders
        self.shader = shader

    def Draw(self):
        
        position = self.properties['position']
        rotation = self.properties['rotation']
        scale = self.properties['scale']

        translation_matrix = np.array([[1,0,0, position[0]],
                                    [0,1,0, position[1]],
                                    [0,0,1, position[2]],
                                    [0,0,0,1]], dtype = np.float32)
        
        rotation_z_matrix = np.array([
                                    [np.cos(rotation[2]), -np.sin(rotation[2]), 0, 0],
                                    [np.sin(rotation[2]), np.cos(rotation[2]), 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]
                                ], dtype=np.float32)
        rotation_x_matrix = np.array([
                                    [1, 0, 0, 0],
                                    [0, np.cos(rotation[0]), -np.sin(rotation[0]), 0],
                                    [0, np.sin(rotation[0]), np.cos(rotation[0]), 0],
                                    [0, 0, 0, 1]
                                ], dtype=np.float32)
        rotation_y_matrix = np.array([
                                    [np.cos(rotation[1]), 0, np.sin(rotation[1]), 0],
                                    [0, 1, 0, 0],
                                    [-np.sin(rotation[1]), 0, np.cos(rotation[1]), 0],
                                    [0, 0, 0, 1]
                                ], dtype=np.float32)

        scale_matrix = np.array([[scale[0], 0,0,0],
                                [0,scale[1],0,0],
                                [0,0,scale[2],0],
                                [0,0,0,1]], dtype = np.float32)
        
        rotationMatrix = rotation_z_matrix @ rotation_y_matrix @ rotation_x_matrix # Roll then pitch then yaw in order (right to left applied)
        self.modelMatrix = translation_matrix @ rotationMatrix @ scale_matrix

        # Bind the shader, set uniforms, bind vao (automatically binds vbo) and ibo
        self.shader.Use()
        modelMatrixLocation = glGetUniformLocation(self.shader.ID, "modelMatrix".encode('utf-8'))
        glUniformMatrix4fv(modelMatrixLocation, 1, GL_TRUE, self.modelMatrix)
        
        colourLocation = glGetUniformLocation(self.shader.ID, "objectColour".encode('utf-8'))
        glUniform4f(colourLocation, self.properties["colour"][0], 
                    self.properties["colour"][1], 
                    self.properties["colour"][2], 
                    self.properties["colour"][3])
        self.vao.Use()
        self.ibo.Use()

        # Issue Draw call with primitive type
        glDrawElements(GL_TRIANGLES, self.ibo.count, GL_UNSIGNED_INT, None)

object_shader = {
    "vertex_shader" : '''
        
        #version 330 core
        layout(location = 0) in vec3 vertexPosition;
        layout(location = 1) in vec3 vertexNormal;

        out vec3 fragmentPosition;
        out vec3 fragmentNormal;

        uniform mat4 modelMatrix;
        uniform mat4 extrinsicMatrix;
        uniform mat4 intrinsicMatrix;
        uniform float near;
        uniform float far;
        uniform float width;
        uniform float height;

        void main() {
            // fragmentPosition = vertexPosition;
            fragmentNormal = vertexNormal;
            
            vec4 P_worldcoord = modelMatrix * vec4(vertexPosition, 1.0);
            fragmentPosition = P_worldcoord.xyz;
            
            vec4 P_camcoord = extrinsicMatrix * P_worldcoord;
            float z_coord = ((-1.0 * P_camcoord.z) - ((far+near)/2.0))/((far-near)/2.0);
            
            P_camcoord.z = (-1.0 * P_camcoord.z);
            vec4 tmp = intrinsicMatrix * P_camcoord;
            vec2 tmp1 = vec2(tmp.x/(tmp.z), tmp.y/(tmp.z));

            float u = (tmp1.x * (2.0/width)) - 1.0;
            float v = (tmp1.y * (2.0/height)) - 1.0;

            gl_Position = vec4(u, v, z_coord, 1.0);
            // gl_Position = vec4(P_camcoord.x, P_camcoord.y, z_coord, 1.0);
        }

        ''',

        "fragment_shader" : '''

        #version 330 core

        in vec3 fragmentPosition;
        in vec3 fragmentNormal;

        out vec4 outputColour;

        uniform vec4 objectColour;
        //uniform vec3 camPosition;

        void main() {
            
            
            vec3 lightDir = normalize(vec3(-1, 0, 1));
            vec3 normal = normalize(fragmentNormal);  // Ensure normal is unit length

            // **Diffuse Lighting**
            float diffuse = max(dot(normal, lightDir), 0.0);

            // **Specular Lighting**
            // vec3 viewDir = normalize(camPosition - fragmentPosition);  // Direction to camera
            // vec3 reflectDir = reflect(-lightDir, normal);             // Reflected light direction
            // float specular = pow(max(dot(reflectDir, viewDir), 0.0), 8);
            float specular = 0.0; // Placeholder for specular calculation

            // **Combine Lighting Effects**
            vec3 lighting = (0.1 + diffuse + specular) * objectColour.rgb;  // Ambient + Diffuse + Specular
            outputColour = vec4(lighting, objectColour.a); // Final color output
            
            // if(length(fragmentPosition) < 0.1) {
            //     outputColour = vec4(1.0, 0.0, 0.0, 1.0); // Red color for debugging
            //     }
            
            //outputColour = objectColour;
            // outputColour = vec4(1.0, 1.0, 1.0, 1.0);
        }

        '''

}

def LoadObj(filepath):
    positions = []
    normals = []
    vertices = []
    indices = []

    # Initialize min/max values to extreme values
    x_min, y_min, z_min = float('inf'), float('inf'), float('inf')
    x_max, y_max, z_max = float('-inf'), float('-inf'), float('-inf')

    with open(filepath, 'r') as file:
        for line in file:
            tokens = line.split()
            if not tokens:
                continue
            
            if tokens[0] == 'v':  # Position vertex
                x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
                positions.append((x, y, z))

                # Update min/max bounds
                x_min, y_min, z_min = min(x_min, x), min(y_min, y), min(z_min, z)
                x_max, y_max, z_max = max(x_max, x), max(y_max, y), max(z_max, z)

            elif tokens[0] == 'vn':  # Normal vertex
                normals.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))

            elif tokens[0] == 'f':  # Face
                for vertex in tokens[1:]:
                    parts = vertex.split('//')  # Format: v//vn (No texture coordinates)
                    pos_idx = int(parts[0]) - 1  # Convert to 0-based index
                    norm_idx = int(parts[1]) - 1  # Convert to 0-based index

                    # Append position + normal as a new vertex
                    vertices.extend(positions[pos_idx] + normals[norm_idx])

                    # Add new index (since each face has new vertices)
                    indices.append(len(vertices) // 6 - 1)  # Each vertex has 6 floats

    return (np.array(vertices, dtype=np.float32), 
            np.array(indices, dtype=np.uint32), 
            np.array([x_min, y_min, z_min], dtype=np.float32), 
            np.array([x_max, y_max, z_max], dtype=np.float32)) 

def GetObjProps(obj_filepath):
    objectVerts, objectInds, objectMinPoint, objectMaxPoint = LoadObj(obj_filepath)
    print("ObjMin: ", objectMinPoint, "\nObjMax: ", objectMaxPoint)
    objectProps = {
        'vertices' : np.array(objectVerts, dtype = np.float32),
        
        'indices' : np.array(objectInds, dtype = np.uint32),

        'position' : np.array([0, 0, 0.1], dtype = np.float32),

        'rotation' : np.array([0, 0, 0], dtype = np.float32),

        'scale' : np.array([0.1, 0.1, 0.1], dtype = np.float32),

        'colour' : np.array([1.0, 1.0, 1.0, 1.0], dtype = np.float32),
    }
    return objectProps

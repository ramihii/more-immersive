""" This program loads a model with PyASSIMP, and display it.

Based on:
- pygame code from http://3dengine.org/Spectator_%28PyOpenGL%29
- http://www.lighthouse3d.com/tutorials
- http://www.songho.ca/opengl/gl_transform.html
- http://code.activestate.com/recipes/325391/
- ASSIMP's C++ SimpleOpenGL viewer

Authors: Séverin Lemaignan, 2012-2016

Modified by Joonatan Kuosa
"""

import os
import logging
from functools import reduce

logger = logging.getLogger("pyassimp")
gllogger = logging.getLogger("OpenGL")
gllogger.setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

import OpenGL

OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
# OpenGL.ERROR_ON_COPY = True
# OpenGL.FULL_LOGGING = True
from OpenGL.GL import *
from OpenGL.arrays import vbo
from OpenGL.GL import shaders

import pygame
import pygame.font
import pygame.image

import math, random
import numpy as np

import pyassimp
from pyassimp.postprocess import *
from pyassimp.helper import *

# Entities type
ENTITY = "entity"
CAMERA = "camera"
MESH = "mesh"

ROTATION_180_X = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)

BASIC_VERTEX_SHADER = """
#version 130

uniform mat4 u_viewProjectionMatrix;
uniform mat4 u_modelMatrix;
uniform mat3 u_normalMatrix;
uniform vec3 u_lightPos;

uniform vec4 u_materialDiffuse;

in vec3 a_vertex;
in vec3 a_normal;

out vec4 v_color;

void main(void)
{
    // Now the normal is in world space, as we pass the light in world space.
    vec3 normal = u_normalMatrix * a_normal;

    float dist = distance(a_vertex, u_lightPos);

    // go to https://www.desmos.com/calculator/nmnaud1hrw to play with the parameters
    // att is not used for now
    float att=1.0/(1.0+0.8*dist*dist);

    vec3 surf2light = normalize(u_lightPos - a_vertex);
    vec3 norm = normalize(normal);
    float dcont=max(0.0,dot(norm,surf2light));

    float ambient = 0.3;
    float intensity = dcont + 0.3 + ambient;

    v_color = u_materialDiffuse  * intensity;

    gl_Position = u_viewProjectionMatrix * u_modelMatrix * vec4(a_vertex, 1.0);
}
"""

BASIC_FRAGMENT_SHADER = """
#version 130

in vec4 v_color;

void main() {
    gl_FragColor = v_color;
}
"""

DEFAULT_CLIP_PLANE_NEAR = 0.001
DEFAULT_CLIP_PLANE_FAR = 1000.0


def get_world_transform(scene, node):
    # scene.rootnode.parent == scene (was expecting it to be None)
    if node == scene.rootnode:
        return scene.rootnode.transformation

    # TODO
    # This seems extremely costly
    # first we do full list append O(N) with allocations
    # then we reverse the list O(N) again (assuming just ref swapping)
    # then we run reduce on the list O(N) again
    # when you could just run recursive matrix multiplications
    # while node.parent: do: world_mat(parent) * node
    parents = reversed(_get_parent_chain(scene, node, []))
    parent_transform = reduce(np.dot, [p.transformation for p in parents])
    return np.dot(parent_transform, node.transformation)


def _get_parent_chain(scene, node, parents):
    parent = node.parent

    parents.append(parent)

    if parent == scene.rootnode:
        return parents

    return _get_parent_chain(scene, parent, parents)


def render_axis( transformation=np.identity(4, dtype=np.float32),
                label=None,
                size=0.2):
    m = transformation.transpose()  # OpenGL row major

    glPushMatrix()
    glMultMatrixf(m)

    glLineWidth(1)

    glBegin(GL_LINES)

    # draw line for x axis
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(size, 0.0, 0.0)

    # draw line for y axis
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, size, 0.0)

    # draw line for Z axis
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, size)

    glEnd()

    if label:
        showtext(label)

    glPopMatrix()

def render_grid():

    glLineWidth(1)
    glColor3f(0.5, 0.5, 0.5)
    glBegin(GL_LINES)
    for i in range(-10, 11):
        glVertex3f(i, -10.0, 0.0)
        glVertex3f(i, 10.0, 0.0)

    for i in range(-10, 11):
        glVertex3f(-10.0, i, 0.0)
        glVertex3f(10.0, i, 0.0)
    glEnd()

def showtext(text, x=0, y=0, z=0, size=20):

    # TODO: alpha blending does not work...
    # glEnable(GL_BLEND)
    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, (10, 10, 10, 255),
                              (255 * 0.18, 255 * 0.18, 255 * 0.18, 0))
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glRasterPos3d(x, y, z)
    glDrawPixels(text_surface.get_width(),
                 text_surface.get_height(),
                 GL_RGBA, GL_UNSIGNED_BYTE,
                 text_data)

    # glDisable(GL_BLEND)


def prepare_gl_buffers(mesh):

    mesh.gl = {}

    # Fill the buffer for vertex and normals positions
    v = np.array(mesh.vertices, 'f')
    n = np.array(mesh.normals, 'f')

    mesh.gl["vbo"] = vbo.VBO(np.hstack((v, n)))

    # Fill the buffer for vertex positions
    mesh.gl["faces"] = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.gl["faces"])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 np.array(mesh.faces, dtype=np.int32),
                 GL_STATIC_DRAW)

    mesh.gl["nbfaces"] = len(mesh.faces)

    # Unbind buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)


def glize(scene, node):

    logger.info("Loading node <%s>" % node)

    node.transformation = node.transformation.astype(np.float32)

    if node.meshes:
        node.type = MESH
    else:
        node.type = ENTITY

    for child in node.children:
        glize(scene, child)


class DefaultCamera:
    def __init__(self, w, h, fov):
        self.name = "default camera"
        self.type = CAMERA
        self.clipplanenear = DEFAULT_CLIP_PLANE_NEAR
        self.clipplanefar = DEFAULT_CLIP_PLANE_FAR
        self.aspect = w / h
        self.horizontalfov = fov * math.pi / 180

        self.transformation = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, -6],
                                        [0, 0, 0, 1.]], dtype=np.float32)


        ROTATION_180_Z = np.array([[-1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=np.float32)
        self.transformation = np.dot(self.transformation, ROTATION_180_Z)

    def __str__(self):
        return self.name


class PyAssimp3DViewer:
    base_name = "PyASSIMP 3D viewer"

    def __init__(self, models, w=1024, h=768):

        self.w = w
        self.h = h

        # Set window position
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)

        pygame.init()
        pygame.display.set_caption(self.base_name)
        pygame.display.set_mode((w, h), pygame.OPENGL | pygame.DOUBLEBUF)

        glClearColor(0.18, 0.18, 0.18, 1.0)

        self.prepare_shaders()

        self.backgroundTex = None

        self.scenes = []
        # stores the OpenGL vertex/faces/normals buffers pointers
        self.meshes = {}

        self.current_cam = DefaultCamera(self.w, self.h, fov=70)
        self.set_camera_projection()

        for model in models:
            self.load_model(model)

    def prepare_shaders(self):

        ### Base shader
        vertex = shaders.compileShader(BASIC_VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment = shaders.compileShader(BASIC_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)

        self.shader = shaders.compileProgram(vertex, fragment)

        self.set_shader_accessors(('u_modelMatrix',
                                   'u_viewProjectionMatrix',
                                   'u_normalMatrix',
                                   'u_lightPos',
                                   'u_materialDiffuse'),
                                  ('a_vertex',
                                   'a_normal'), self.shader)


    @staticmethod
    def set_shader_accessors(uniforms, attributes, shader):
        # add accessors to the shaders uniforms and attributes
        for uniform in uniforms:
            location = glGetUniformLocation(shader, uniform)
            if location in (None, -1):
                raise RuntimeError('No uniform: %s (maybe it is not used '
                                   'anymore and has been optimized out by'
                                   ' the shader compiler)' % uniform)
            setattr(shader, uniform, location)

        for attribute in attributes:
            location = glGetAttribLocation(shader, attribute)
            if location in (None, -1):
                raise RuntimeError('No attribute: %s' % attribute)
            setattr(shader, attribute, location)

    def load_model(self, path, postprocess=aiProcessPreset_TargetRealtime_MaxQuality):
        logger.info("Loading model:" + path + "...")

        if postprocess:
            self.scenes.append(pyassimp.load(path, postprocess))
        else:
            self.scenes.append(pyassimp.load(path))
        logger.info("Done.")

        scene = self.scenes[-1]
        # log some statistics
        logger.info("  meshes: %d" % len(scene.meshes))
        logger.info("  total faces: %d" % sum([len(mesh.faces) for mesh in scene.meshes]))
        logger.info("  materials: %d" % len(scene.materials))
        self.bb_min, self.bb_max = get_bounding_box(scene)
        logger.info("  bounding box:" + str(self.bb_min) + " - " + str(self.bb_max))

        self.scene_center = [(a + b) / 2. for a, b in zip(self.bb_min, self.bb_max)]

        for index, mesh in enumerate(scene.meshes):
            prepare_gl_buffers(mesh)

        glize(scene, scene.rootnode)

        # Finally release the model
        pyassimp.release(scene)
        logger.info("Ready for 3D rendering!")

    """
    def set_overlay_projection(self):
        glViewport(0, 0, self.w, self.h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, self.w - 1.0, 0.0, self.h - 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    """

    def set_camera_projection(self, camera=None):

        if not camera:
            camera = self.current_cam

        znear = DEFAULT_CLIP_PLANE_NEAR
        zfar = DEFAULT_CLIP_PLANE_FAR
        aspect = camera.aspect
        fov = camera.horizontalfov

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Compute gl frustrum
        tangent = math.tan(fov / 2.)
        h = znear * tangent
        w = h * aspect

        # params: left, right, bottom, top, near, far
        glFrustum(-w, w, -h, h, znear, zfar)
        # equivalent to:
        # gluPerspective(fov * 180/math.pi, aspect, znear, zfar)

        self.projection_matrix = glGetFloatv(GL_PROJECTION_MATRIX).transpose()

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def render(self, wireframe=False, twosided=False):

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wireframe else GL_FILL)
        # Enabling CULL_FACE removes our background
        #glDisable(GL_CULL_FACE) if twosided else glEnable(GL_CULL_FACE)

        # FIXME (do we even want these)
        #for scene in self.scenes:
        #    self.recursive_helpers_render(scene.rootnode)

        ### Then, inner shading
        # glDepthMask(GL_TRUE)
        glCullFace(GL_BACK)

        shader = self.shader
        glUseProgram(shader)
        glUniform3f(shader.u_lightPos, -.5, -.5, .5)

        glUniformMatrix4fv(shader.u_viewProjectionMatrix, 1, GL_TRUE,
                           np.dot(self.projection_matrix, self.view_matrix))

        for scene in self.scenes:
            self.recursive_render(scene, scene.rootnode, shader)

        glUseProgram(0)


    """
    def recursive_helpers_render(self, node):
        m = get_world_transform(self.scene, node)

        render_axis(m, label=node.name if node != self.scene.rootnode else None)

        for child in node.children:
                self.recursive_helpers_render(child)
    """


    def render_mesh(self, mesh, wt, shader):
        stride = 24  # 6 * 4 bytes

        diffuse = mesh.material.properties["diffuse"]
        if len(diffuse) == 3:  # RGB instead of expected RGBA
            diffuse.append(1.0)
        glUniform4f(shader.u_materialDiffuse, *diffuse)
        # if ambient:
        #    glUniform4f( shader.Material_ambient, *mat["ambient"] )

        normal_matrix = np.linalg.inv(np.dot(self.view_matrix, wt)[0:3, 0:3]).transpose()
        glUniformMatrix3fv(shader.u_normalMatrix, 1, GL_TRUE, normal_matrix)

        glUniformMatrix4fv(shader.u_modelMatrix, 1, GL_TRUE, wt)

        vbo = mesh.gl["vbo"]
        vbo.bind()

        glEnableVertexAttribArray(shader.a_vertex)
        glEnableVertexAttribArray(shader.a_normal)

        glVertexAttribPointer(
            shader.a_vertex,
            3, GL_FLOAT, False, stride, vbo
        )

        glVertexAttribPointer(
            shader.a_normal,
            3, GL_FLOAT, False, stride, vbo + 12
        )

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.gl["faces"])
        glDrawElements(GL_TRIANGLES, mesh.gl["nbfaces"] * 3, GL_UNSIGNED_INT, None)

        vbo.unbind()
        glDisableVertexAttribArray(shader.a_vertex)

        glDisableVertexAttribArray(shader.a_normal)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)


    def recursive_render(self, scene, node, shader):
        """ Main recursive rendering method.
        """

        m = get_world_transform(scene, node)

        # Mesh rendering modes
        ###
        if node.type == MESH:
            for mesh in node.meshes:
                self.render_mesh(mesh, m, shader)


        for child in node.children:
            self.recursive_render(scene, child, shader)


    """
    def switch_to_overlay(self):
        glPushMatrix()
        self.set_overlay_projection()

    def switch_from_overlay(self):
        self.set_camera_projection()
        glPopMatrix()
    """

    def loop(self):

        pygame.display.flip()

        if not self.process_events():
            return False  # ESC has been pressed

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        return True

    def process_events(self):
        ok = True

        for evt in pygame.event.get():
            if evt.type == pygame.KEYDOWN:
                ok = (ok and self.process_keystroke(evt.key, evt.mod))

        return ok

    def process_keystroke(self, key, mod):

        if key in [pygame.K_ESCAPE, pygame.K_q]:
            return False

        return True

    def draw_background(self, frame):
        # TODO do we need to push/pop Projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDepthMask(GL_FALSE)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_TEXTURE_2D)
        if not self.backgroundTex:
            self.backgroundTex = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.backgroundTex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        # TODO is the frame data type unsigned byte?
        # TODO we call this every frame? is this right way of doing it?
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.shape[1], frame.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, frame)

        glBegin(GL_TRIANGLES)

        # TODO use vbuffer not these old commands
        glTexCoord2f(0.0, 0.0)
        # TODO we don't want to blend color here (we need to otherwise it's red)
        # tbh it should be fixed if we use a shader instead
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(0.0, 1.0, 0.0)

        glTexCoord2f(1.0, 0.0)
        glVertex3f(1.0, 1.0, 0.0)

        glTexCoord2f(0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)

        # second
        glTexCoord2f(0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)

        glTexCoord2f(1.0, 0.0)
        glVertex3f(1.0, 1.0, 0.0)

        glTexCoord2f(1.0, 1.0)
        glVertex3f(1.0, 0.0, 0.0)

        glEnd();

        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);


    def update_view_camera(self):

        self.view_matrix = np.linalg.inv(self.current_cam.transformation)

        # Rotate by 180deg around X to have Z pointing backward (OpenGL convention)
        self.view_matrix = np.dot(ROTATION_180_X, self.view_matrix)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixf(self.view_matrix.transpose())


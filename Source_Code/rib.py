# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "Rib Bones",
    "description": "Generate model of human ribs",
    "author": "Cecily <cecily.gao@yale.edu>",
    "version": (1, 0),
    "blender": (2, 83, 0),
    "location": "View 3D > Properties Panel",
    "category": "Object",
    }
    
    
# Libraries
import bpy, bmesh
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, EnumProperty
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras import object_utils

import math, os, numpy
import pandas as pd
from scipy.optimize import fsolve
import mathutils
from mathutils import Vector, Matrix


'''
Linear Regression Models for Rib Generation
'''

# Read csv files for linear regression parameters
path_to_data = '/Users/cecegao/Desktop/Fall 2020/Graphics/Midterm Project/Gao_Cecily_Graphics_Project_1/RibLinReg/'

S_x_df      = pd.read_csv(path_to_data + 'S_x.csv', index_col=0)
X_pk_df     = pd.read_csv(path_to_data + 'X_pk.csv', index_col=0)
Y_pk_df     = pd.read_csv(path_to_data + 'Y_pk.csv', index_col=0)
B_d_df      = pd.read_csv(path_to_data + 'B_d.csv', index_col=0)
B_p_df      = pd.read_csv(path_to_data + 'B_p.csv', index_col=0)
phi_pia_df  = pd.read_csv(path_to_data + 'phi_pia.csv', index_col=0)
alpha_ph_df = pd.read_csv(path_to_data + 'alpha_ph.csv', index_col=0)
alpha_ls_df = pd.read_csv(path_to_data + 'alpha_ls.csv', index_col=0)
alpha_bh_df = pd.read_csv(path_to_data + 'alpha_bh.csv', index_col=0)

# Linear Regression Model
def linReg(df, ribNum, age, sex, height, weight):
    I = df.at[ribNum,'I']* 10.0**df.at[ribNum,'I_sf']
    A = df.at[ribNum,'A']; A_sf = df.at[ribNum,'A_sf']
    S = df.at[ribNum,'S']; S_sf = df.at[ribNum,'S_sf']
    H = df.at[ribNum,'H']; H_sf = df.at[ribNum,'H_sf']
    W = df.at[ribNum,'W']; W_sf = df.at[ribNum,'W_sf']
    age *= 10.0**A_sf
    sex *= 10.0**S_sf
    height *= 10.0**H_sf
    weight *= 10.0**W_sf
    return I + A*age + S*sex + H*height + W*weight

# Default values for custom parameter inputs
n = 6; a = 25; s = 1; h = 1.7; w = 70.0
default_S_x      = linReg(S_x_df, n, a, s, h, w)
default_X_pk     = linReg(X_pk_df, n, a, s, h, w)
default_Y_pk     = linReg(Y_pk_df, n, a, s, h, w)
default_B_d      = linReg(B_d_df, n, a, s, h, w)
default_B_p      = linReg(B_p_df, n, a, s, h, w)
default_phi_pia  = linReg(phi_pia_df, n, a, s, h, w)
default_alpha_ph = linReg(alpha_ph_df, n, a, s, h, w)
default_alpha_ls = linReg(alpha_ls_df, n, a, s, h, w)
default_alpha_bh = linReg(alpha_bh_df, n, a, s, h, w)


#############################################################################################
#  PANEL
#############################################################################################

class RIB_PT_Class(Panel):
    bl_space_type = "VIEW_3D"
    bl_context = "objectmode"
    bl_region_type = "UI"
    bl_label = "Rib Model"
    bl_category = "Human Rib"
    bl_description = "Generate a model of a human rib"


    def draw(self, context):
        scn = context.scene
        settings = scn.ribs
        layout = self.layout

        row = layout.row(align=True)
        row.label(text='Input Parameters')
        row.scale_y = 1.5

        row = layout.row(align=True)
        row.label(text='Mode')
        row.prop(settings, 'mode',  expand=True)

        if context.scene.ribs.mode == 'S':
            row = layout.row(align=True)
            row.label(text='Subject Characteristics')
            
            row = layout.row(align=True)
            row.label(text='Sex', icon="OUTLINER_OB_ARMATURE")
            row.prop(settings, 'sex',  expand=True)
            
            row = layout.row(align=True)
            row.prop(settings, 'age',  slider=True)

            row = layout.row(align=True)
            row.prop(settings, 'height',  slider=True)
            row.prop(settings, 'weight',  slider=True)

        elif context.scene.ribs.mode == 'P':
            row = layout.row(align=True)
            row.label(text='Custom Rib Parameters')

            row = layout.row(align=True)
            row.prop(settings, 'X_pk',  slider=True)
            row.prop(settings, 'Y_pk',  slider=True)

            row = layout.row(align=True)
            row.prop(settings, 'B_d',  slider=True)
            row.prop(settings, 'B_p',  slider=True)

            row = layout.row(align=True)
            row.prop(settings, 'phi_pia',  slider=True)
            row.prop(settings, 'S_x',  slider=True)

            row = layout.row(align=True)
            row.prop(settings, 'alpha_ph',  slider=True)
            row.prop(settings, 'alpha_ls',  slider=True)        
            row.prop(settings, 'alpha_bh',  slider=True)        

        row = layout.row(align=True)
        row.label(text='Rendering Settings')
        row.scale_y = 1.5
        
        row = layout.row(align=True)
        row.label(text="Outline only")
        row.prop(settings, 'outlineOnly')
        row.label(text="Full set")
        row.prop(settings, 'fullSet')

        if context.scene.ribs.fullSet == False:
            row = layout.row(align=True)
            row.label(text="Rib number")
            row.prop(settings, 'ribNumber', slider=True)
        
        row = layout.row(align=True)
        row.label(text="Side count")
        row.prop(settings, 'sides', slider=True)
        
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("ribs.create", text="Create Ribcage" if context.scene.ribs.fullSet else "Create Rib", icon="BONE_DATA")

class RIB_Create(Operator):
    bl_idname = "ribs.create"
    bl_label = "Create Rib"
    bl_description = "Create rib"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Deselect selected objects
        for obj in bpy.context.selected_objects:
            bpy.context.view_layer.objects.active = obj
            bpy.context.active_object.select_set(False)

        sides = context.scene.ribs.sides
        outlineOnly = context.scene.ribs.outlineOnly
        fullSet = context.scene.ribs.fullSet

        # get distScale based on rib 6
        distScale = parameterizeRib(context, 6)['S_x']

        if not fullSet:
            generateRib(context, sides, outlineOnly, context.scene.ribs.ribNumber, distScale)

        else:
            ribSet = bpy.data.collections.new("Ribs")
            context.scene.collection.children.link(ribSet)

            for i in range (1, 13):
                rib = generateRib(context, sides, outlineOnly, i, distScale)
                mirror(rib)         # Mirror to get right rib
                ribSet.objects.link(rib)
                try:
                    bpy.context.scene.collection.objects.unlink(rib) 
                except:
                    print('Cannot unlink')
            
        return {'FINISHED'}


#############################################################################################
# RIB PARAMATERIZATION
#############################################################################################

# Planar Rotation Values
ribRotation = {
    1: math.radians(90),
    2: math.radians(75),
    3: math.radians(15),
    4: math.radians(5),
    5: math.radians(0),
    6: math.radians(0),
    7: math.radians(0),
    8: math.radians(0),
    9: math.radians(0),
    10: math.radians(0),
    11: math.radians(0),
    12: math.radians(-5)
}

# Parameterization of desired rib
def parameterizeRib(context, ribNum):
    """ Retrieve or compute parameters (S_x, X_pk, Y_pk, B_d, B_p, phi_pia) for the rib model

    :type context: Context\\
    :param context: Plugin context

    :rtype: Dictionary with parameters
    """

    d = {}

    age = context.scene.ribs.age
    sex = 1 if context.scene.ribs.sex=='F' else 0
    weight = context.scene.ribs.weight
    height = context.scene.ribs.height

    if context.scene.ribs.mode == 'S':
        d['S_x']     = linReg(S_x_df, ribNum, age, sex, height, weight)
        d['X_pk']    = linReg(X_pk_df, ribNum, age, sex, height, weight)
        d['Y_pk']    = linReg(Y_pk_df, ribNum, age, sex, height, weight)
        d['B_d']     = linReg(B_d_df, ribNum, age, sex, height, weight)
        d['B_p']     = linReg(B_p_df, ribNum, age, sex, height, weight)
        d['phi_pia'] = math.radians(linReg(phi_pia_df, ribNum, age, sex, height, weight))
        d['alpha_ph'] = math.radians(90-linReg(alpha_ph_df, ribNum, age, sex, height, weight))
        d['alpha_ls'] = math.radians(linReg(alpha_ls_df, ribNum, age, sex, height, weight))
        d['alpha_bh'] = math.radians(linReg(alpha_bh_df, ribNum, age, sex, height, weight))

    else:
        d['S_x']     = context.scene.ribs.S_x
        d['X_pk']    = context.scene.ribs.X_pk
        d['Y_pk']    = context.scene.ribs.Y_pk
        d['B_d']     = context.scene.ribs.B_d
        d['B_p']     = context.scene.ribs.B_p
        d['phi_pia'] = math.radians(context.scene.ribs.phi_pia)
        d['alpha_ph'] = math.radians(90-context.scene.ribs.alpha_ph)
        d['alpha_ls'] = math.radians(context.scene.ribs.alpha_ls)
        d['alpha_bh'] = math.radians(context.scene.ribs.alpha_bh)

    # print(d)
    return d


#############################################################################################
# RIB GENERATION
#############################################################################################

# Rib Generator Function
def generateRib(context, sides, outlineOnly, ribNum, distScale):
    d = parameterizeRib(context, ribNum)
    mesh = bpy.data.meshes.new(name="Rib_" + str(ribNum))
    bm = bmesh.new()

    adjusted_S_x, bm = generateDistalCurve(parameters=d, 
                            rib=ribNum,
                            sides=sides,
                            bm=bm)

    bm = generateProximalCurve(parameters=d, 
                                rib=ribNum,
                                sides=sides,
                                bm=bm)

    rib = loftedRib(bm, mesh, adjusted_S_x/100.0, d, ribNum, distScale, outline=outlineOnly)

    return rib


# Curve Generators
def generateDistalCurve(parameters, rib, sides, bm):
    """ Generate the distal curve of the rib

    :type parameters: Dictionary
    :param parameters: contains rib parameters

    :type rib: Int
    :param rib: rib number (1-12)

    :type n: Int
    :param n: number of sample points along curve

    :type sides: Int
    :param sxides: number of subdivision faces for the orthogonal faces

    :type bm: bmesh
    :param bm: rib mesh

    :rtype: adjusted S_x, bmesh with added geometries
    """

    B_d = parameters['B_d'] 
    X_pk = parameters['X_pk']
    Y_pk = parameters['Y_pk']
    S_x = parameters['S_x']

    def x(theta):
        return -1.0 * pow(math.e, B_d*theta) * math.cos(theta)
    def y(theta):
        return pow(math.e, B_d*theta) * math.sin(theta)

    def solveSlope(theta, x_pk, y_pk, M):
        return numpy.float((y(theta) - y_pk)/(x(theta) - x_pk) - M)

    # Find theta_peak
    theta_peak = 2.0 * math.atan(math.sqrt(B_d*B_d + 1.0) + B_d)

    '''
    Find theta_end
    '''
    # Calculate M
    M = (0.0 - Y_pk)/(1.0 - X_pk)

    # Calculate x_pk and y_pk
    x_pk = x(theta_peak)
    y_pk = y(theta_peak)

    # Solve for theta_end
    theta_end = -1; i = 1
    while theta_end <= theta_peak:
        theta_end = fsolve(solveSlope, [theta_peak+i], args=(x_pk, y_pk, M))[0]
        i += 1

    '''
    Find distal scale factor
    '''
    dist_xy = math.sqrt(pow(Y_pk, 2) + pow(1.0 - X_pk, 2))
    dist_XY = math.sqrt(pow(x(theta_peak)-x(theta_end), 2) + pow(y(theta_peak)-y(theta_end), 2))
    SF_d = dist_xy/dist_XY

    '''
    Scale and translate: final functions for X and Y
    '''
    def X(theta, theta_peak, SF_d):
        return (x(theta) - x_pk) * SF_d + X_pk
    def Y(theta, theta_peak, SF_d):
        return (y(theta) - y_pk) * SF_d + Y_pk

    '''
    Get n points on the curve from theta_peak to theta_end
    '''
    n = abs(math.ceil((theta_end - theta_peak)/0.01))

    for i in range(0, n):
        theta = theta_peak + (i*(theta_end - theta_peak)/(n-1))     
        x_val = X(theta, theta_peak, SF_d)
        y_val = Y(theta, theta_peak, SF_d)
        
        bm.verts.new((x_val, y_val, 0.0))
        generateOrthogonalCircle(bm, x_val, y_val, X_pk, Y_pk, rib, sides, True if i == n-1 else False)
        
        if i == n-1: adjustedS_x = S_x/x_val        # Adjust S_x based on transformed and normalized x-intercept

    return adjustedS_x, bm

def generateProximalCurve(parameters, rib, sides, bm): 
    """ Generate the proximal curve of the rib

    :type parameters: Dictionary
    :type rib: Int
    :type n: Int
    :type sides: Int
    :type bm: bmesh

    :rtype: bmesh with added geometries
    """

    phi_pia = parameters['phi_pia']
    B_p = parameters['B_p']
    X_pk = parameters['X_pk']
    Y_pk = parameters['Y_pk']

    def x(theta):
        return math.exp(B_p*theta) * math.cos(theta)
    def y(theta):
        return math.exp(B_p*theta) * math.sin(theta)
    
    def dx(theta):
        return B_p * x(theta) - y(theta)
    def dy(theta):
        return B_p * y(theta) + x(theta)

    # Computes the angle of a tangent line at a point on the curve relative to the horizontal
    def angleDelta(theta, true_value):
        y_pt = y(theta)
        x_pt = x(theta)

        # slope of tangent line
        if dx(theta) != 0:
            slope = dy(theta)/dx(theta)
        else:
            return -math.inf

        # find intercepts of tangent line
        y_intercept = y_pt - slope*x_pt
        x_intercept = -y_intercept/slope

        if x_intercept-x_pt != 0:
            ratio = y_pt/(x_intercept-x_pt)
        else:
            return -math.inf

        return math.atan(ratio) - true_value    # = 0 when the angle is the one we're looking for 

    steps = [p*math.pi/10 for p in range(0, 20)]    # theta_cut and theta_tangent must lie between 0 and 2*pi
    
    '''
    Compute the first 'cut' of the spiral (theta_cut)
    '''
    # Find angle between horizontal from rib peak
    alpha = math.atan(Y_pk/X_pk)

    # Find theta_cut: the theta for which the line tangent to the curve forms an angle of alpha to the horizontal
    theta_cut = -1
    for i in steps:
        temp = fsolve(angleDelta, i, args=(alpha))[0]
        if temp < math.pi and temp > 0 and (theta_cut == -1 or (temp > theta_cut and theta_cut != -1)):
            theta_cut = temp
    
    '''
    Compute the second 'cut' of the spiral (theta_tangent)
    '''
    # Find alpha_tangent
    alpha_tangent = phi_pia - alpha

    # Find theta_tangent: the theta for which the line tangent to the curve forms an angle of alpha_tangent to the horizontal
    theta_tangent = -1
    for i in steps:
        temp = fsolve(angleDelta, i, args=(-alpha_tangent))[0]
        if temp < 2*math.pi and temp > theta_cut:
            if theta_tangent == -1: # or (temp > theta_tangent): # and y(temp) >= y(theta_cut)):
                theta_tangent = temp
                break

    # print("Theta-cut: ", theta_cut, ", Theta-tangent: ", theta_tangent)

    # If a theta_tangent wasn't found, the combination of inputs cannot produce a feasible rib model
    if theta_tangent == -1:
        raise RuntimeError ('A combination of one or more input parameters are invalid for rib ' + str(rib) +
                             '. Please try another set of parameters') from OSError

    '''
    Compute the linear end of sprial (0,0) in transformed space
    '''
    x_0 = x(theta_tangent) - ((y(theta_tangent) - y(theta_cut))
                                /math.tan(alpha_tangent))

    '''
    Get transformation values
    '''
    len_xy = x(theta_cut) - x_0
    len_XY = math.sqrt(Y_pk**2 + X_pk**2)
    SF_p = len_XY/len_xy

    # Transformation matrices
    translation = numpy.matrix([[1, 0, -x_0], 
                                [0, 1, -y(theta_cut)],
                                [0, 0, 1]])

    scale = numpy.matrix([[SF_p, 0, 0], 
                          [0, SF_p, 0],
                          [0, 0, 1]])

    rotation = numpy.matrix([[math.cos(alpha), -math.sin(alpha), 0], 
                             [math.sin(alpha), math.cos(alpha), 0],
                             [0, 0, 1]])
                             
    n = abs(math.ceil(theta_cut-theta_tangent/0.03))

    for i in range(0, n):
        theta = theta_tangent + (i*(theta_cut-theta_tangent)/n)
        untransformed = numpy.vstack([x(theta), y(theta), 1])

        # Transform point with homogeneous coordinates
        transformed = translation.dot(untransformed)
        transformed = scale.dot(transformed)
        transformed = rotation.dot(transformed)

        # Normalize post-transformation point
        transformed_norm = transformed/transformed[2,0]

        x_val = transformed_norm[0,0]
        y_val = transformed_norm[1,0]

        if y_val < 0: continue

        bm.verts.new((x_val, y_val, 0.0))
        generateOrthogonalCircle(bm, x_val, y_val, X_pk, Y_pk, rib, sides, False)

        '''
        Add linear portion from (X(theta_tangent), Y(theta_tangent)) to (0,0) in transformed space
        '''
        if i == 0 and y_val > 0:
            samples_linear = math.ceil(y_val/0.003)
            for point in range (0, samples_linear):
                x_lin_val = point*(x_val/samples_linear)
                y_lin_val = point*(y_val/samples_linear)
                bm.verts.new((x_lin_val, y_lin_val, 0.0))

                generateOrthogonalCircle(bm, x_lin_val, y_lin_val, X_pk, Y_pk, rib, sides, True if point == 0 else False)

    return bm


# Sweep/Lofting
def generateOrthogonalCircle(bm, x_val, y_val, X_pk, Y_pk, rib, sides, endCap=False): 
    """ Generate and transform orthogonal circle at a given vertex of mesh
    :type x_val: Float
    :param x_val: x-coordinate of vertex

    :type y_val: Float
    :param y_val: y-coordinate of vertex

    :type X_pk: Float
    :param X_pk: x-coordinate of rib peak

    :type Y_pk: Float
    :param Y_pk: y-coordinate of rib peak

    :type rib: Int
    :type sides: Int

    :type endCap: Boolean
    :param endCap: True if face should be generated

    :rtype: None
    """

    # Translation
    mat_loc = Matrix.Translation((x_val, y_val, 0))

    # Rotation
    mat_rot1 = Matrix.Rotation(math.radians(90.0), 4, 'X')  # rotation in the x-y plane 
    r = math.atan2(y_val, x_val-X_pk)
    mat_rot2 = Matrix.Rotation(r, 4, 'Y')                   # rotation relative to center of (X_pk, 0)
    mat_rot3 = Matrix.Rotation(ribRotation[rib], 4, 'Z')    # rib number rotation
    mat_rot = mat_rot1 @ mat_rot2 @ mat_rot3

    # Scaling
    mat_sca_x = Matrix.Scale(0.25, 4, (1.0, 0.0, 0.0))
    mat_sca_y = Matrix.Scale(0.7, 4, (0.0, 1.0, 0.0))
    mat_sca = mat_sca_x @ mat_sca_y
    
    # Overall Transformation Matrix
    transformation = mat_loc @ mat_rot @ mat_sca

    bmesh.ops.create_circle(bm,
                            cap_ends=endCap,
                            radius=0.05,
                            segments=sides,
                            matrix=transformation)

def loftedRib(bm, mesh, scale, parameters, ribNum, distScale, outline=True):
    """ Loft through circles
    :rtype: Object (rip)
    """

    if not outline:
        ret = bmesh.ops.bridge_loops(bm, edges=bm.edges)
        bmesh.ops.subdivide_edges(bm,
                                edges=ret['edges'],
                                smooth=1.0,
                                smooth_falloff='LINEAR',
                                cuts=4)

    bm.to_mesh(mesh)
    mesh.update()
    rib = object_utils.object_data_add(bpy.context, mesh)

    S = Matrix.Scale(scale, 4)
    print(ribNum, parameters['alpha_ph'])
    R_ph = Matrix.Rotation(parameters['alpha_ph'], 4, 'Z')
    # R_bh = Matrix.Rotation(parameters['alpha_bh'], 4, 'Y')
    # R_ls = Matrix.Rotation(parameters['alpha_ls'], 4, 'Z')

    # Translate ribs into a stack (does NOT follow real path)
    vertical_difference = 0.1 + 0.04/95.0*distScale
    T = Matrix.Translation((0, vertical_difference, vertical_difference*(12-ribNum)))  

    mesh.transform(S)
    mesh.transform(R_bh)
    # mesh.transform(R_ph)
    # mesh.transform(R_ls)
    mesh.transform(T)

    mesh.update()

    return rib

def mirror(rib):
    """ Mirror rib over YZ plane
    :rtype: None
    """
    
    bpy.ops.object.mode_set(mode='EDIT')
    me = rib.data
    bm = bmesh.from_edit_mesh(me)

    bmesh.ops.mirror(bm,
                    geom=bm.faces[:]+bm.verts[:]+bm.edges[:],
                    axis='Y')

    bmesh.update_edit_mesh(me)

    bpy.ops.object.mode_set(mode='OBJECT')


#############################################################################################
# PROPERTIES
#############################################################################################

class RibSettings(PropertyGroup):
    mode : EnumProperty(
        description = "Input Mode",
        items = (('S', 'Subject', 'Model based on a subject\'s characteristics'),
                 ('P', 'Custom', 'Custom input based on six-parameter model')),
        default = "S"
    )

    # Subject Characteristics
    age : IntProperty(
        name = "Age",
        description = "Age of subject",
        default = 21,
        min = 0,
        max = 150,
        subtype = 'NONE'
    )   
    sex: EnumProperty(
        name = "Sex",
        description = "Sex of subject",
        items = (('F', 'Female', 'Female'), ('M', 'Male', 'Male')),
        default = "F"
    )
    height : FloatProperty(
        name = "Height",
        description = "Subject\'s height in meters",
        default = 1.7,
        min = 0,
        max = 3.0,
    )
    weight : FloatProperty(
        name = "Weight",
        description = "Subject\'s weight in kilograms",
        default = 70,
        min = 0,
        max = 500
    )

    # Custom Rib Parameters
    S_x : FloatProperty(
        name = "Scale",
        description = "Scale factor for relative rib size",
        default = default_S_x,
        min = 50,
        max = 250
    )
    X_pk : FloatProperty(
        name = "X_pk",
        description = "Normalized x-coordinate of rib peak",
        default = default_X_pk,
        min = 0.1,
        max = 0.4
    )
    Y_pk : FloatProperty(
        name = "Y_pk",
        description = "Normalized y-coordinate of rib peak",
        default = default_Y_pk,
        min = 0.1,
        max = 0.6
    )
    B_d : FloatProperty(
        name = "B_d",
        description = "Distal spiral shape factor",
        default = default_B_d,
        min = -1.0,
        max = 2.0,
    )
    B_p : FloatProperty(
        name = "B_p",
        description = "Proximal spiral shape factor",
        default = default_B_p,
        min = -1.0,
        max = 0.5
    )
    phi_pia : FloatProperty(
        name = "Φ_pia",
        description = "Inner angle of the proximal curve, relative to the x-axis (in degrees)",
        default = default_phi_pia,
        min = 45,
        max = 130
    )
    alpha_ph : FloatProperty(
        name = "α_ph",
        description = "Normalized x-coordinate of rib peak",
        default = default_alpha_ph,
        min = 35,
        max = 85
    )
    alpha_ls : FloatProperty(
        name = "α_ph",
        description = "Normalized y-coordinate of rib peak",
        default = default_alpha_ls,
        min = 15,
        max = 75
    )
    alpha_bh : FloatProperty(
        name = "α_bh",
        description = "Scale factor for relative rib size",
        default = default_alpha_bh,
        min = -10,
        max = 35
    )

    # Rendering Settings
    fullSet: BoolProperty(
        name = "",
        description = "Full set of ribs or just one",
        default = False,
    )
    outlineOnly: BoolProperty(
        name = "",
        description = "No lofting, just the skeleton of the model (heh)",
        default = False,
    )
    ribNumber: IntProperty(
        name = "#",
        description = "Indicate rib number, 1-12, which influences rotation of the rib",
        default = 6,
        min = 1,
        max = 12,
    )
    n: IntProperty(
        name = "n",
        description = "Number of sample points to take along curves",
        default = 30,
        min = 10,
        max = 200,
    )
    sides: IntProperty(
        name = "sides",
        description = "Number of sides of the orthogonal circles (more sides -> smoother appearance)",
        default = 20,
        min = 5,
        max = 200,
    )


#############################################################################################
# REGISTER ADD-ON
#############################################################################################

classes = (
    RIB_PT_Class,
    RIB_Create,
    RibSettings,
    )

reg, unreg = bpy.utils.register_classes_factory(classes)

# Register
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ribs = PointerProperty(type=RibSettings)

# Unregister
def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.ribs


if __name__ == "__main__":
    register()
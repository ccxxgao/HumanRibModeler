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
    "name": "Periosteum Texturing",
    "description": "Apply bone-like texturing to objects",
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


#############################################################################################
#  PANEL
#############################################################################################

class PERIOSTEUM_PT_Class(Panel):
    bl_space_type = "VIEW_3D"
    bl_context = "objectmode"
    bl_region_type = "UI"
    bl_label = "Bone Texturing"
    bl_category = "Periosteum"

    def draw(self, context):
        scn = context.scene
        settings = scn.periosteum
        layout = self.layout
        
        row = layout.row(align=True)
        row.label(text="Surface Cracks")
        row.prop(settings, 'cracks')
        
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("periosteum.create", text="Texture Object(s)", icon="BONE_DATA")

class PERIOSTEUM_Texturing(Operator):
    bl_idname = "periosteum.create"
    bl_label = "Texture Object"
    bl_description = "Apply periosteum texturing to object"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context) -> bool:
        return bool(context.selected_objects)

    def execute(self, context):
        cracks = context.scene.periosteum.cracks

        # Check if materials already exist
        name = 'Cracked_Periosteum' if cracks else 'Periosteum'
        peri_mat = bpy.data.materials.get(name)
        
        # If not, create new material
        if peri_mat is None:
            # For cracks, copy 'Periosteum' material (if it exists) to avoid unnecessarily repeating computations
            base_mat = bpy.data.materials.get('Periosteum')
            peri_mat = createPeriosteumMaterial(mat=base_mat.copy() if (base_mat and cracks) else None,
                                                cracks=cracks)

        # Set new material to selected objects
        for i in context.selected_objects:
            i.active_material = peri_mat

        return {'FINISHED'}


#############################################################################################
# MATERIAL GENERATION
#############################################################################################

def hex_to_rgb(hex):
    """ Converts color hexadecimal to RGB.
    
    :type hex: hexadecimal
    :param hex: Color hexadecimal

    :rtype: tuple of (R, G, B, o), where o = 1
    """

    b = (hex & 0xFF) / 255.0
    g = ((hex >> 8) & 0xFF) / 255.0
    r = ((hex >> 16) & 0xFF) / 255.0
    return (r, g, b, 1)

# Crack Texture
def addCrackTexture(material):
    """ Adds cracked texture (Voronoi) to the material
    :type material: Material
    :param material: Material to add cracked texture to

    :rtype: Material
    """

    # Change material name
    material.name = 'Cracked_Periosteum'

    # Get default nodes
    BSDF            = material.node_tree.nodes.get('Principled BSDF')
    tex_coord       = material.node_tree.nodes.get('Texture Coordinate')
    bump_peri       = material.node_tree.nodes.get('Bump')

    # Create shader nodes
    noise           = material.node_tree.nodes.new('ShaderNodeTexNoise')
    add             = material.node_tree.nodes.new('ShaderNodeMixRGB')
    add.blend_type  = 'ADD'
    voronoi         = material.node_tree.nodes.new('ShaderNodeTexVoronoi')
    color_ramp      = material.node_tree.nodes.new('ShaderNodeValToRGB')
    bump            = material.node_tree.nodes.new('ShaderNodeBump')
    multiply        = material.node_tree.nodes.new('ShaderNodeMixRGB')
    multiply.blend_type = 'MULTIPLY'

    # Position nodes nicely
    x_shift = -300; y_pos = -300
    multiply.location   = (x_shift, 0)
    bump.location       = (x_shift*2, y_pos)
    color_ramp.location = (x_shift*3, y_pos)
    voronoi.location    = (x_shift*4, y_pos)
    add.location        = (x_shift*5, y_pos)
    noise.location      = (x_shift*6, y_pos)
    tex_coord.location  = (x_shift*7, 0)

    # Link shader nodes
    links = material.node_tree.links
    links.new(tex_coord.outputs['Object'], noise.inputs['Vector'])
    links.new(tex_coord.outputs['Object'], add.inputs['Color1'])
    links.new(noise.outputs['Color'], add.inputs['Color2'])
    links.new(add.outputs['Color'], voronoi.inputs['Vector'])
    links.new(voronoi.outputs['Distance'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], multiply.inputs['Color2'])
    links.new(bump_peri.outputs['Normal'], multiply.inputs['Color1'])
    links.new(multiply.outputs['Color'], BSDF.inputs['Normal'])

    # Set variables
    noise.inputs['Roughness'].default_value     = 0.30
    noise.inputs['Scale'].default_value         = 3.0
    noise.inputs['Detail'].default_value        = 4.0
    noise.inputs['Distortion'].default_value    = 10.0

    add.inputs['Fac'].default_value             = 0.030

    voronoi.feature                             = 'DISTANCE_TO_EDGE'
    voronoi.voronoi_dimensions                  = '3D'
    voronoi.inputs['Scale'].default_value       = 2.0
    voronoi.inputs['Randomness'].default_value  = 0.90

    color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[0])
    color_ramp.color_ramp.elements.new(0.0)
    color_ramp.color_ramp.elements[0].color     = (0,0,0,1)
    color_ramp.color_ramp.elements[1].position  = (0.003)
    color_ramp.color_ramp.elements[1].color     = (1, 1, 1, 1)

    bump.inputs['Strength'].default_value       = 1.0
    bump.inputs['Distance'].default_value       = 0.05
    bump.invert                                 = True

    multiply.inputs['Fac'].default_value        = 0.80

    return material

# Periosteum Material
def createPeriosteumMaterial(mat=None, cracks=False):
    """ Creates bone material using nodes in Blender.
    :type mat: Material
    :param mat: Periosteum material (if it already exists)

    :type cracks: boolean
    :param cracks: True if cracks should be added

    :rtype: Material
    """

    if not mat:
        # Create a new material
        material = bpy.data.materials.new(name="Periosteum")
        material.use_nodes = True

        # Get default nodes
        BSDF            = material.node_tree.nodes.get('Principled BSDF')
        material_output = material.node_tree.nodes.get('Material Output')

        # Create shader nodes
        bump            = material.node_tree.nodes.new('ShaderNodeBump')
        noise           = material.node_tree.nodes.new('ShaderNodeTexNoise')
        mapping         = material.node_tree.nodes.new('ShaderNodeMapping')
        tex_coord       = material.node_tree.nodes.new('ShaderNodeTexCoord')

        # Position nodes nicely
        x_shift = -300; y_pos = 200
        bump.location       = (x_shift*2, y_pos)
        noise.location      = (x_shift*3, y_pos)
        mapping.location    = (x_shift*4, y_pos)
        tex_coord.location  = (x_shift*5, y_pos)

        # Link shader nodes
        links = material.node_tree.links
        links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
        links.new(noise.outputs['Color'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], BSDF.inputs['Normal'])
        links.new(BSDF.outputs['BSDF'], material_output.inputs['Surface'])

        # Set variables
        noise.inputs['Roughness'].default_value = 0.938
        noise.inputs['Scale'].default_value     = 6.200
        noise.inputs['Detail'].default_value    = 12.60

        bump.inputs['Strength'].default_value   = 0.408
        bump.inputs['Distance'].default_value   = 1.0

        BSDF.inputs['Base Color'].default_value = hex_to_rgb(0xFAFAE4) # Ivory: 0xe3dac9

        if cracks:
            material = addCrackTexture(material)    # Add cracked texture

    else:
        material = addCrackTexture(mat)     # Add cracked texture to copied periosteum material
    
    return material


#############################################################################################
# PROPERTIES
#############################################################################################

class TextureSettings(PropertyGroup):
    cracks: BoolProperty(
        name = "",
        description = "Add cracks to bone texture",
        default = False,
    )


#############################################################################################
# REGISTER ADD-ON
#############################################################################################

classes = (
    PERIOSTEUM_PT_Class,
    PERIOSTEUM_Texturing,
    TextureSettings,
    )

reg, unreg = bpy.utils.register_classes_factory(classes)

# Register
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.periosteum = PointerProperty(type=TextureSettings)

# Unregister
def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.periosteum

if __name__ == "__main__":
    register()

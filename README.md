## Description
This Blender add-on generates human rib bones based on subject's age, sex, height, and weight. Example below:

![alt text](https://github.com/ccxxgao/HumanRibModeler/blob/master/Rendered%20Images/F_35_1.65_75_anterior.png?raw=true)


## Add-On Setup
### __Install Third-Party Python Libraries (scipy, pandas):__
1.  In Blender's Python console, run:
    ```
    import sys
    sys.exec_prefix
    ```

    This should return the location of the Blender's python package, e.g.,
    ```
    /Applications/Blender.app/Contents/Resources/2.90/python`
    ```
2. In a regular terminal, run:
    ```
    cd /path/to/blender/python/bin  # Don't forget to add '/bin' to the path
    ls
    ```
    This should output the Python executable used by the Blender, e.g.,
    ```
    python3.5
    ```

3. To enable pip and install the packages:
    ```
    ./python3.5m -m ensurepip
    ./python3.5m -m pip install scipy
    ./python3.5m -m pip install pandas
    ./python3.5m -m pip install numpy   # the most recent version of Blender should come preinstalled with this
    ```
<br>
The packages should be installed now, and you should be able to run my plugin code in Blender!

<br><br>
### __Load Rib and Periosteum Plugins in Blender:__
1. Open midterm_project.blend

2. Go to 'Scripting' tab

3. `rib.py` and `periosteum.py` should already be loaded in the Text Editor.
    - If not, open the two files, which should be located in the 'Source Code' folder within my project folder

4. Change the path to the folder containing the linear regression parameters. In rib.py, change the `path_to_data` variable on line 45 to the 'RibLinReg' folder in my project folder, e.g.,
    ```
    path_to_data = '/Users/cecegao/Downloads/Gao_Cecily_Graphics_Project_1/RibLinReg/'
    ```

5. Run both scripts.

6. Two tabs titled 'Human Rib' and 'Periosteum' should appear in the Properties Panel.

<br>

You should be able to use the plugins now!

<br><br>
### __How to Use Rib Plugin:__
1. Toggle to the 'Human Rib' tab in the Preferences Panel.

2. Change the input parameters to the subject of interest. Change the subject's demographics (sex, age, height weight). You can also toggle to the 'Custom' mode to modify individual parameters.

3. Change the rendering settings. Choose whether you want to generate the full set or a single rib or just the outline (no lofting).

4. Click 'Create Rib(cage)' button at the bottom to generate your rib(s)!

<br><br>
### __How to Use Periosteum Plugin:__
1. Make sure Blender's built-in 'Node' add-ons are enabled:
    1. Go to Blender Preferences: Edit > Preferences
    2. Search 'Node'
    3. Check all three Node add-ons

2. Toggle to the 'Periosteum' tab in the Preferences Panel.

3. Select the objects you would like to texture with the periosteum material.

4. Check 'Cracks' if you want to add the cracks in addition to the Periosteum texture.

5. Click 'Texture Object(s)' to add the bone texture to your selected objects!

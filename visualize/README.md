# shape-completion-visualization

Place for shape completion visualization code

## Installation

```bash
pip install -r requirements.txt
```

Clone BlenderProc repository and install it as a Python package.

```bash
git clone git@github.com:DLR-RM/BlenderProc.git
cd BlenderProc
pip install -e .
```

Comment the entire content of `BlenderProc/blenderproc/__init__.py`.
Comment the following lines (34-36) inside `BlenderProc/blenderproc/python/utility/Initializer.py`:

```python
if bpy.context.preferences.view.language != "en_US":                                                                                                                                                     
    print("Setting blender language settings to english during this run")                                                                                                                                
    bpy.context.preferences.view.language = "en_US"
```
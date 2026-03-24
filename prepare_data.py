from roboflow import Roboflow
rf = Roboflow(api_key="lNH6mEKbeuqv27ZQZHNH")
project = rf.workspace("new-workspace-5uval").project("human-crowd-vbdc9")
version = project.version(1)
dataset = version.download("yolov8", location="./roboflow_downloads", overwrite=False)

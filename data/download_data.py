from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("WORKSPACE_NAME").project("PROJECT_NAME")
version = project.version(VERSION_NUMBER)
dataset = version.download("FORMAT_NAME")

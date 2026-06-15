"""
Kaggle one-file launcher for FacePJ.

Run inside a Kaggle notebook after cloning the repo:

    !python kaggle_run.py

The script starts the Gradio app with share=True so Kaggle prints a public gradio.live link.
"""

import os
import runpy

os.environ["KAGGLE_KERNEL_RUN_TYPE"] = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["PORT"] = os.environ.get("PORT", "7860")

runpy.run_path("app_gradio.py", run_name="__main__")

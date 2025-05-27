# AXZ viewer: a web application to visualize unprocessed AFM-IR data

[![Application][application-shield]][application-link]
[![CC BY 4.0][license-shield]][license-link]
[![DOI (repository)][doi-repo-shield]][doi-repo-link]
![DOI (paper)][doi-paper-shield]

This repository contains the source code for the web application published in **Duverger et al. (in revision)**, allowing users to view unprocessed AFM-IR data and metadata from .axz files. The application is accessible at [anasys-python-tools-gui.streamlit.app][application-link] and includes an example dataset ([source/example.axz](source/example.axz)) and demonstration video ([source/demo.mp4](source/demo.mp4)).

This code relies on the [anasys-python-tools](https://github.com/GeorgRamer/anasys-python-tools) package, co-published in Duverger et al. (in revision), which provides the necessary functions to read and process AFM-IR data files. The application is built using [Streamlit](https://streamlit.io/), a Python library for creating web applications.

**Known limitations:** This app is not intended to replace data analysis
software. Instead, it is designed to provide a quick overview of AFM-IR (meta-)data,
to data quality, and to foster the sharing of data between researchers. Furthermore,
this application is not yet compatible with IconIR data files. If you experience
any other issues, please open an issue on GitHub.

## Local installation

The application can be ran locally by executing the following commands, assuming git and Python (version 3.12) are installed:

```
git clone https://github.com/wduverger/anasys-python-tools-gui
cd anasys-python-tools-gui
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m streamlit run source/app.py 
```

## Citing this work

This work is licensed under the Creative Commons Attribution 4.0 International License [(CC BY 4.0)][license-link]. If you use this code in your work, please cite the paper, once published. 

When explicitly referencing this code, you can use the DOI assigned to the repository. For example:
> Duverger, Ramer, Louros, Schymkowitz & Rousseau. AXZ Viewer. GitHub repository. (2025) doi:[10.6084/m9.figshare.27991898][doi-repo-link].

[license-link]:       http://creativecommons.org/licenses/by/4.0/
[license-image]:      https://i.creativecommons.org/l/by/4.0/88x31.png
[license-shield]:     https://img.shields.io/badge/License-CC%20BY%204.0-deepskyblue.svg

[application-link]: https://anasys-python-tools-gui.streamlit.app
[application-shield]: https://img.shields.io/badge/Open_on_Streamlit-tomato

[doi-repo-shield]:  https://img.shields.io/badge/DOI_(repository)-10.6084/m9.figshare.27991898-deepskyblue
[doi-repo-link]: https://doi.org/10.6084/m9.figshare.27991898

[doi-paper-shield]:  https://img.shields.io/badge/DOI_(paper)-pending-gainsboro

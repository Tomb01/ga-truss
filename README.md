# 2D lattice truss optimization with genetic algorithm

## General information

Hi, I am **Tommaso Bari**, a student at University of Padua.
This repository contains all the material related to my Bachelor's degree thesis. he goal of the thesis was to develop a genetic algorithm for generating 2D lattice structures under different constraints and loads.

The algorithm is written in python with a small amout of external libraries: primary _numpy_ and _matplotlib_.

All the source code is freely available under the [GNU General Public License v3.0](LICENSE.md).
If you use my work in your academic project, please [cite me](CITATION.cff)!

* [Final thesis relation in PDF format (italian only)](thesis.pdf)
* [Final thesis presentation uploaded on University thesis archive (italian only)](https://thesis.unipd.it/handle/20.500.12608/72294)

## Setup

- Ensure Python is installed on your system. You can download it from [Python's official website](https://www.python.org/downloads/).
- Install `pip`, Python's package installer, which usually comes with Python.

1. **Create a Virtual Environment**  
    on Windows
    ```bash
    python -m venv env
    ```
    On macOS and Linux:
    ```bash
    python3 -m venv env
    ```

2. **Activate the Virtual Environment**  
    On Windows:
    ```bash
    env\Scripts\activate
    ```
    On macOS and Linux:
    ```bash
    source env/bin/activate
    ```
    After activation, you will see the environment name (e.g., `(env)`) in your terminal prompt.

3. **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Choose a Template:**  
   Copy one of the sample templates from the `examples` folder.

2. **Customize Parameters:**  
   Edit the structure and evolution parameters to suit your needs.

3. **Configure the Simulation:**  
   Update the setup section in `simulate.py` according to your template and desired configuration.

4. **Run the Simulation:**  
   Open a terminal and execute the following command:
   ```bash
   python simulate.py <simulation_folder> <run_count>
   ```
   For example, to run the simulation 10 times and save the results in a folder called output:
   ```bash
   python simulate.py output 10
   ```
5. **Enjoy the Results:**
    Your simulations will run, and the results will be saved in the specified folder.
    Sit back and watch the magic happen!


